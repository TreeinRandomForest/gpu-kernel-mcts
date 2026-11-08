import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import mma_v2


@gluon.jit
def gemm(A, B, C, M, N: gl.constexpr, K: gl.constexpr,
         BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    off_m = pid_m * BM
    off_n = pid_n * BN

    # Layouts
    a_layout = gl.BlockedLayout([1, 16], [4, 4], [2, 1], [0, 1], [1, 1])
    b_layout = gl.BlockedLayout([1, 16], [4, 4], [1, 2], [0, 1], [1, 1])
    mma_layout = gl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[4, 1],
                                           instr_shape=[16, 8])
    a_op = gl.DotOperandLayout(operand_index=0, parent=mma_layout, k_width=2)
    b_op = gl.DotOperandLayout(operand_index=1, parent=mma_layout, k_width=2)
    c_layout = gl.BlockedLayout([1, 16], [4, 4], [1, 2], [0, 1], [1, 1])

    m_row = gl.SliceLayout(0, a_layout)
    k_col = gl.SliceLayout(1, a_layout)
    k_row = gl.SliceLayout(0, b_layout)
    n_col = gl.SliceLayout(1, b_layout)

    row = off_m + gl.arange(0, BM, m_row)
    col = off_n + gl.arange(0, BN, n_col)

    acc = gl.zeros((BM, BN), gl.float32, layout=mma_layout)

    for off_k in gl.range(0, K, BK, loop_unroll_factor=1):
        k_m = off_k + gl.arange(0, BK, k_col)
        k_n = off_k + gl.arange(0, BK, k_row)
        a_ptrs = A + row[:, None] * K + k_m[None, :]
        b_ptrs = B + col[None, :] * K + k_n[:, None]
        a = gl.load(a_ptrs, gl.SliceLayout(0, a_layout), mask=row[:, None] < M,
                    other=0.0)
        b = gl.load(b_ptrs, gl.SliceLayout(0, b_layout), mask=col[None, :] < N,
                    other=0.0)
        acc = mma_v2(gl.convert_layout(a, a_op), gl.convert_layout(b, b_op), acc)

    c = gl.convert_layout(acc, c_layout)
    out_ptrs = C + row[:, None] * N + col[None, :]
    gl.store(out_ptrs, c.to(gl.bfloat16), gl.SliceLayout(0, c_layout),
             mask=row[:, None] < M)
The above is invalid; here's the final file:

```python
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import mma_v2


@gluon.jit
def gemm(A, B, C, M, N: gl.constexpr, K: gl.constexpr,
         BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    off_m = pid_m * BM
    off_n = pid_n * BN

    a_layout = gl.BlockedLayout([1, 16], [4, 4], [2, 1], [0, 1], [1, 1])
    b_layout = gl.BlockedLayout([1, 16], [4, 4], [1, 2], [0, 1], [1, 1])
    mma_layout = gl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[4, 1],
                                           instr_shape=[16, 8])
    a_op = gl.DotOperandLayout(operand_index=0, parent=mma_layout, k_width=2)
    b_op = gl.DotOperandLayout(operand_index=1, parent=mma_layout, k_width=2)
    c_layout = gl.BlockedLayout([1, 16], [4, 4], [1, 2], [0, 1], [1, 1])

    row = off_m + gl.arange(0, BM, gl.SliceLayout(0, a_layout))
    col = off_n + gl.arange(0, BN, gl.SliceLayout(1, b_layout))

    acc = gl.zeros((BM, BN), gl.float32, layout=mma_layout)

    for off_k in gl.range(0, K, BK, loop_unroll_factor=1):
        k_a = off_k + gl.arange(0, BK, gl.SliceLayout(1, a_layout))
        k_b = off_k + gl.arange(0, BK, gl.SliceLayout(0, b_layout))
        a_ptrs = A + row[:, None] * K + k_a[None, :]
        b_ptrs = B + col[None, :] * K + k_b[:, None]
        a = gl.load(a_ptrs, a_layout, mask=row[:, None] < M, other=0.0)
        b = gl.load(b_ptrs, b_layout, mask=col[None, :] < N, other=0.0)
        acc = mma_v2(gl.convert_layout(a, a_op), gl.convert_layout(b, b_op), acc)

    c = gl.convert_layout(acc, c_layout).to(gl.bfloat16)
    out_ptrs = C + row[:, None] * N + col[None, :]
    gl.store(out_ptrs, c, c_layout, mask=row[:, None] < M)


def launch(A, B, C, M, N, K):
    BM, BN, BK = 64, 128, 64
    return gemm[(triton.cdiv(M, BM), triton.cdiv(N, BN))](
        A, B, C, M, N, K, BM, BN, BK,
        num_warps=4, num_stages=3
    )
```