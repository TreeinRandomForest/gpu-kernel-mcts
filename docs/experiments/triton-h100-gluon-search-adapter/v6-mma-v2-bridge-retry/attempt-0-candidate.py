import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import mma_v2


@gluon.jit
def bf16_gemm_gluon(A, B, C, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
                    BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr):
    pid_m = gl.program_id(0)
    pid_n = gl.program_id(1)

    a_layout: gl.constexpr = gl.BlockedLayout([1, 16], [4, 8], [2, 2], [1, 0])
    b_layout: gl.constexpr = gl.BlockedLayout([16, 1], [8, 4], [2, 2], [0, 1])
    out_layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
    mma_layout: gl.constexpr = gl.NVMMADistributedLayout(version=[2, 0],
                                                         warps_per_cta=[4, 1],
                                                         instr_shape=[16, 8])
    a_op: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma_layout, k_width=2)
    b_op: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mma_layout, k_width=2)

    row_a = pid_m * BM + gl.arange(0, BM, gl.SliceLayout(1, a_layout))
    k_a = gl.arange(0, BK, gl.SliceLayout(0, a_layout))
    k_b = gl.arange(0, BK, gl.SliceLayout(1, b_layout))
    col_b = pid_n * BN + gl.arange(0, BN, gl.SliceLayout(0, b_layout))
    row_c = pid_m * BM + gl.arange(0, BM, gl.SliceLayout(1, out_layout))
    col_c = pid_n * BN + gl.arange(0, BN, gl.SliceLayout(0, out_layout))

    row_a = row_a[:, None]
    col_c_mask = col_c[None, :]

    acc = gl.zeros((BM, BN), gl.float32, layout=mma_layout)

    for off_k in range(0, K, BK):
        koff_a = (off_k + k_a)[None, :]
        a_ptrs = A + row_a * K + koff_a
        a_mask = (row_a < M) & (koff_a < K)
        a = gl.load(a_ptrs, mask=a_mask, other=0.0)

        koff_b = (off_k + k_b)[:, None]
        col_b2 = col_b[None, :]
        b_ptrs = B + col_b2 * K + koff_b
        b_mask = (col_b2 < N) & (koff_b < K)
        b = gl.load(b_ptrs, mask=b_mask, other=0.0)

        a = gl.convert_layout(a.to(gl.bfloat16), a_op)
        b = gl.convert_layout(b.to(gl.bfloat16), b_op)
        acc = mma_v2(a, b, acc)

    out = gl.convert_layout(acc, out_layout)
    c_ptrs = C + row_c[:, None] * N + col_c_mask
    c_mask = (row_c[:, None] < M) & (col_c_mask < N)
    gl.store(c_ptrs, out.to(gl.bfloat16), mask=c_mask)


def launch(A, B, C, M, N, K):
    BM, BN, BK = 64, 128, 64
    return bf16_gemm_gluon[(triton.cdiv(M, BM), triton.cdiv(N, BN))](
        A, B, C, M, N, K, BM, BN, BK,
        num_warps=4, num_stages=3
    )