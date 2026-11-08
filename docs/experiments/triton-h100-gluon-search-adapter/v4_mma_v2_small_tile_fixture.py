import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import mma_v2


@gluon.jit
def gemm(A, B, C, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
         BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr):
    pid_m = gl.program_id(0)
    pid_n = gl.program_id(1)
    off_m = pid_m * BM
    off_n = pid_n * BN

    # A tile: [BM, BK], K contiguous (row-major A)
    a_layout: gl.constexpr = gl.BlockedLayout([1, 16], [4, 8], [2, 2], [0, 1])
    # B tile: [BK, BN], K contiguous (column-major B: B[k,n] at B+n*K+k)
    b_layout: gl.constexpr = gl.BlockedLayout([16, 1], [8, 4], [2, 2], [1, 0])
    mma_layout: gl.constexpr = gl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[4, 1],
                                                          instr_shape=[16, 8])
    a_op: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma_layout, k_width=2)
    b_op: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mma_layout, k_width=2)
    # C tile: [BM, BN], N contiguous (row-major C)
    out_layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [4, 1], [0, 1])

    # row-axis offsets slice away dim 1; column-axis offsets slice away dim 0
    row_a = off_m + gl.arange(0, BM, gl.SliceLayout(1, a_layout))
    k_a = gl.arange(0, BK, gl.SliceLayout(0, a_layout))
    k_b = gl.arange(0, BK, gl.SliceLayout(1, b_layout))
    col_b = off_n + gl.arange(0, BN, gl.SliceLayout(0, b_layout))
    row_c = off_m + gl.arange(0, BM, gl.SliceLayout(1, out_layout))
    col_c = off_n + gl.arange(0, BN, gl.SliceLayout(0, out_layout))

    acc = gl.zeros((BM, BN), gl.float32, layout=mma_layout)

    for off_k in range(0, K, BK):
        ka = off_k + k_a
        kb = off_k + k_b
        a_ptrs = A + row_a[:, None] * K + ka[None, :]
        b_ptrs = B + col_b[None, :] * K + kb[:, None]
        a = gl.load(a_ptrs, mask=row_a[:, None] < M, other=0.0)
        b = gl.load(b_ptrs, mask=col_b[None, :] < N, other=0.0)
        a_d = gl.convert_layout(a, a_op)
        b_d = gl.convert_layout(b, b_op)
        acc = mma_v2(a_d, b_d, acc)

    c = gl.convert_layout(acc, out_layout).to(gl.bfloat16)
    out_ptrs = C + row_c[:, None] * N + col_c[None, :]
    gl.store(out_ptrs, c,
             mask=(row_c[:, None] < M) & (col_c[None, :] < N))


def launch(A, B, C, M, N, K):
    BM, BN, BK = 32, 64, 64
    return gemm[(triton.cdiv(M, BM), triton.cdiv(N, BN))](
        A, B, C, M, N, K, BM, BN, BK,
        num_warps=4, num_stages=3
    )
