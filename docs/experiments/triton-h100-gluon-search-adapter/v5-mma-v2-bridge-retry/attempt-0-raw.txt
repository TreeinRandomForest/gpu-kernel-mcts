import triton
import triton.language as tl
import gluon
import gluon.language as gl
from triton.experimental.gluon.language.nvidia.ampere import (
    mma_v2,
    NVMMADistributedLayout,
    DotOperandLayout,
)


@gluon.jit
def bf16_gemm_gluon(A, B, C, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
                    BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr):
    a_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 16], threads_per_warp=[4, 8],
        warps_per_cta=[2, 2], order=[1, 0])
    b_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[16, 1], threads_per_warp=[8, 4],
        warps_per_cta=[2, 2], order=[0, 1])
    c_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8], threads_per_warp=[4, 8],
        warps_per_cta=[4, 1], order=[1, 0])
    mma_layout: gl.constexpr = NVMMADistributedLayout(
        warps_per_cta=[4, 1], instr_shape=[16, 8])
    a_op: gl.constexpr = gl.DotOperandLayout(0, parent=mma_layout, k_width=2)
    b_op: gl.constexpr = gl.DotOperandLayout(1, parent=mma_layout, k_width=2)

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    row = pid_m * BM + tl.arange(0, BM)
    col = pid_n * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)

    row_a: gl.constexpr = gl.SliceLayout(1, a_layout)
    k_a: gl.constexpr = gl.SliceLayout(0, a_layout)
    k_b: gl.constexpr = gl.SliceLayout(1, b_layout)
    col_b: gl.constexpr = gl.SliceLayout(0, b_layout)

    row_m = tl.reshape(row, (BM, 1))
    col_m = tl.reshape(col, (1, BN))

    acc = gl.zeros((BM, BN), mma_layout, tl.float32)

    for off_k in range(0, K, BK):
        a_off = tl.reshape(tl.convert_layout(row, row_a), (BM, 1)) * K \
            + tl.reshape(tl.convert_layout(k + off_k, k_a), (1, BK))
        b_off = tl.reshape(tl.convert_layout(k + off_k, k_b), (BK, 1)) \
            + tl.reshape(tl.convert_layout(col, col_b), (1, BN)) * K
        a = gl.load(A + a_off,
                    mask=(row[:, None] < M) & (k[None, :] + off_k < K),
                    other=0.0)
        b = gl.load(B + b_off,
                    mask=(k[:, None] + off_k < K) & (col[None, :] < N),
                    other=0.0)
        a = gl.convert_layout(a, a_op)
        b = gl.convert_layout(b, b_op)
        acc = mma_v2(a, b, acc)

    c = gl.convert_layout(acc.to(tl.bfloat16), c_layout)
    gl.store(C + row_m * N + col_m, c,
             mask=(row[:, None] < M) & (col[None, :] < N))


def launch(A, B, C, M, N, K):
    BM, BN, BK = 64, 128, 64
    return bf16_gemm_gluon[(triton.cdiv(M, BM), triton.cdiv(N, BN))](
        A, B, C, M, N, K, BM, BN, BK,
        num_warps=4, num_stages=3
    )