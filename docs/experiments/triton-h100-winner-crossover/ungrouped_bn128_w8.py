"""Optimized Triton H100 BF16 GEMM (A row-major, B column-major [K,N])."""

import triton
import triton.language as tl


@triton.jit
def bf16_gemm_triton(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                     GROUP_M: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    row = pid_m * BM + tl.arange(0, BM)
    col = pid_n * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)

    # Precompute base pointers.
    a_ptrs = A + row[:, None] * K + kk[None, :]
    # B is [K,N] column-major: K stride is 1, N stride is K.
    b_ptrs = B + kk[:, None] + col[None, :] * K

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        a = tl.load(a_ptrs,
                    (row[:, None] < M) & (k * BK + kk[None, :] < K),
                    other=0.0)
        b = tl.load(b_ptrs,
                    (k * BK + kk[:, None] < K) & (col[None, :] < N),
                    other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BK
        b_ptrs += BK

    c_ptrs = C + row[:, None] * N + col[None, :]
    tl.store(c_ptrs, acc.to(tl.bfloat16),
             (row[:, None] < M) & (col[None, :] < N))


def launch(A, B, C, M, N, K):
    BM, BN, BK, GROUP_M = 128, 128, 64, 8
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    return bf16_gemm_triton[grid](
        A, B, C, M, N, K, BM, BN, BK, GROUP_M,
        num_warps=8, num_stages=3
    )