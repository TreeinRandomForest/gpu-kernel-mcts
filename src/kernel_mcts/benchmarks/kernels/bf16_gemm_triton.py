"""Triton root for the existing H100 BF16 GEMM workload."""

import triton
import triton.language as tl


@triton.jit
def bf16_gemm_triton(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + kk
        a = tl.load(A + row[:, None] * K + k[None, :],
                    (row[:, None] < M) & (k[None, :] < K), other=0)
        # B is [K,N] column-major, so its K stride is 1 and N stride is K.
        b = tl.load(B + k[:, None] + col[None, :] * K,
                    (k[:, None] < K) & (col[None, :] < N), other=0)
        acc += tl.dot(a, b)
    tl.store(C + row[:, None] * N + col[None, :], acc.to(tl.bfloat16),
             (row[:, None] < M) & (col[None, :] < N))


def launch(A, B, C, M, N, K):
    bm, bn, bk = 32, 64, 32
    return bf16_gemm_triton[(triton.cdiv(M, bm), triton.cdiv(N, bn))](
        A, B, C, M, N, K, bm, bn, bk, num_warps=4
    )
