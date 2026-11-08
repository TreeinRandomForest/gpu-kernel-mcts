"""Triton H100 BF16 GEMM: 128x128x64 tile, 8 warps, 3 stages, pointer-advancing loop."""

import triton
import triton.language as tl


@triton.jit
def bf16_gemm_triton(A, B, C, M, N, K,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)

    # K-contiguous physical addressing: A row stride K, B column stride K.
    a_ptrs = A + row[:, None] * K + kk[None, :]
    b_ptrs = B + col[None, :] * K + kk[:, None]

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BK)):
        a = tl.load(a_ptrs, mask=(row[:, None] < M) & (kk[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(kk[:, None] < K) & (col[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BK
        b_ptrs += BK

    c_ptrs = C + row[:, None] * N + col[None, :]
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=(row[:, None] < M) & (col[None, :] < N))


def launch(A, B, C, M, N, K):
    BM, BN, BK = 128, 256, 64
    return bf16_gemm_triton[(triton.cdiv(M, BM), triton.cdiv(N, BN))](
        A, B, C, M, N, K, BM, BN, BK, num_warps=8, num_stages=3
    )