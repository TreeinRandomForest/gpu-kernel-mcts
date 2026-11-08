import triton
import triton.language as tl


@triton.jit
def bf16_gemm_triton(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)

    # A is [M,K] row-major: A[m,k] at A + m*K + k -> row stride K, k stride 1
    a_ptrs = A + row[:, None] * K + kk[None, :]
    # B is [K,N] column-major: B[k,n] at B + n*K + k -> column stride K, k stride 1
    b_ptrs = B + col[None, :] * K + kk[:, None]

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        a = tl.load(a_ptrs,
                    mask=(row[:, None] < M) & (kk[None, :] + k * BK < K),
                    other=0.0)
        b = tl.load(b_ptrs,
                    mask=(kk[:, None] + k * BK < K) & (col[None, :] < N),
                    other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BK
        b_ptrs += BK

    tl.store(C + row[:, None] * N + col[None, :],
             acc.to(tl.bfloat16),
             mask=(row[:, None] < M) & (col[None, :] < N))


def launch(A, B, C, M, N, K):
    BM, BN, BK = 128, 256, 64
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    return bf16_gemm_triton[grid](
        A, B, C, M, N, K, BM, BN, BK, num_warps=8, num_stages=3
    )