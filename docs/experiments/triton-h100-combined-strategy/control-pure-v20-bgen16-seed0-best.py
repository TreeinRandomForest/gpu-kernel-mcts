import triton
import triton.language as tl


@triton.jit
def bf16_gemm_triton(A, B, C, M, N: tl.constexpr, K: tl.constexpr,
                      BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)

    a_ptrs = A + row[:, None] * K + kk[None, :]
    # B is [K,N] column-major: B[k,n] at B + n*K + k
    b_ptrs = B + kk[:, None] + col[None, :] * K

    for k in range(0, tl.cdiv(K, BK)):
        a = tl.load(a_ptrs,
                    (row[:, None] < M) & (kk[None, :] + k * BK < K), other=0)
        b = tl.load(b_ptrs,
                    (kk[:, None] + k * BK < K) & (col[None, :] < N), other=0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BK
        b_ptrs += BK

    tl.store(C + row[:, None] * N + col[None, :], acc.to(tl.bfloat16),
             (row[:, None] < M) & (col[None, :] < N))


def launch(A, B, C, M, N, K):
    BM, BN, BK = 128, 256, 64
    return bf16_gemm_triton[(triton.cdiv(M, BM), triton.cdiv(N, BN))](
        A, B, C, M, N, K, BM, BN, BK, num_warps=8, num_stages=3
    )