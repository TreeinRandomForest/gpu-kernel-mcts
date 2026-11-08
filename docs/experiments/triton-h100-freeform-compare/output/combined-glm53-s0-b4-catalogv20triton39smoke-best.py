import triton
import triton.language as tl


@triton.jit
def bf16_gemm_triton(A, B, C, M, N, K,
                      BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)

    a_ptrs = A + row[:, None] * K + kk[None, :]
    # B is [K,N] column-major: B[k,n] at B + n*K + k
    b_ptrs = B + kk[:, None] + col[None, :] * K

    for k in range(0, tl.cdiv(K, BK)):
        k_idx = k * BK + kk
        a = tl.load(a_ptrs,
                    (row[:, None] < M) & (k_idx[None, :] < K), other=0)
        b = tl.load(b_ptrs,
                    (k_idx[:, None] < K) & (col[None, :] < N), other=0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BK
        b_ptrs += BK

    tl.store(C + row[:, None] * N + col[None, :], acc.to(tl.bfloat16),
             (row[:, None] < M) & (col[None, :] < N))


def launch(A, B, C, M, N, K):
    BM, BN, BK = 128, 128, 64
    return bf16_gemm_triton[(triton.cdiv(M, BM), triton.cdiv(N, BN))](
        A, B, C, M, N, K, BM, BN, BK, num_warps=8, num_stages=3
    )