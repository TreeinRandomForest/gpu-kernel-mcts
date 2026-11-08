import triton
import triton.language as tl


@triton.jit
def bf16_gemm_triton(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    row = pid_m * BM + tl.arange(0, BM)
    col = pid_n * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)

    acc = tl.zeros((BM, BN), dtype=tl.float32)

    # Base pointers for the tiles.
    a_ptrs = A + row[:, None] * K + kk[None, :]
    # B is [K,N] column-major: element (k, n) is at B[k + n*K].
    # Load as (BN, BK) so the contiguous k dimension is the fast axis.
    b_ptrs = B + col[:, None] * K + kk[None, :]

    for k in range(0, tl.cdiv(K, BK)):
        a = tl.load(a_ptrs, mask=(row[:, None] < M) & (kk[None, :] + k * BK < K), other=0.0)
        bT = tl.load(b_ptrs, mask=(col[:, None] < N) & (kk[None, :] + k * BK < K), other=0.0)
        acc += tl.dot(a, tl.trans(bT))
        a_ptrs += BK
        b_ptrs += BK

    c_ptrs = C + row[:, None] * N + col[None, :]
    tl.store(c_ptrs, acc.to(tl.bfloat16),
             mask=(row[:, None] < M) & (col[None, :] < N))


def launch(A, B, C, M, N, K):
    bm, bn, bk = 128, 256, 64
    grid = (triton.cdiv(M, bm), triton.cdiv(N, bn))
    return bf16_gemm_triton[grid](
        A, B, C, M, N, K, bm, bn, bk,
        num_warps=8, num_stages=4
    )