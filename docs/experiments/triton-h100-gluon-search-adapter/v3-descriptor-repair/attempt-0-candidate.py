import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def bf16_gemm_tma(a_desc, b_desc, C, M, N, K,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    off_m = pid_m * BM
    off_n = pid_n * BN
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for off_k in range(0, K, BK):
        a = a_desc.load([off_m, off_k])
        b = b_desc.load([off_n, off_k])
        acc += tl.dot(a, tl.trans(b))
    row = off_m + tl.arange(0, BM)
    col = off_n + tl.arange(0, BN)
    tl.store(C + row[:, None] * N + col[None, :],
             acc.to(tl.bfloat16),
             (row[:, None] < M) & (col[None, :] < N))


@triton.jit
def bf16_gemm_fallback(A, B, C, M, N, K,
                       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + kk
        a = tl.load(A + row[:, None] * K + k[None, :],
                    (row[:, None] < M) & (k[None, :] < K), other=0)
        b = tl.load(B + k[:, None] + col[None, :] * K,
                    (k[:, None] < K) & (col[None, :] < N), other=0)
        acc += tl.dot(a, b)
    tl.store(C + row[:, None] * N + col[None, :], acc.to(tl.bfloat16),
             (row[:, None] < M) & (col[None, :] < N))


def launch(A, B, C, M, N, K):
    BM, BN, BK = 64, 128, 64
    if (M % BM == 0 and N % BN == 0 and K % BK == 0
            and A.data_ptr() % 16 == 0 and B.data_ptr() % 16 == 0
            and C.data_ptr() % 16 == 0):
        a_desc = TensorDescriptor(A, shape=[M, K], strides=[K, 1],
                                  block_shape=[BM, BK])
        # B is [K,N] column-major == zero-copy contiguous B.T [N,K].
        b_desc = TensorDescriptor(B, shape=[N, K], strides=[K, 1],
                                  block_shape=[BN, BK])
        return bf16_gemm_tma[(M // BM, N // BN)](
            a_desc, b_desc, C, M, N, K, BM, BN, BK,
            num_warps=8, num_stages=3
        )
    else:
        return bf16_gemm_fallback[(triton.cdiv(M, BM), triton.cdiv(N, BN))](
            A, B, C, M, N, K, BM, BN, BK, num_warps=8
        )