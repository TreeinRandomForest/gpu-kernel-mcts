import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def bf16_gemm_tma(a_desc, b_desc, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    row = pid_m * BM + tl.arange(0, BM)
    col = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for start in range(tl.cdiv(K, BK)):
        off_k = start * BK
        a = a_desc.load([pid_m * BM, off_k])
        b = b_desc.load([pid_n * BN, off_k])
        acc += tl.dot(a, tl.trans(b))
    tl.store(C + row[:, None] * N + col[None, :], acc.to(tl.bfloat16),
             (row[:, None] < M) & (col[None, :] < N))


def launch(A, B, C, M, N, K):
    BM, BN, BK = 64, 128, 64
    a_desc = TensorDescriptor(A, shape=[M, K], strides=[K, 1], block_shape=[BM, BK])
    b_desc = TensorDescriptor(B, shape=[N, K], strides=[K, 1], block_shape=[BN, BK])
    return bf16_gemm_tma[(triton.cdiv(M, BM), triton.cdiv(N, BN))](
        a_desc, b_desc, C, M, N, K, BM, BN, BK, num_warps=8
    )