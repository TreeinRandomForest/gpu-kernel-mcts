"""H100 BF16 GEMM: tuned Triton kernel (row-major A, column-major B, row-major C)."""

import triton
import triton.language as tl


@triton.jit
def bf16_gemm_kernel(A, B, C, M, N, K,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)

    # A is row-major [M, K]
    a_ptrs = A + offs_m[:, None] * K + offs_k[None, :]
    # B is column-major [K, N]: element (k, n) lives at B + n*K + k
    b_ptrs = B + offs_k[:, None] + offs_n[None, :] * K

    a_ptrs = tl.max_contiguous(tl.multiple_of(a_ptrs, (1, BK)), (1, BK))
    b_ptrs = tl.max_contiguous(tl.multiple_of(b_ptrs, (BK, 1)), (BK, 1))

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc)
        a_ptrs += BK
        b_ptrs += BK

    c = acc.to(tl.bfloat16)
    c_ptrs = C + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, c)


def launch(A, B, C, M, N, K):
    BM, BN, BK = 128, 256, 64
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN), 1)
    return bf16_gemm_kernel[grid](
        A, B, C, M, N, K, BM, BN, BK,
        num_warps=8, num_stages=3
    )