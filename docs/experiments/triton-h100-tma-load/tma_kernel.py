"""Fast B schedule with only its A/B global-to-shared loads changed to TMA.

The descriptor API here is the one shipped in pinned Triton 3.3.1. B is
physically [N, K] row-major and logically [K, N] column-major.
"""

import triton
import triton.language as tl
from triton.tools.experimental_descriptor import create_2d_tma_descriptor


@triton.jit
def bf16_gemm_tma(a_desc, b_desc, C, M: tl.constexpr, N: tl.constexpr,
                  K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                  BK: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    row = pid_m * BM + tl.arange(0, BM)
    col = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        a = tl._experimental_descriptor_load(
            a_desc, [pid_m * BM, k * BK], [BM, BK], tl.bfloat16)
        b_physical = tl._experimental_descriptor_load(
            b_desc, [pid_n * BN, k * BK], [BN, BK], tl.bfloat16)
        acc = tl.dot(a, b_physical.T, acc)

    c = acc.to(tl.bfloat16)
    c_ptrs = C + row[:, None] * N + col[None, :]
    tl.store(c_ptrs, c, mask=(row[:, None] < M) & (col[None, :] < N))


def prepare(A, B, C, M, N, K):
    if (A.shape != (M, K) or B.shape != (K, N) or C.shape != (M, N)
            or A.stride() != (K, 1) or B.stride() != (1, K)
            or C.stride() != (N, 1)):
        raise ValueError("expected contiguous A/C and transposed-contiguous B")
    if M % 128 or N % 256 or K % 64:
        raise ValueError("this isolated experiment uses full aligned tiles")
    bm, bn, bk, group_m = 128, 256, 64, 8
    a_desc = create_2d_tma_descriptor(A.data_ptr(), M, K, bm, bk, A.element_size())
    b_desc = create_2d_tma_descriptor(B.data_ptr(), N, K, bn, bk, B.element_size())
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)

    def launch():
        return bf16_gemm_tma[grid](
            a_desc, b_desc, C, M, N, K, bm, bn, bk, group_m,
            num_warps=8, num_stages=3,
        )

    return launch
