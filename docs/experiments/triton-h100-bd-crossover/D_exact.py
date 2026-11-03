import triton
import triton.language as tl


@triton.jit
def bf16_gemm_triton(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                     GROUP_M: tl.constexpr):
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
    kk = tl.arange(0, BK)

    a_ptrs = A + row[:, None] * K + kk[None, :]
    b_ptrs = B + kk[:, None] + col[None, :] * K

    acc = tl.full((BM, BN), 0.0, tl.float32)

    a_mask_row = row[:, None] < M
    b_mask_col = col[None, :] < N

    for k0 in range(0, K, BK):
        k = k0 + kk
        kmask = k < K
        a = tl.load(a_ptrs, a_mask_row & kmask[None, :], other=0.0)
        b = tl.load(b_ptrs, kmask[:, None] & b_mask_col, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BK
        b_ptrs += BK

    c_ptrs = C + row[:, None] * N + col[None, :]
    tl.store(c_ptrs, acc.to(tl.bfloat16), a_mask_row & b_mask_col)


def launch(A, B, C, M, N, K):
    BM, BN, BK = 128, 256, 64
    GROUP_M = 8
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    return bf16_gemm_triton[grid](
        A, B, C, M, N, K, BM, BN, BK, GROUP_M,
        num_warps=8, num_stages=3
    )