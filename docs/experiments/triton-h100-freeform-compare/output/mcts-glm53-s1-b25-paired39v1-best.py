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
    rk = tl.arange(0, BK)

    row = tl.max_contiguous(tl.multiple_of(row, BM), BM)
    col = tl.max_contiguous(tl.multiple_of(col, BN), BN)

    A_ptrs = A + row[:, None].to(tl.int64) * K + rk[None, :]
    # B is [K,N] column-major: K stride 1, N stride K
    B_ptrs = B + rk[:, None] + col[None, :].to(tl.int64) * K

    acc = tl.zeros((BM, BN), dtype=tl.float32)

    EVEN_K: tl.constexpr = (K % BK) == 0
    if EVEN_K:
        for _ in range(0, K, BK):
            a = tl.load(A_ptrs)
            b = tl.load(B_ptrs)
            acc = tl.dot(a, b, acc)
            A_ptrs += BK
            B_ptrs += BK
    else:
        for k in range(0, K, BK):
            mask_k = (k + rk) < K
            a = tl.load(A_ptrs, mask=mask_k[None, :], other=0.0)
            b = tl.load(B_ptrs, mask=mask_k[:, None], other=0.0)
            acc = tl.dot(a, b, acc)
            A_ptrs += BK
            B_ptrs += BK

    c_ptrs = C + row[:, None].to(tl.int64) * N + col[None, :]
    mask = (row[:, None] < M) & (col[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=mask)


def launch(A, B, C, M, N, K):
    BM, BN, BK = 128, 256, 64
    GROUP_M = 8
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    return bf16_gemm_triton[grid](
        A, B, C, M, N, K, BM, BN, BK, GROUP_M,
        num_warps=8, num_stages=4,
    )