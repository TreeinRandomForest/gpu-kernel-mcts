import triton
import triton.language as tl


@triton.jit
def bf16_gemm_kernel(A, B, C, M, N, K,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                     GROUP_M: tl.constexpr,
                     EVEN_K: tl.constexpr, FULL_MN: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)

    # L2-friendly grouped ordering of tiles.
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    off_m = pid_m * BM
    off_n = pid_n * BN
    rm = off_m + tl.arange(0, BM)
    rn = off_n + tl.arange(0, BN)
    rk = tl.arange(0, BK)

    rm = tl.max_contiguous(tl.multiple_of(rm, BM), BM)
    rn = tl.max_contiguous(tl.multiple_of(rn, BN), BN)

    # A is row-major [M,K]: addr(m,k) = m*K + k
    a_ptrs = A + rm[:, None] * K + rk[None, :]
    # B is column-major [K,N]: addr(k,n) = n*K + k
    # (equivalently B.T viewed as row-major [N,K], zero-copy)
    b_ptrs = B + rn[None, :] * K + rk[:, None]

    acc = tl.zeros((BM, BN), dtype=tl.float32)

    if EVEN_K:
        for _ in range(0, tl.cdiv(K, BK)):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            acc = tl.dot(a, b, acc)
            a_ptrs += BK
            b_ptrs += BK
    else:
        for k in range(0, tl.cdiv(K, BK) * BK, BK):
            a = tl.load(a_ptrs, mask=rm[:, None] < M, other=0.0)
            b = tl.load(b_ptrs, mask=rn[None, :] < N, other=0.0)
            acc = tl.dot(a, b, acc)
            a_ptrs += BK
            b_ptrs += BK

    c = acc.to(tl.bfloat16)
    c_ptrs = C + rm[:, None] * N + rn[None, :]
    if FULL_MN:
        tl.store(c_ptrs, c)
    else:
        tl.store(c_ptrs, c, mask=(rm[:, None] < M) & (rn[None, :] < N))


def launch(A, B, C, M, N, K):
    BM, BN, BK = 128, 256, 64
    GROUP_M = 8
    EVEN_K = (K % BK) == 0
    FULL_MN = (M % BM) == 0 and (N % BN) == 0
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    return bf16_gemm_kernel[grid](
        A, B, C, M, N, K,
        BM, BN, BK, GROUP_M,
        EVEN_K, FULL_MN,
        num_warps=8, num_stages=3,
    )