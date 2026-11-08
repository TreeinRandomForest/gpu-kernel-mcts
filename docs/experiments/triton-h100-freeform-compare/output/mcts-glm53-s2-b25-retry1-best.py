import triton
import triton.language as tl


@triton.jit
def bf16_gemm_triton(A, B, C, M, N, K,
                      BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                      GROUP_M: tl.constexpr,
                      EVEN_K: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)

    # A row-major [M,K]: stride (K, 1)
    a_ptrs = A + rm[:, None] * K + rk[None, :]
    # B column-major [K,N]: K stride 1, N stride K
    b_ptrs = B + rk[:, None] + rn[None, :] * K

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    if EVEN_K:
        a_ptrs = tl.multiple_of(a_ptrs, (16, 16))
        b_ptrs = tl.multiple_of(b_ptrs, (16, 16))
        for k in range(0, tl.cdiv(K, BK)):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            acc = tl.dot(a, b, acc)
            a_ptrs += BK
            b_ptrs += BK
    else:
        for k in range(0, tl.cdiv(K, BK)):
            a = tl.load(a_ptrs, mask=(rm[:, None] < M) & (rk[None, :] + k * BK < K), other=0.0)
            b = tl.load(b_ptrs, mask=(rk[:, None] + k * BK < K) & (rn[None, :] < N), other=0.0)
            acc = tl.dot(a, b, acc)
            a_ptrs += BK
            b_ptrs += BK

    c = acc.to(tl.bfloat16)
    c_ptrs = C + rm[:, None] * N + rn[None, :]
    tl.store(c_ptrs, c, mask=(rm[:, None] < M) & (rn[None, :] < N))


def launch(A, B, C, M, N, K):
    BM, BN, BK, GROUP_M = 128, 256, 64, 8
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    return bf16_gemm_triton[grid](
        A, B, C, M, N, K,
        BM, BN, BK, GROUP_M,
        K % BK == 0,
        num_warps=8,
        num_stages=3,
    )