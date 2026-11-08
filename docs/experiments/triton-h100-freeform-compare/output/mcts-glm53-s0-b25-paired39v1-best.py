import triton
import triton.language as tl


@triton.jit
def bf16_gemm_triton(A, B, C, M, N, K,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                     GROUP_M: tl.constexpr, EVEN_MN: tl.constexpr,
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

    row = pid_m * BM + tl.arange(0, BM)
    col = pid_n * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)

    offs_a = row[:, None] * K + kk[None, :]
    offs_b = kk[:, None] + col[None, :] * K

    a_ptrs = A + offs_a
    b_ptrs = B + offs_b

    acc = tl.zeros((BM, BN), dtype=tl.float32)

    if EVEN_MN:
        if EVEN_K:
            for _ in range(0, tl.cdiv(K, BK)):
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
                acc = tl.dot(a, b, acc)
                a_ptrs += BK
                b_ptrs += BK
        else:
            for k in range(0, tl.cdiv(K, BK)):
                k_rem = K - k * BK
                a = tl.load(a_ptrs, mask=(kk[None, :] < k_rem), other=0.0)
                b = tl.load(b_ptrs, mask=(kk[:, None] < k_rem), other=0.0)
                acc = tl.dot(a, b, acc)
                a_ptrs += BK
                b_ptrs += BK
    else:
        a_mask = row[:, None] < M
        b_mask = col[None, :] < N
        if EVEN_K:
            for _ in range(0, tl.cdiv(K, BK)):
                a = tl.load(a_ptrs, mask=a_mask, other=0.0)
                b = tl.load(b_ptrs, mask=b_mask, other=0.0)
                acc = tl.dot(a, b, acc)
                a_ptrs += BK
                b_ptrs += BK
        else:
            for k in range(0, tl.cdiv(K, BK)):
                k_rem = K - k * BK
                a = tl.load(a_ptrs, mask=a_mask & (kk[None, :] < k_rem), other=0.0)
                b = tl.load(b_ptrs, mask=(kk[:, None] < k_rem) & b_mask, other=0.0)
                acc = tl.dot(a, b, acc)
                a_ptrs += BK
                b_ptrs += BK

    c = acc.to(tl.bfloat16)
    c_ptrs = C + row[:, None] * N + col[None, :]
    if EVEN_MN:
        tl.store(c_ptrs, c)
    else:
        tl.store(c_ptrs, c, mask=(row[:, None] < M) & (col[None, :] < N))


def launch(A, B, C, M, N, K):
    BM, BN, BK, GROUP_M = 128, 256, 64, 8
    EVEN_MN = (M % BM == 0) and (N % BN == 0)
    EVEN_K = (K % BK) == 0
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    return bf16_gemm_triton[grid](
        A, B, C, M, N, K,
        BM, BN, BK, GROUP_M, EVEN_MN, EVEN_K,
        num_warps=8, num_stages=4
    )