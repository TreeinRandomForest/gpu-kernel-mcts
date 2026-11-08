import triton
import triton.language as tl


@triton.jit
def bf16_gemm_triton(A, B, C, M, N: tl.constexpr, K: tl.constexpr,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                     GROUP_M: tl.constexpr, EVEN_K: tl.constexpr):
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
    # B is [K,N] column-major: K stride 1, N stride K.
    b_ptrs = B + kk[:, None] + col[None, :] * K

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    if EVEN_K:
        for k in range(0, tl.cdiv(K, BK)):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            acc = tl.dot(a, b, acc)
            a_ptrs += BK
            b_ptrs += BK
    else:
        for k in range(0, tl.cdiv(K, BK)):
            a = tl.load(a_ptrs, mask=(row[:, None] < M) & (kk[None, :] + k * BK < K), other=0.0)
            b = tl.load(b_ptrs, mask=(kk[:, None] + k * BK < K) & (col[None, :] < N), other=0.0)
            acc = tl.dot(a, b, acc)
            a_ptrs += BK
            b_ptrs += BK

    c = acc.to(tl.bfloat16)
    c_ptrs = C + row[:, None] * N + col[None, :]
    tl.store(c_ptrs, c, mask=(row[:, None] < M) & (col[None, :] < N))


def launch(A, B, C, M, N, K):
    bm, bn, bk = 128, 256, 64
    group_m = 8
    even_k = (K % bk == 0) and (M % bm == 0) and (N % bn == 0)
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    return bf16_gemm_triton[grid](
        A, B, C, M, N, K, bm, bn, bk, group_m, even_k,
        num_warps=8, num_stages=4
    )