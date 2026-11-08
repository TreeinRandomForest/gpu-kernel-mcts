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
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)

    # A row-major [M,K]: k contiguous
    a_ptrs = A + rm[:, None] * K + rk[None, :]
    # B column-major [K,N]: for fixed column n, k contiguous.
    # Load as (BN, BK) with k fast, transpose for dot.
    b_ptrs = B + rn[:, None] * K + rk[None, :]

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        a = tl.load(a_ptrs, mask=(rm[:, None] < M) & ((rk[None, :] + k * BK) < K), other=0.0)
        b = tl.load(b_ptrs, mask=(rn[:, None] < N) & ((rk[None, :] + k * BK) < K), other=0.0)
        acc = tl.dot(a, tl.trans(b), acc)
        a_ptrs += BK
        b_ptrs += BK

    c_ptrs = C + rm[:, None] * N + rn[None, :]
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=(rm[:, None] < M) & (rn[None, :] < N))


def launch(A, B, C, M, N, K):
    bm, bn, bk = 128, 256, 64
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    return bf16_gemm_triton[grid](
        A, B, C, M, N, K, bm, bn, bk, 8,
        num_warps=8, num_stages=4
    )