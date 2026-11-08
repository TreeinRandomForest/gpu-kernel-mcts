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

    row = pid_m * BM + tl.arange(0, BM)
    col = pid_n * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + kk
        a = tl.load(A + row[:, None] * K + k[None, :],
                    (row[:, None] < M) & (k[None, :] < K), other=0)
        # B is [K,N] column-major: K stride is 1, N stride is K.
        # Load a [BN, BK] tile so memory access is contiguous along K,
        # then transpose for the dot product.
        b = tl.load(B + col[:, None] * K + k[None, :],
                    (col[:, None] < N) & (k[None, :] < K), other=0)
        acc = tl.dot(a, tl.trans(b), acc)

    tl.store(C + row[:, None] * N + col[None, :], acc.to(tl.bfloat16),
             (row[:, None] < M) & (col[None, :] < N))


def launch(A, B, C, M, N, K):
    BM, BN, BK, GROUP_M = 128, 256, 64, 8
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    return bf16_gemm_triton[grid](
        A, B, C, M, N, K, BM, BN, BK, GROUP_M,
        num_warps=8, num_stages=4
    )