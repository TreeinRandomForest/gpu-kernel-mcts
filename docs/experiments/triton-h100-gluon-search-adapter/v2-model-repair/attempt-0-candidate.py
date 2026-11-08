import torch
import triton
import triton.language as tl


def _alloc(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


_alloc_initialized = False


@triton.jit
def bf16_gemm_tma(A, B, C, M, N, K,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                  GROUP_M: tl.constexpr):
    # A is row-major [M,K] -> descriptor with strides [K, 1].
    a_desc = tl.make_tensor_descriptor(A, [M, K], [K, 1], [BM, BK])
    # B is column-major [K,N]; viewed physically as B.T [N,K] row-major
    # (element (n,k) sits at B + n*K + k), giving strides [K, 1].
    b_desc = tl.make_tensor_descriptor(B, [N, K], [K, 1], [BN, BK])

    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    off_m = pid_m * BM
    off_n = pid_n * BN

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for kk in range(0, tl.cdiv(K, BK)):
        a = a_desc.load([off_m, kk * BK])
        b = b_desc.load([off_n, kk * BK])
        acc = tl.dot(a, tl.trans(b), acc)

    c = acc.to(tl.bfloat16)
    rows = off_m + tl.arange(0, BM)
    cols = off_n + tl.arange(0, BN)
    tl.store(C + rows[:, None] * N + cols[None, :], c,
             mask=(rows[:, None] < M) & (cols[None, :] < N))


def launch(A, B, C, M, N, K):
    global _alloc_initialized
    if not _alloc_initialized:
        triton.set_allocator(_alloc)
        _alloc_initialized = True

    BM, BN, BK, GROUP_M = 128, 256, 64, 8
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    return bf16_gemm_tma[grid](
        A, B, C, M, N, K,
        BM, BN, BK, GROUP_M,
        num_warps=8,
        num_stages=3,
    )