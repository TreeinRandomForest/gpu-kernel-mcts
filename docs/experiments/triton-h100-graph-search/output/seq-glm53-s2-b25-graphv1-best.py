import torch
import triton
import triton.language as tl


@triton.jit
def bf16_gemm_triton(A, B, C, M, N, K, BM: tl.constexpr, BN: tl.constexpr,
                     BK: tl.constexpr, GROUP_M: tl.constexpr,
                     EVEN_K: tl.constexpr, NUM_SMS: tl.constexpr):
    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_tiles = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        row = pid_m * BM + tl.arange(0, BM)
        col = pid_n * BN + tl.arange(0, BN)
        kk = tl.arange(0, BK)

        a_ptrs = A + row[:, None] * K + kk[None, :]
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
                a = tl.load(a_ptrs, (row[:, None] < M) & (kk[None, :] < K), other=0.0)
                b = tl.load(b_ptrs, (kk[:, None] < K) & (col[None, :] < N), other=0.0)
                acc = tl.dot(a, b, acc)
                a_ptrs += BK
                b_ptrs += BK

        c_ptrs = C + row[:, None] * N + col[None, :]
        tl.store(c_ptrs, acc.to(tl.bfloat16),
                 (row[:, None] < M) & (col[None, :] < N))


def launch(A, B, C, M, N, K):
    BM, BN, BK, GROUP_M = 128, 256, 64, 8
    even_k = (K % BK) == 0 and (M % BM) == 0 and (N % BN) == 0
    num_sms = torch.cuda.get_device_properties(A.device).multi_processor_count
    grid = (min(num_sms, triton.cdiv(M, BM) * triton.cdiv(N, BN)),)
    return bf16_gemm_triton[grid](
        A, B, C, M, N, K, BM, BN, BK, GROUP_M, even_k, num_sms,
        num_warps=8, num_stages=4
    )