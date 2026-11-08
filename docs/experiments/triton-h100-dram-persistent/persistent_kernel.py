"""Fast B TMA kernel with persistent CTA assignment and unchanged tile math."""

import triton
import triton.language as tl


@triton.jit
def bf16_gemm_tma_persistent(a_desc, b_desc, C,
                             M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                             GROUP_M: tl.constexpr, NUM_SMS: tl.constexpr):
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_tiles = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_M * num_pid_n
    for tile_id in tl.range(tl.program_id(0), num_tiles, NUM_SMS, flatten=True):
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        row = pid_m * BM + tl.arange(0, BM)
        col = pid_n * BN + tl.arange(0, BN)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BK)):
            a = tl._experimental_descriptor_load(
                a_desc, [pid_m * BM, k * BK], [BM, BK], tl.bfloat16)
            b_physical = tl._experimental_descriptor_load(
                b_desc, [pid_n * BN, k * BK], [BN, BK], tl.bfloat16)
            acc = tl.dot(a, b_physical.T, acc)

        c = acc.to(tl.bfloat16)
        c_ptrs = C + row[:, None] * N + col[None, :]
        tl.store(c_ptrs, c, mask=(row[:, None] < M) & (col[None, :] < N))
