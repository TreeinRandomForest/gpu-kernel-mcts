import triton
import triton.language as tl


@triton.jit
def bf16_gemm_triton(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                     GROUP_M: tl.constexpr, EVEN_K: tl.constexpr,
                     NUM_SMS: tl.constexpr):
    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    k_tiles = tl.cdiv(K, BK)
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

        row = tl.max_contiguous(tl.multiple_of(row, BM), BM)
        col = tl.max_contiguous(tl.multiple_of(col, BN), BN)

        acc = tl.zeros((BM, BN), dtype=tl.float32)

        a_ptrs = A + row[:, None] * K + kk[None, :]
        b_ptrs = B + kk[:, None] + col[None, :] * K

        if EVEN_K:
            for ki in tl.range(0, k_tiles, num_stages=4):
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
                acc = tl.dot(a, b, acc)
                a_ptrs += BK
                b_ptrs += BK
        else:
            for ki in range(0, k_tiles):
                k_rem = K - ki * BK
                a = tl.load(a_ptrs, mask=kk[None, :] < k_rem, other=0.0)
                b = tl.load(b_ptrs, mask=kk[:, None] < k_rem, other=0.0)
                acc = tl.dot(a, b, acc)
                a_ptrs += BK
                b_ptrs += BK

        c_ptrs = C + row[:, None] * N + col[None, :]
        tl.store(c_ptrs, acc.to(tl.bfloat16))


def launch(A, B, C, M, N, K):
    bm, bn, bk = 128, 256, 64
    NUM_SMS = 132
    grid = (min(NUM_SMS, triton.cdiv(M, bm) * triton.cdiv(N, bn)),)
    return bf16_gemm_triton[grid](
        A, B, C, M, N, K, bm, bn, bk, 8,
        K % bk == 0,
        NUM_SMS,
        num_warps=8,
        num_stages=4,
    )