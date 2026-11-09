import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def bf16_gemm_tma_persistent(a_desc, b_desc, c_desc, M, N, K,
                              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                              GROUP_M: tl.constexpr, NUM_SMS: tl.constexpr):
    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    k_tiles = tl.cdiv(K, BK)
    num_tiles = num_pid_m * num_pid_n

    tiles_per_sm = tl.cdiv(num_tiles, NUM_SMS)

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS,
                            flatten=True, warp_specialize=True):
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + (tile_id % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        off_m = pid_m * BM
        off_n = pid_n * BN

        acc = tl.zeros((BM, BN), dtype=tl.float32)

        for ki in tl.range(k_tiles):
            off_k = ki * BK
            a = a_desc.load([off_m, off_k])
            b = b_desc.load([off_n, off_k])
            acc = tl.dot(a, tl.trans(b), acc)

        c_desc.store([off_m, off_n], acc.to(tl.bfloat16))


def launch(A, B, C, M, N, K):
    BM, BN, BK = 128, 256, 64
    GROUP_M = 8
    NUM_SMS = 132

    a_desc = TensorDescriptor.from_tensor(A, [BM, BK])
    # B is column-major [K, N]; B.t() is a row-major [N, K] view.
    b_desc = TensorDescriptor.from_tensor(B.t(), [BN, BK])
    c_desc = TensorDescriptor.from_tensor(C, [BM, BN])

    grid = (NUM_SMS,)
    return bf16_gemm_tma_persistent[grid](
        a_desc, b_desc, c_desc, M, N, K,
        BM, BN, BK, GROUP_M, NUM_SMS,
        num_warps=8,
        num_stages=3,
    )