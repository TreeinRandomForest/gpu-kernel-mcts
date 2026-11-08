"""Two adjacent 128-wide outputs per persistent cluster, with optional A reuse."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (
    fence_async_shared, mbarrier, tma, warpgroup_mma, warpgroup_mma_init,
    warpgroup_mma_wait,
)


BLOCK_M = 256
BLOCK_N = 128
BLOCK_K = 64
WARPS = 8
WORKERS = 66


@gluon.jit
def gemm(a_desc, b_desc, c_desc, M_TILES: gl.constexpr, N_PAIRS: gl.constexpr,
         WORKERS: gl.constexpr, REUSE_A: gl.constexpr):
    block_m: gl.constexpr = a_desc.block_shape[0]
    block_n: gl.constexpr = b_desc.block_shape[0]
    block_k: gl.constexpr = a_desc.block_shape[1]
    dtype: gl.constexpr = a_desc.dtype
    steps = a_desc.shape[1] // block_k

    a0_smem = gl.allocate_shared_memory(dtype, [3] + a_desc.block_shape, a_desc.layout)
    if not REUSE_A:
        a1_smem = gl.allocate_shared_memory(dtype, [3] + a_desc.block_shape, a_desc.layout)
    b0_smem = gl.allocate_shared_memory(dtype, [3] + b_desc.block_shape, b_desc.layout)
    b1_smem = gl.allocate_shared_memory(dtype, [3] + b_desc.block_shape, b_desc.layout)
    c_smem = gl.allocate_shared_memory(dtype, c_desc.block_shape, c_desc.layout)
    bars = mbarrier.allocate_mbarrier(batch=3)
    mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0], warps_per_cta=[8, 1], instr_shape=[16, block_n, 16],
        cga_layout=[[1, 0]],
    )

    for tile_id in range(gl.program_id(0), M_TILES * N_PAIRS, WORKERS):
        pid_m = tile_id % M_TILES
        pid_n_pair = tile_id // M_TILES
        off_m = pid_m * block_m
        off_n0 = pid_n_pair * (2 * block_n)
        off_n1 = off_n0 + block_n
        initial = gl.zeros((block_m, block_n), dtype=gl.float32, layout=mma_layout)
        acc0 = warpgroup_mma_init(initial)
        acc1 = warpgroup_mma_init(initial)

        for initial_slot in gl.static_range(3):
            mbarrier.init(bars.index(initial_slot), count=1)
        for pre in gl.static_range(3):
            a0 = a0_smem.index(pre)
            b0 = b0_smem.index(pre)
            b1 = b1_smem.index(pre)
            bar = bars.index(pre)
            if REUSE_A:
                mbarrier.expect(bar, a_desc.nbytes_per_cta + 2 * b_desc.nbytes_per_cta)
            else:
                a1 = a1_smem.index(pre)
                mbarrier.expect(bar, 2 * a_desc.nbytes_per_cta + 2 * b_desc.nbytes_per_cta)
            tma.async_load(a_desc, [off_m, pre * block_k], bar, a0)
            if not REUSE_A:
                tma.async_load(a_desc, [off_m, pre * block_k], bar, a1)
            tma.async_load(b_desc, [off_n0, pre * block_k], bar, b0,
                           multicast=False)
            tma.async_load(b_desc, [off_n1, pre * block_k], bar, b1,
                           multicast=False)

        for step in range(steps):
            slot = step % 3
            a0 = a0_smem.index(slot)
            b0 = b0_smem.index(slot)
            b1 = b1_smem.index(slot)
            bar = bars.index(slot)
            if REUSE_A:
                a1 = a0
                mbarrier.wait(bar, phase=(step // 3) % 2, deps=[a0, b0, b1])
            else:
                a1 = a1_smem.index(slot)
                mbarrier.wait(bar, phase=(step // 3) % 2, deps=[a0, a1, b0, b1])
            acc0 = warpgroup_mma(a0, b0.permute((1, 0)), acc0, is_async=True)
            acc1 = warpgroup_mma(a1, b1.permute((1, 0)), acc1, is_async=True)
            warpgroup_mma_wait(num_outstanding=0, deps=(acc0, acc1))
            if step > 0:
                future = step + 2
                if future < steps:
                    next_slot = future % 3
                    next_a0 = a0_smem.index(next_slot)
                    next_b0 = b0_smem.index(next_slot)
                    next_b1 = b1_smem.index(next_slot)
                    next_bar = bars.index(next_slot)
                    if REUSE_A:
                        mbarrier.expect(next_bar, a_desc.nbytes_per_cta + 2 * b_desc.nbytes_per_cta)
                    else:
                        next_a1 = a1_smem.index(next_slot)
                        mbarrier.expect(next_bar, 2 * a_desc.nbytes_per_cta + 2 * b_desc.nbytes_per_cta)
                    tma.async_load(a_desc, [off_m, future * block_k], next_bar, next_a0)
                    if not REUSE_A:
                        tma.async_load(a_desc, [off_m, future * block_k], next_bar,
                                       next_a1)
                    tma.async_load(b_desc, [off_n0, future * block_k], next_bar,
                                   next_b0, multicast=False)
                    tma.async_load(b_desc, [off_n1, future * block_k], next_bar,
                                   next_b1, multicast=False)

        acc0, acc1 = warpgroup_mma_wait(num_outstanding=0, deps=(acc0, acc1))
        for final_slot in gl.static_range(3):
            mbarrier.invalidate(bars.index(final_slot))
        c_smem.store(acc0.to(dtype))
        fence_async_shared()
        tma.async_store(c_desc, [off_m, off_n0], c_smem)
        tma.store_wait(pendings=0)
        c_smem.store(acc1.to(dtype))
        fence_async_shared()
        tma.async_store(c_desc, [off_m, off_n1], c_smem)
        tma.store_wait(pendings=0)
