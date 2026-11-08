"""Wait-one Gluon GEMM with independently chosen tile order, N tile, persistence."""

from __future__ import annotations

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (
    fence_async_shared,
    mbarrier,
    tma,
    warpgroup_mma,
    warpgroup_mma_init,
    warpgroup_mma_wait,
)


CTA_M = 128
BLOCK_K = 64
WARPS = 8


@gluon.jit
def gemm(a_desc, b_desc, c_desc,
         M_TILES: gl.constexpr, N_TILES: gl.constexpr,
         WORKERS: gl.constexpr, ORDER: gl.constexpr):
    block_m: gl.constexpr = a_desc.block_shape[0]
    block_n: gl.constexpr = b_desc.block_shape[0]
    block_k: gl.constexpr = a_desc.block_shape[1]
    dtype: gl.constexpr = a_desc.dtype
    k_size = a_desc.shape[1]

    a_smem = gl.allocate_shared_memory(dtype, [3] + a_desc.block_shape, a_desc.layout)
    b_smem = gl.allocate_shared_memory(dtype, [3] + b_desc.block_shape, b_desc.layout)
    bars = mbarrier.allocate_mbarrier(batch=3)
    mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0], warps_per_cta=[8, 1], instr_shape=[16, block_n, 16],
        cga_layout=[[1, 0]],
    )
    steps = k_size // block_k
    for tile_id in range(gl.program_id(0), M_TILES * N_TILES, WORKERS):
        if ORDER == 0:
            pid_m = tile_id % M_TILES
            pid_n = tile_id // M_TILES
        else:
            pid_n = tile_id % N_TILES
            pid_m = tile_id // N_TILES
        off_m = pid_m * block_m
        off_n = pid_n * block_n

        initial_acc = gl.zeros((block_m, block_n), dtype=gl.float32, layout=mma_layout)
        acc = warpgroup_mma_init(initial_acc)
        for initial_slot in gl.static_range(3):
            mbarrier.init(bars.index(initial_slot), count=1)
        for pre in gl.static_range(3):
            a_tile = a_smem.index(pre)
            b_tile = b_smem.index(pre)
            bar = bars.index(pre)
            mbarrier.expect(bar, a_desc.nbytes_per_cta + b_desc.nbytes_per_cta)
            tma.async_load(a_desc, [off_m, pre * block_k], bar, a_tile)
            tma.async_load(b_desc, [off_n, pre * block_k], bar, b_tile,
                           multicast=False)
        for step in range(steps):
            slot = step % 3
            a_tile = a_smem.index(slot)
            b_tile = b_smem.index(slot)
            mbarrier.wait(bars.index(slot), phase=(step // 3) % 2,
                          deps=[a_tile, b_tile])
            acc = warpgroup_mma(a_tile, b_tile.permute((1, 0)), acc, is_async=True)
            warpgroup_mma_wait(num_outstanding=1, deps=(acc,))
            if step > 0:
                future = step + 2
                if future < steps:
                    next_slot = future % 3
                    next_a = a_smem.index(next_slot)
                    next_b = b_smem.index(next_slot)
                    next_bar = bars.index(next_slot)
                    mbarrier.expect(next_bar, a_desc.nbytes_per_cta + b_desc.nbytes_per_cta)
                    tma.async_load(a_desc, [off_m, future * block_k], next_bar, next_a)
                    tma.async_load(b_desc, [off_n, future * block_k], next_bar,
                                   next_b, multicast=False)

        acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))
        for final_slot in gl.static_range(3):
            mbarrier.invalidate(bars.index(final_slot))
        c_smem = gl.allocate_shared_memory(dtype, c_desc.block_shape, c_desc.layout)
        c_smem.store(acc.to(dtype))
        fence_async_shared()
        tma.async_store(c_desc, [off_m, off_n], c_smem)
        tma.store_wait(pendings=0)
