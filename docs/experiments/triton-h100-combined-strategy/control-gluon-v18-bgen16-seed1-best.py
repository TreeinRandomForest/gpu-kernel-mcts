import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import (
    mbarrier, tma, warpgroup_mma, warpgroup_mma_init, warpgroup_mma_wait,
    fence_async_shared,
)


@gluon.jit
def bf16_gemm_tma_wgmma(A, B, C, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
                         BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr,
                         a_desc, b_desc, c_desc):
    pid_m = gl.program_id(0)
    pid_n = gl.program_id(1)

    a_smem = gl.allocate_shared_memory(a_desc.dtype, [3] + a_desc.block_shape, a_desc.layout)
    b_smem = gl.allocate_shared_memory(b_desc.dtype, [3] + b_desc.block_shape, b_desc.layout)
    c_smem = gl.allocate_shared_memory(c_desc.dtype, c_desc.block_shape, c_desc.layout)

    bars = mbarrier.allocate_mbarrier(batch=3)
    for initial_slot in gl.static_range(3):
        mbarrier.init(bars.index(initial_slot), count=1)

    mma_layout: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0],
                                                         warps_per_cta=[8, 1],
                                                         instr_shape=[16, 256, 16])

    acc = warpgroup_mma_init(gl.zeros((BM, BN), gl.float32, layout=mma_layout))

    off_m = pid_m * BM
    off_n = pid_n * BN

    steps = K // BK

    for pre in gl.static_range(3):
        a_tile = a_smem.index(pre)
        b_tile = b_smem.index(pre)
        bar = bars.index(pre)
        mbarrier.expect(bar, a_desc.nbytes_per_cta + b_desc.nbytes_per_cta)
        tma.async_load(a_desc, [off_m, pre * BK], bar, a_tile)
        tma.async_load(b_desc, [off_n, pre * BK], bar, b_tile)

    for step in range(steps):
        slot = step % 3
        a_tile = a_smem.index(slot)
        b_tile = b_smem.index(slot)
        mbarrier.wait(bars.index(slot), phase=(step // 3) % 2, deps=[a_tile, b_tile])
        acc = warpgroup_mma(a_tile, b_tile.permute((1, 0)), acc, is_async=True)
        warpgroup_mma_wait(num_outstanding=1, deps=(acc,))

        if step > 0:
            future = step + 2
            if future < steps:
                refill_slot = future % 3
                a_next = a_smem.index(refill_slot)
                b_next = b_smem.index(refill_slot)
                bar_next = bars.index(refill_slot)
                mbarrier.expect(bar_next, a_desc.nbytes_per_cta + b_desc.nbytes_per_cta)
                tma.async_load(a_desc, [off_m, future * BK], bar_next, a_next)
                tma.async_load(b_desc, [off_n, future * BK], bar_next, b_next)

    acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))
    for done_slot in gl.static_range(3):
        mbarrier.invalidate(bars.index(done_slot))

    c_smem.store(acc.to(gl.bfloat16))
    fence_async_shared()
    tma.async_store(c_desc, [off_m, off_n], c_smem)
    tma.store_wait(pendings=0)


def launch(A, B, C, M, N, K):
    BM, BN, BK = 128, 256, 64

    assert A.dtype == torch.bfloat16
    assert C.dtype == torch.bfloat16
    assert A.data_ptr() % 16 == 0
    assert C.data_ptr() % 16 == 0
    assert A.stride(1) == 1 and A.stride(0) == K
    assert C.stride(1) == 1 and C.stride(0) == N
    assert B.stride(0) == 1 and B.stride(1) == K

    a_layout = gl.NVMMASharedLayout.get_default_for([BM, BK], gl.bfloat16)
    b_layout = gl.NVMMASharedLayout.get_default_for([BN, BK], gl.bfloat16)
    c_layout = gl.NVMMASharedLayout.get_default_for([BM, BN], gl.bfloat16)

    a_desc = TensorDescriptor.from_tensor(A, [BM, BK], a_layout)
    b_desc = TensorDescriptor.from_tensor(B.T, [BN, BK], b_layout)
    c_desc = TensorDescriptor.from_tensor(C, [BM, BN], c_layout)

    return bf16_gemm_tma_wgmma[(triton.cdiv(M, BM), triton.cdiv(N, BN))](A, B, C, M, N, K, BM, BN, BK, a_desc, b_desc, c_desc, num_warps=8)