"""Four-warp one-CTA variant of the validated Gluon wait-one kernel."""

from __future__ import annotations

import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import (
    fence_async_shared,
    mbarrier,
    tma,
    warpgroup_mma,
    warpgroup_mma_init,
    warpgroup_mma_wait,
)


CTA_M = 128
CTA_N = 256
BLOCK_K = 64
WARPS = 4


@gluon.jit(do_not_specialize=["B_MODE"])
def gemm(a_desc, b_desc, c_desc, CLUSTERED: gl.constexpr,
         PIPELINED: gl.constexpr, B_MODE):
    block_m: gl.constexpr = a_desc.block_shape[0]
    block_n: gl.constexpr = b_desc.block_shape[0]
    block_k: gl.constexpr = a_desc.block_shape[1]
    dtype: gl.constexpr = a_desc.dtype
    k_size = a_desc.shape[1]

    a_smem = gl.allocate_shared_memory(dtype, [3] + a_desc.block_shape, a_desc.layout)
    b_smem = gl.allocate_shared_memory(dtype, [3] + b_desc.block_shape, b_desc.layout)
    pid_m = gl.program_id(0)
    pid_n = gl.program_id(1)
    off_m = pid_m * block_m
    off_n = pid_n * block_n

    if CLUSTERED:
        mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
            version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, 256, 16],
            cga_layout=[[1, 0]],
        )
    else:
        mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
            version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, 256, 16],
        )
    initial_acc = gl.zeros((block_m, block_n), dtype=gl.float32, layout=mma_layout)
    acc = warpgroup_mma_init(initial_acc)
    bars = mbarrier.allocate_mbarrier(batch=3)
    for initial_slot in gl.static_range(3):
        mbarrier.init(bars.index(initial_slot), count=1)
    steps = k_size // block_k
    for pre in gl.static_range(3):
        a_tile = a_smem.index(pre)
        b_tile = b_smem.index(pre)
        bar = bars.index(pre)
        mbarrier.expect(bar, a_desc.nbytes_per_cta + b_desc.nbytes_per_cta)
        tma.async_load(a_desc, [off_m, pre * block_k], bar, a_tile)
        if B_MODE == 1:
            tma.async_load(b_desc, [off_n, pre * block_k], bar, b_tile,
                           multicast=False)
        else:
            tma.async_load(b_desc, [off_n, pre * block_k], bar, b_tile,
                           multicast=True)
    for step in range(steps):
        slot = step % 3
        a_tile = a_smem.index(slot)
        b_tile = b_smem.index(slot)
        mbarrier.wait(bars.index(slot), phase=(step // 3) % 2,
                      deps=[a_tile, b_tile])
        acc = warpgroup_mma(a_tile, b_tile.permute((1, 0)), acc, is_async=True)
        # Keep the current group in flight while releasing the previous slot.
        warpgroup_mma_wait(num_outstanding=1, deps=(acc,))
        if step > 0:
            # The prior group has released its slot before we refill it.
            future = step + 2
            if future < steps:
                next_slot = future % 3
                next_a = a_smem.index(next_slot)
                next_b = b_smem.index(next_slot)
                next_bar = bars.index(next_slot)
                mbarrier.expect(next_bar, a_desc.nbytes_per_cta + b_desc.nbytes_per_cta)
                tma.async_load(a_desc, [off_m, future * block_k], next_bar, next_a)
                if B_MODE == 1:
                    tma.async_load(b_desc, [off_n, future * block_k], next_bar,
                                   next_b, multicast=False)
                else:
                    tma.async_load(b_desc, [off_n, future * block_k], next_bar,
                                   next_b, multicast=True)

    acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))
    for final_slot in gl.static_range(3):
        mbarrier.invalidate(bars.index(final_slot))
    c_smem = gl.allocate_shared_memory(dtype, c_desc.block_shape, c_desc.layout)
    c_smem.store(acc.to(dtype))
    fence_async_shared()
    tma.async_store(c_desc, [off_m, off_n], c_smem)
    tma.store_wait(pendings=0)


def launch(A, B, C, M, N, K):
    """Compute C=A@B for A[M,K], column-major B[K,N], row-major C[M,N]."""
    import torch

    if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16 or C.dtype != torch.bfloat16:
        raise ValueError("expected BF16 A, B and C")
    if A.shape != (M, K) or B.shape != (K, N) or C.shape != (M, N):
        raise ValueError("GEMM dimensions disagree")
    if not A.is_contiguous() or not C.is_contiguous() or B.stride() != (1, K):
        raise ValueError("expected row-major A/C and column-major B")
    # The search ABI views B as [K,N] with strides (1,K). Its transpose is a
    # zero-copy, contiguous [N,K] view for the validated descriptor layout.
    b_physical = B.T
    if not b_physical.is_contiguous():
        raise ValueError("B transpose must be physically contiguous")
    cluster_size = 1
    block_m = CTA_M * cluster_size
    if M % block_m or N % CTA_N or K % BLOCK_K or K < 3 * BLOCK_K:
        raise ValueError("wait-one Gluon reference requires full tiles and three K stages")
    a_layout = gl.NVMMASharedLayout.get_default_for(
        [block_m, BLOCK_K], gl.bfloat16)
    b_layout = gl.NVMMASharedLayout.get_default_for(
        [CTA_N, BLOCK_K], gl.bfloat16)
    c_layout = gl.NVMMASharedLayout.get_default_for(
        [block_m, CTA_N], gl.bfloat16)
    a_desc = TensorDescriptor.from_tensor(A, [block_m, BLOCK_K], a_layout)
    b_desc = TensorDescriptor.from_tensor(b_physical, [CTA_N, BLOCK_K], b_layout)
    c_desc = TensorDescriptor.from_tensor(C, [block_m, CTA_N], c_layout)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, CTA_N))
    return gemm[grid](a_desc, b_desc, c_desc, False, True, 1,
                      num_warps=WARPS, num_ctas=cluster_size)
