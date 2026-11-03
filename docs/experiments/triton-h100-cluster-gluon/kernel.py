"""Hopper BF16 GEMM with explicit two-CTA B multicast in Triton Gluon.

The single-CTA and clustered arms own the same 128x256 output tile per CTA.
The clustered arm shards A/C along M and broadcasts B with TMA multicast.
This experiment requires Triton 3.9.0; it is not an MCTS candidate.
"""

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
    warpgroup_mma_wait,
)


CTA_M = 128
CTA_N = 256
BLOCK_K = 64
WARPS = 8


@gluon.jit
def gemm(a_desc, b_desc, c_desc, CLUSTERED: gl.constexpr):
    block_m: gl.constexpr = a_desc.block_shape[0]
    block_n: gl.constexpr = b_desc.block_shape[0]
    block_k: gl.constexpr = a_desc.block_shape[1]
    dtype: gl.constexpr = a_desc.dtype
    k_size = a_desc.shape[1]

    a_smem = gl.allocate_shared_memory(dtype, a_desc.block_shape, a_desc.layout)
    b_smem = gl.allocate_shared_memory(dtype, b_desc.block_shape, b_desc.layout)
    pid_m = gl.program_id(0)
    pid_n = gl.program_id(1)
    off_m = pid_m * block_m
    off_n = pid_n * block_n

    if CLUSTERED:
        mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
            version=[3, 0], warps_per_cta=[8, 1], instr_shape=[16, 256, 16],
            cga_layout=[[1, 0]],
        )
    else:
        mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
            version=[3, 0], warps_per_cta=[8, 1], instr_shape=[16, 256, 16],
        )
    acc = gl.zeros((block_m, block_n), dtype=gl.float32, layout=mma_layout)

    bar = mbarrier.allocate_mbarrier()
    mbarrier.init(bar, count=1)
    phase = 0
    for k in range(0, k_size, block_k):
        mbarrier.expect(bar, a_desc.nbytes_per_cta + b_desc.nbytes_per_cta)
        tma.async_load(a_desc, [off_m, k], bar, a_smem)
        tma.async_load(b_desc, [off_n, k], bar, b_smem, multicast=CLUSTERED)
        mbarrier.wait(bar, phase=phase, deps=[a_smem, b_smem])
        phase ^= 1
        acc = warpgroup_mma(a_smem, b_smem.permute((1, 0)), acc, is_async=True)
        acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))

    mbarrier.invalidate(bar)
    c_smem = gl.allocate_shared_memory(dtype, c_desc.block_shape, c_desc.layout)
    c_smem.store(acc.to(dtype))
    fence_async_shared()
    tma.async_store(c_desc, [off_m, off_n], c_smem)
    tma.store_wait(pendings=0)


def launch(a, b, c, *, clustered: bool):
    """Launch for A[M,K], physical B[N,K], C[M,N], all BF16 contiguous."""
    import torch

    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16 or c.dtype != torch.bfloat16:
        raise ValueError("expected BF16 A, B and C")
    m, k = a.shape
    n, b_k = b.shape
    if k != b_k or c.shape != (m, n):
        raise ValueError("GEMM dimensions disagree")
    if not a.is_contiguous() or not b.is_contiguous() or not c.is_contiguous():
        raise ValueError("expected physically contiguous inputs and output")
    cluster_size = 2 if clustered else 1
    block_m = CTA_M * cluster_size
    if m % block_m or n % CTA_N or k % BLOCK_K:
        raise ValueError("this isolated experiment requires full tiles")
    cga_m = [[1, 0]] if clustered else []
    cga_b = [[0, 0]] if clustered else []
    a_layout = gl.NVMMASharedLayout.get_default_for(
        [block_m, BLOCK_K], gl.bfloat16, cga_layout=cga_m)
    b_layout = gl.NVMMASharedLayout.get_default_for(
        [CTA_N, BLOCK_K], gl.bfloat16, cga_layout=cga_b)
    c_layout = gl.NVMMASharedLayout.get_default_for(
        [block_m, CTA_N], gl.bfloat16, cga_layout=cga_m)
    a_desc = TensorDescriptor.from_tensor(a, [block_m, BLOCK_K], a_layout)
    b_desc = TensorDescriptor.from_tensor(b, [CTA_N, BLOCK_K], b_layout)
    c_desc = TensorDescriptor.from_tensor(c, [block_m, CTA_N], c_layout)
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, CTA_N))
    return gemm[grid](a_desc, b_desc, c_desc, clustered,
                      num_warps=WARPS, num_ctas=cluster_size)
