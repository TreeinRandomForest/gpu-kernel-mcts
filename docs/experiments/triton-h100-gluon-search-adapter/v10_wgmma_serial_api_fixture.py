import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import (
    mbarrier,
    tma,
    warpgroup_mma,
    warpgroup_mma_init,
    warpgroup_mma_wait,
    fence_async_shared,
)


@gluon.jit
def gluon_gemm_tma_wgmma(
    a_desc,
    b_desc,
    c_desc,
    M,
    N,
    K,
    BM: gl.constexpr,
    BN: gl.constexpr,
    BK: gl.constexpr,
):
    pid_m = gl.program_id(0)
    pid_n = gl.program_id(1)

    off_m = pid_m * BM
    off_n = pid_n * BN

    a_smem = gl.allocate_shared_memory(gl.bfloat16, a_desc.block_shape, a_desc.layout)
    b_smem = gl.allocate_shared_memory(gl.bfloat16, b_desc.block_shape, b_desc.layout)
    c_smem = gl.allocate_shared_memory(gl.bfloat16, c_desc.block_shape, c_desc.layout)

    mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0], warps_per_cta=[8, 1], instr_shape=[16, 256, 16]
    )

    bars = mbarrier.allocate_mbarrier(batch=1)
    bar = bars.index(0)
    mbarrier.init(bar, count=1)

    acc = warpgroup_mma_init(gl.zeros((BM, BN), gl.float32, layout=mma_layout))

    num_k_blocks = (K + BK - 1) // BK
    for step in range(0, num_k_blocks):
        mbarrier.expect(bar, a_desc.nbytes_per_cta + b_desc.nbytes_per_cta)
        tma.async_load(a_desc, [off_m, step * BK], bar, a_smem)
        tma.async_load(b_desc, [off_n, step * BK], bar, b_smem)
        mbarrier.wait(bar, phase=step % 2, deps=[a_smem, b_smem])

        acc = warpgroup_mma(a_smem, b_smem.permute((1, 0)), acc, is_async=True)
        warpgroup_mma_wait(num_outstanding=0, deps=(acc,))

    acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))
    mbarrier.invalidate(bar)

    c_smem.store(acc.to(gl.bfloat16))

    fence_async_shared()
    tma.async_store(c_desc, [off_m, off_n], c_smem)
    tma.store_wait(pendings=0)


def launch(A, B, C, M, N, K):
    BM, BN, BK = 128, 256, 64

    assert A.dtype == torch.bfloat16 and C.dtype == torch.bfloat16
    assert A.is_contiguous() and C.is_contiguous()
    assert B.stride(0) == 1 and B.stride(1) == K

    BT = B.T

    a_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BM, BK], gl.bfloat16)
    b_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BN, BK], gl.bfloat16)
    c_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BM, BN], gl.bfloat16)

    a_desc = TensorDescriptor.from_tensor(A, [BM, BK], a_layout)
    b_desc = TensorDescriptor.from_tensor(BT, [BN, BK], b_layout)
    c_desc = TensorDescriptor.from_tensor(C, [BM, BN], c_layout)

    return gluon_gemm_tma_wgmma[(triton.cdiv(M, BM), triton.cdiv(N, BN))](
        a_desc, b_desc, c_desc, M, N, K, BM, BN, BK, num_warps=8
    )