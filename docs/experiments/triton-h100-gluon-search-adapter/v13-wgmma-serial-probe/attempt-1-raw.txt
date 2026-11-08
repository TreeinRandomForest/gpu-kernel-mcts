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
def gluon_gemm_tma(A, B, C, M, N, K, a_desc, b_desc, c_desc):
    pid_m = gl.program_id(0)
    pid_n = gl.program_id(1)

    a_smem = gl.allocate_shared_memory(a_desc.dtype, a_desc.block_shape, a_desc.layout)
    b_smem = gl.allocate_shared_memory(b_desc.dtype, b_desc.block_shape, b_desc.layout)
    c_smem = gl.allocate_shared_memory(c_desc.dtype, c_desc.block_shape, c_desc.layout)

    bars = mbarrier.allocate_mbarrier(batch=1)
    bar = bars.index(0)
    mbarrier.init(bar, count=1)

    off_m = pid_m * a_desc.block_shape[0]
    off_n = pid_n * b_desc.block_shape[0]
    BK: gl.constexpr = a_desc.block_shape[1]

    mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0], warps_per_cta=[8, 1], instr_shape=[16, 256, 16]
    )

    acc = warpgroup_mma_init(
        gl.zeros((128, 256), gl.float32, layout=mma_layout)
    )

    for step in range(0, (K + BK - 1) // BK):
        mbarrier.expect(bar, a_desc.nbytes_per_cta + b_desc.nbytes_per_cta)
        tma.async_load(a_desc, [off_m, step * BK], bar, a_smem)
        tma.async_load(b_desc, [off_n, step * BK], bar, b_smem)
        mbarrier.wait(bar, phase=step % 2, deps=[a_smem, b_smem])

        warpgroup_mma(a_smem, b_smem.permute((1, 0)), acc, is_async=True)
        warpgroup_mma_wait(num_outstanding=0, deps=(acc,))

    acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))
    mbarrier.invalidate(bar)

    out = gl.convert_layout(acc, layout=c_desc.layout)
    c_smem.store(out.to(gl.bfloat16))
    fence_async_shared()
    tma.async_store(c_desc, [off_m, off_n], c_smem)
    tma.store_wait(pendings=0)


def launch(A, B, C, M, N, K):
    BM, BN, BK = 128, 256, 64

    assert A.dtype == torch.bfloat16 and C.dtype == torch.bfloat16
    assert A.is_contiguous() and C.is_contiguous()
    assert B.stride(0) == 1 and B.stride(1) == K

    a_layout = gl.NVMMASharedLayout.get_default_for([BM, BK], gl.bfloat16)
    b_layout = gl.NVMMASharedLayout.get_default_for([BN, BK], gl.bfloat16)
    c_layout = gl.NVMMASharedLayout.get_default_for([BM, BN], gl.bfloat16)

    a_desc = TensorDescriptor.from_tensor(A, [BM, BK], a_layout)
    b_desc = TensorDescriptor.from_tensor(B.T, [BN, BK], b_layout)
    c_desc = TensorDescriptor.from_tensor(C, [BM, BN], c_layout)

    return gluon_gemm_tma[(triton.cdiv(M, BM), triton.cdiv(N, BN))](
        A, B, C, M, N, K, a_desc, b_desc, c_desc, num_warps=8
    )