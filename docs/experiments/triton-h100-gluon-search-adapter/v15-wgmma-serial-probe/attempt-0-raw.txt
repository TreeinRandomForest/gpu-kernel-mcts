import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (
    mbarrier,
    tma,
    warpgroup_mma,
    warpgroup_mma_init,
    warpgroup_mma_wait,
    fence_async_shared,
)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor


@gluon.jit
def gluon_gemm_tma_wgmma(
    A,
    B,
    C,
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

    a_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BM, BK], gl.bfloat16)
    b_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BN, BK], gl.bfloat16)
    c_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BM, BN], gl.bfloat16)

    a_smem = gl.alloc_shared((BM, BK), gl.bfloat16, layout=a_layout)
    b_smem = gl.alloc_shared((BN, BK), gl.bfloat16, layout=b_layout)
    c_smem = gl.alloc_shared((BM, BN), gl.bfloat16, layout=c_layout)

    mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0], warps_per_cta=[8, 1], instr_shape=[16, 256, 16]
    )

    bars = mbarrier.allocate_mbarrier(batch=1)
    bar = bars.index(0)
    mbarrier.init(bar, count=1)

    acc = warpgroup_mma_init(gl.zeros((BM, BN), gl.float32, layout=mma_layout))

    num_steps = (K + BK - 1) // BK
    for step in range(num_steps):
        mbarrier.expect(bar, a_smem.nbytes + b_smem.nbytes)
        tma.async_load(A, [off_m, step * BK], bar, a_smem)
        tma.async_load(B, [off_n, step * BK], bar, b_smem)
        mbarrier.wait(bar, phase=step % 2, deps=[a_smem, b_smem])

        acc = warpgroup_mma(a_smem, b_smem.permute((1, 0)), acc, is_async=True)
        warpgroup_mma_wait(num_outstanding=0, deps=(acc,))

    acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))
    mbarrier.invalidate(bar)

    c_smem.store(acc.to(gl.bfloat16))
    fence_async_shared()
    tma.async_store(C, [off_m, off_n], c_smem)
    tma.store_wait(pendings=0)


def launch(A, B, C, M, N, K):
    assert A.dtype == triton.bfloat16 and C.dtype == triton.bfloat16
    assert A.is_contiguous() and C.is_contiguous()
    assert B.stride(0) == 1 and B.stride(1) == K
    Bt = B.T

    BM, BN, BK = 128, 256, 64

    a_layout = gl.NVMMASharedLayout.get_default_for([BM, BK], gl.bfloat16)
    b_layout = gl.NVMMASharedLayout.get_default_for([BN, BK], gl.bfloat16)
    c_layout = gl.NVMMASharedLayout.get_default_for([BM, BN], gl.bfloat16)

    a_desc = TensorDescriptor.from_tensor(A, [BM, BK], a_layout)
    b_desc = TensorDescriptor.from_tensor(Bt, [BN, BK], b_layout)
    c_desc = TensorDescriptor.from_tensor(C, [BM, BN], c_layout)

    return gluon_gemm_tma_wgmma[(triton.cdiv(M, BM), triton.cdiv(N, BN))](
        a_desc, b_desc, c_desc, M, N, K, BM, BN, BK, num_warps=8, num_ctas=1
    )