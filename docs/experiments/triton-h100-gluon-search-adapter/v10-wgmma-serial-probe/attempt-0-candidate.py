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

    a_smem_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        dtype=gl.bfloat16, swizzle=128, element_bitwidth=16, rank=2
    )
    b_smem_layout: gl.constexpr = a_smem_layout
    c_smem_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        dtype=gl.bfloat16, swizzle=128, element_bitwidth=16, rank=2
    )

    a_smem = gl.allocate_shared((BM, BK), gl.bfloat16, a_smem_layout)
    b_smem = gl.allocate_shared((BN, BK), gl.bfloat16, b_smem_layout)
    c_smem = gl.allocate_shared((BM, BN), gl.bfloat16, c_smem_layout)

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
        tma.async_load(a_desc, [off_m, step * BK], a_smem, bar=bar)
        tma.async_load(b_desc, [off_n, step * BK], b_smem, bar=bar)
        mbarrier.wait(bar, phase=step % 2, deps=[a_smem, b_smem])

        acc = warpgroup_mma(a_smem, b_smem.permute((1, 0)), acc, is_async=True)
        warpgroup_mma_wait(num_outstanding=0, deps=(acc,))

    acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))
    mbarrier.invalidate(bar)

    out_layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
    out = gl.convert_layout(acc, layout=out_layout).to(gl.bfloat16)

    row_c = off_m + gl.arange(0, BM, gl.SliceLayout(1, out_layout))
    col_c = off_n + gl.arange(0, BN, gl.SliceLayout(0, out_layout))
    c_mask = (row_c[:, None] < M) & (col_c[None, :] < N)
    c_ptrs_local = gl.tensor(
        (
            row_c[:, None] * 0
            + col_c[None, :] * 0
            + (
                gl.arange(0, BM, gl.SliceLayout(1, out_layout))[:, None] * BN
                + gl.arange(0, BN, gl.SliceLayout(0, out_layout))[None, :]
            ).broadcast_to((BM, BN))
        ),
        dtype=gl.int32,
    )
    # store into C shared memory tile
    s_row = gl.arange(0, BM, gl.SliceLayout(1, out_layout))
    s_col = gl.arange(0, BN, gl.SliceLayout(0, out_layout))
    c_smem[s_row[:, None], s_col[None, :]] = out

    fence_async_shared()
    tma.async_store(c_desc, [off_m, off_n], c_smem)
    tma.store_wait(pendings=0)
    tma.store_commit()


def launch(A, B, C, M, N, K):
    BM, BN, BK = 128, 256, 64

    assert A.dtype == triton.bfloat16 and C.dtype == triton.bfloat16
    assert A.is_contiguous() and C.is_contiguous()
    assert B.stride(0) == 1 and B.stride(1) == K

    BT = B.T  # zero-copy [N, K] contiguous physical view

    a_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        dtype=gl.bfloat16, swizzle=128, element_bitwidth=16, rank=2
    )
    b_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        dtype=gl.bfloat16, swizzle=128, element_bitwidth=16, rank=2
    )
    c_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        dtype=gl.bfloat16, swizzle=128, element_bitwidth=16, rank=2
    )

    a_desc = TensorDescriptor.from_tensor(A, layout=a_layout, block_shape=[BM, BK])
    b_desc = TensorDescriptor.from_tensor(BT, layout=b_layout, block_shape=[BN, BK])
    c_desc = TensorDescriptor.from_tensor(C, layout=c_layout, block_shape=[BM, BN])

    return gluon_gemm_tma_wgmma[(triton.cdiv(M, BM), triton.cdiv(N, BN))](
        a_desc, b_desc, c_desc, M, N, K, BM, BN, BK, num_warps=8
    )