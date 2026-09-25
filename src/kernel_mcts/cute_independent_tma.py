from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

from .cute_independent import IndependentCuteGemmKernel, validate_independent_cute_gemm


IndependentTmaDebugStage = Literal[
    "compile_only",
    "launch_empty",
    "single_cta_a_load",
    "single_cta_a",
    "single_cta_ab",
    "cluster_ab_no_multicast",
    "cluster_ab_multicast",
]
INDEPENDENT_TMA_DEBUG_STAGES = (
    "compile_only",
    "launch_empty",
    "single_cta_a_load",
    "single_cta_a",
    "single_cta_ab",
    "cluster_ab_no_multicast",
    "cluster_ab_multicast",
)


@dataclass(frozen=True, slots=True)
class IndependentTmaDiagnosticSource:
    configuration_hash: str
    source: str

    @property
    def source_hash(self) -> str:
        return hashlib.sha256(self.source.encode("utf-8")).hexdigest()


def render_independent_tma_copy_diagnostic(
    kernel: IndependentCuteGemmKernel,
    *,
    debug_stage: IndependentTmaDebugStage = "cluster_ab_multicast",
) -> IndependentTmaDiagnosticSource:
    """Render a fixed-cluster TMA round-trip diagnostic from typed state."""

    legality = validate_independent_cute_gemm(kernel)
    if not legality.valid:
        messages = "; ".join(item.message for item in legality.violations)
        raise ValueError(f"cannot render illegal TMA diagnostic: {messages}")
    if debug_stage not in INDEPENDENT_TMA_DEBUG_STAGES:
        raise ValueError(f"unknown independent TMA debug stage {debug_stage!r}")
    mainloop = kernel.mainloop
    active_cluster = (
        (1, 1)
        if debug_stage
        in ("launch_empty", "single_cta_a_load", "single_cta_a", "single_cta_ab")
        else (mainloop.cluster_m, mainloop.cluster_n)
    )
    enable_b = debug_stage not in (
        "launch_empty",
        "single_cta_a_load",
        "single_cta_a",
    )
    enable_multicast = debug_stage in (
        "compile_only",
        "cluster_ab_multicast",
    )
    launch_kernel = debug_stage != "compile_only"
    empty_kernel = debug_stage == "launch_empty"
    load_only = debug_stage == "single_cta_a_load"
    consumer_threads = (
        kernel.consumer.warp_groups_m
        * kernel.consumer.warp_groups_n
        * 128
    )
    source = f'''# Generated independent Hopper TMA copy-only diagnostic.
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute.runtime import from_dlpack

CONFIGURATION_HASH = {kernel.configuration_hash!r}
DEBUG_STAGE = {debug_stage!r}
TILE_SHAPE_MNK = ({mainloop.tile_m}, {mainloop.tile_n}, {mainloop.tile_k})
CLUSTER_SHAPE_MN = {active_cluster!r}
MAINLOOP_STAGES = {mainloop.pipeline_stages}
SWIZZLE_BYTES = {mainloop.a_copy.swizzle_bytes}
THREADS_PER_CTA = {consumer_threads}
ENABLE_B = {enable_b!r}
ENABLE_MULTICAST = {enable_multicast!r}
LAUNCH_KERNEL = {launch_kernel!r}
EMPTY_KERNEL = {empty_kernel!r}
LOAD_ONLY = {load_only!r}


def _select_layout_atom(layout, element_type, major_mode_size, *, loc=None, ip=None):
    selected = _PINNED_LAYOUT_SELECTOR(
        layout, element_type, major_mode_size, loc=loc, ip=ip
    )
    if SWIZZLE_BYTES == 128:
        return selected
    selected_name = getattr(selected, "name", str(selected).rsplit(".", 1)[-1])
    target_name = "MN_SW64" if selected_name.startswith("MN_") else "K_SW64"
    return getattr(type(selected), target_name)


_PINNED_LAYOUT_SELECTOR = sm90_utils.get_smem_layout_atom


class IndependentTmaCopyKernel:
    def __init__(self):
        self.tile_shape_mnk = TILE_SHAPE_MNK
        self.cluster_shape_mn = CLUSTER_SHAPE_MN
        self.threads_per_cta = THREADS_PER_CTA
        self.buffer_align_bytes = 1024

    @cute.jit
    def __call__(self, a, b, a_out, b_out, stream: cuda.CUstream):
        self.dtype = a.element_type
        a_layout = utils.LayoutEnum.from_tensor(a)
        b_layout = utils.LayoutEnum.from_tensor(b)

        pinned_selector = sm90_utils.get_smem_layout_atom
        sm90_utils.get_smem_layout_atom = _select_layout_atom
        a_smem_layout_staged = sm90_utils.make_smem_layout_a(
            a_layout, self.tile_shape_mnk, self.dtype, MAINLOOP_STAGES
        )
        b_smem_layout_staged = sm90_utils.make_smem_layout_b(
            b_layout, self.tile_shape_mnk, self.dtype, MAINLOOP_STAGES
        )
        sm90_utils.get_smem_layout_atom = pinned_selector

        @cute.struct
        class SharedStorage:
            load_barrier: cute.struct.MemRange[cutlass.Int64, 1]
            sA: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(a_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(b_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage
        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        tma_a, tensor_a = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
            a,
            a_smem_layout,
            (TILE_SHAPE_MNK[0], TILE_SHAPE_MNK[2]),
            num_multicast=1,
        )
        b_load_op = (
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp()
            if ENABLE_MULTICAST
            else cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        )
        tma_b, tensor_b = cute.nvgpu.cpasync.make_tiled_tma_atom(
            b_load_op,
            b,
            b_smem_layout,
            (TILE_SHAPE_MNK[1], TILE_SHAPE_MNK[2]),
            num_multicast=CLUSTER_SHAPE_MN[0] if ENABLE_MULTICAST else 1,
        )
        tma_a_out, tensor_a_out = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
            a_out,
            a_smem_layout,
            (TILE_SHAPE_MNK[0], TILE_SHAPE_MNK[2]),
        )
        tma_b_out, tensor_b_out = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
            b_out,
            b_smem_layout,
            (TILE_SHAPE_MNK[1], TILE_SHAPE_MNK[2]),
        )
        cta_layout_mnk = cute.make_layout((*CLUSTER_SHAPE_MN, 1))
        self.kernel(
            tma_a,
            tensor_a,
            tma_b,
            tensor_b,
            tma_a_out,
            tensor_a_out,
            tma_b_out,
            tensor_b_out,
            cta_layout_mnk,
            a_smem_layout_staged,
            b_smem_layout_staged,
        ).launch(
            grid=(*CLUSTER_SHAPE_MN, 1),
            block=(THREADS_PER_CTA, 1, 1),
            cluster=(*CLUSTER_SHAPE_MN, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tma_a,
        tensor_a,
        tma_b,
        tensor_b,
        tma_a_out,
        tensor_a_out,
        tma_b_out,
        tensor_b_out,
        cta_layout_mnk,
        a_smem_layout_staged,
        b_smem_layout_staged,
    ):
        if cutlass.const_expr(EMPTY_KERNEL):
            return
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, _, _ = cute.arch.block_idx()
        cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        cluster_coord = cta_layout_mnk.get_flat_coord(cta_rank)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        load_barrier = storage.load_barrier.data_ptr()
        a_smem = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem = cute.slice_(b_smem_layout_staged, (None, None, 0))
        transaction_bytes = cute.size_in_bytes(self.dtype, a_smem)
        if cutlass.const_expr(ENABLE_B):
            transaction_bytes = transaction_bytes + cute.size_in_bytes(
                self.dtype, b_smem
            )

        if tidx == 0:
            cute.arch.mbarrier_init(load_barrier, 1)
            cute.arch.mbarrier_expect_tx(load_barrier, transaction_bytes)
        cute.arch.mbarrier_init_fence()
        pipeline.sync(barrier_id=1)
        if cute.size(CLUSTER_SHAPE_MN) > 1:
            # Every destination barrier must exist before a multicast can target it.
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

        gA = cute.local_tile(
            tensor_a,
            (TILE_SHAPE_MNK[0], TILE_SHAPE_MNK[2]),
            (None, None),
        )
        gB = cute.local_tile(
            tensor_b,
            (TILE_SHAPE_MNK[1], TILE_SHAPE_MNK[2]),
            (None, None),
        )
        a_cta_layout = cute.make_layout(1)
        b_cta_layout = cute.make_layout(1)
        b_cta_coord = 0
        if cutlass.const_expr(ENABLE_MULTICAST):
            b_cta_layout = cute.make_layout(
                cute.slice_(cta_layout_mnk, (None, 0, 0)).shape
            )
            b_cta_coord = cluster_coord[0]
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_a,
            0,
            a_cta_layout,
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA, 0, 2),
        )
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_b,
            b_cta_coord,
            b_cta_layout,
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB, 0, 2),
        )
        b_mcast_mask = 0
        if cutlass.const_expr(ENABLE_MULTICAST):
            b_mcast_mask = cute.make_layout_image_mask(
                cta_layout_mnk, cluster_coord, mode=0
            )
        if warp_idx == 0:
            cute.copy(
                tma_a,
                tAgA[(None, bidx, 0)],
                tAsA[(None, 0)],
                tma_bar_ptr=load_barrier,
            )
            if cutlass.const_expr(ENABLE_B):
                cute.copy(
                    tma_b,
                    tBgB[(None, 0, 0)],
                    tBsB[(None, 0)],
                    tma_bar_ptr=load_barrier,
                    mcast_mask=b_mcast_mask,
                )
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(load_barrier)

        cute.arch.mbarrier_wait(load_barrier, 0)
        if cutlass.const_expr(LOAD_ONLY):
            return
        cute.arch.fence_proxy("async.shared", space="cta")
        pipeline.sync(barrier_id=1)

        gA_out = cute.local_tile(
            tensor_a_out,
            (TILE_SHAPE_MNK[0], TILE_SHAPE_MNK[2]),
            (None, None),
        )
        tAo_s, tAo_g = cute.nvgpu.cpasync.tma_partition(
            tma_a_out,
            0,
            cute.make_layout(1),
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA_out, 0, 2),
        )
        if warp_idx == 0:
            cute.copy(tma_a_out, tAo_s[(None, 0)], tAo_g[(None, bidx, 0)])

        if cutlass.const_expr(ENABLE_B) and cta_rank == 0:
            gB_out = cute.local_tile(
                tensor_b_out,
                (TILE_SHAPE_MNK[1], TILE_SHAPE_MNK[2]),
                (None, None),
            )
            tBo_s, tBo_g = cute.nvgpu.cpasync.tma_partition(
                tma_b_out,
                0,
                cute.make_layout(1),
                cute.group_modes(sB, 0, 2),
                cute.group_modes(gB_out, 0, 2),
            )
            if warp_idx == 0:
                cute.copy(tma_b_out, tBo_s[(None, 0)], tBo_g[(None, 0, 0)])

        if warp_idx == 0:
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(0)


def run_diagnostic():
    import torch

    print(f"Independent TMA {{DEBUG_STAGE}}: allocating tensors", flush=True)
    torch.manual_seed(0)
    a = torch.randn(
        (TILE_SHAPE_MNK[0] * CLUSTER_SHAPE_MN[0], TILE_SHAPE_MNK[2]),
        device="cuda",
        dtype=torch.bfloat16,
    )
    b = torch.randn(
        (TILE_SHAPE_MNK[1], TILE_SHAPE_MNK[2]),
        device="cuda",
        dtype=torch.bfloat16,
    )
    a_out = torch.zeros_like(a)
    b_out = torch.zeros_like(b)
    tensors = [
        from_dlpack(value, assumed_align=16).mark_layout_dynamic(leading_dim=1)
        for value in (a, b, a_out, b_out)
    ]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    diagnostic = IndependentTmaCopyKernel()
    print(f"Independent TMA {{DEBUG_STAGE}}: JIT started", flush=True)
    compiled = cute.compile(diagnostic, *tensors, stream)
    print(f"Independent TMA {{DEBUG_STAGE}}: JIT completed", flush=True)
    if not LAUNCH_KERNEL:
        return {{
            "status": "compile_only_passed",
            "debug_stage": DEBUG_STAGE,
            "configuration_hash": CONFIGURATION_HASH,
        }}
    print(f"Independent TMA {{DEBUG_STAGE}}: kernel launch started", flush=True)
    compiled(*tensors, stream)
    print(f"Independent TMA {{DEBUG_STAGE}}: kernel launch returned", flush=True)
    print(f"Independent TMA {{DEBUG_STAGE}}: synchronize started", flush=True)
    torch.cuda.synchronize()
    print(f"Independent TMA {{DEBUG_STAGE}}: synchronize completed", flush=True)
    if EMPTY_KERNEL:
        return {{
            "status": "empty_launch_passed",
            "debug_stage": DEBUG_STAGE,
            "configuration_hash": CONFIGURATION_HASH,
        }}
    if LOAD_ONLY:
        return {{
            "status": "load_only_passed",
            "debug_stage": DEBUG_STAGE,
            "configuration_hash": CONFIGURATION_HASH,
        }}
    a_equal = bool(torch.equal(a, a_out))
    b_equal = bool(torch.equal(b, b_out)) if ENABLE_B else None
    passed = a_equal and (b_equal if ENABLE_B else True)
    return {{
        "status": "ok" if passed else "correctness_failed",
        "debug_stage": DEBUG_STAGE,
        "configuration_hash": CONFIGURATION_HASH,
        "a_exact": a_equal,
        "b_exact": b_equal,
        "a_max_error": float((a.float() - a_out.float()).abs().max().item()),
        "b_max_error": (
            float((b.float() - b_out.float()).abs().max().item())
            if ENABLE_B
            else None
        ),
        "cluster_shape_mn": list(CLUSTER_SHAPE_MN),
        "tile_shape_mnk": list(TILE_SHAPE_MNK),
        "swizzle_bytes": SWIZZLE_BYTES,
    }}
'''
    return IndependentTmaDiagnosticSource(kernel.configuration_hash, source)
