from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .cute_independent import (
    IndependentCuteGemmKernel,
    validate_independent_cute_gemm,
)


@dataclass(frozen=True, slots=True)
class IndependentCuteLowering:
    """Concrete CUTLASS 4.5.1 source for the static lowering checkpoint."""

    configuration_hash: str
    source: str
    implemented_components: tuple[str, ...]
    missing_components: tuple[str, ...]

    @property
    def source_hash(self) -> str:
        return hashlib.sha256(self.source.encode("utf-8")).hexdigest()

    @property
    def executable_kernel(self) -> bool:
        return not self.missing_components

    def as_dict(self) -> dict[str, object]:
        return {
            "configuration_hash": self.configuration_hash,
            "source_hash": self.source_hash,
            "implemented_components": list(self.implemented_components),
            "missing_components": list(self.missing_components),
            "executable_kernel": self.executable_kernel,
        }


def lower_independent_cute_gemm_static(
    kernel: IndependentCuteGemmKernel,
) -> IndependentCuteLowering:
    """Lower typed structure to real CuTe constructors, short of a device loop.

    The generated module intentionally contains no import of NVIDIA's dense-GEMM
    kernel class. It binds the independent state directly to the pinned CUTLASS
    4.5.1 APIs that a subsequent device-kernel lowering will consume.
    """

    legality = validate_independent_cute_gemm(kernel)
    if not legality.valid:
        messages = "; ".join(item.message for item in legality.violations)
        raise ValueError(f"cannot lower illegal independent CuTe GEMM: {messages}")

    mainloop = kernel.mainloop
    consumer = kernel.consumer
    epilogue = kernel.epilogue
    swizzle = mainloop.a_copy.swizzle_bytes
    source = f'''# Generated independent Hopper GEMM static lowering.
# CUTLASS/CuTe DSL API target: 4.5.1.
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils

CONFIGURATION_HASH = {kernel.configuration_hash!r}
TILE_SHAPE_MNK = ({mainloop.tile_m}, {mainloop.tile_n}, {mainloop.tile_k})
CLUSTER_SHAPE_MN = ({mainloop.cluster_m}, {mainloop.cluster_n})
ATOM_LAYOUT_MNK = ({consumer.warp_groups_m}, {consumer.warp_groups_n}, 1)
MAINLOOP_STAGES = {mainloop.pipeline_stages}
EPILOGUE_STAGES = {epilogue.pipeline_stages}
EPILOGUE_TILE = ({epilogue.tile_m}, {epilogue.tile_n})
SHARED_MEMORY_SWIZZLE_BYTES = {swizzle}
THREADS_PER_WARP_GROUP = 128
THREADS_PER_CTA = THREADS_PER_WARP_GROUP * {consumer.warp_groups_m * consumer.warp_groups_n}


def _select_layout_atom(layout, element_type, major_mode_size):
    selected = sm90_utils.get_smem_layout_atom(
        layout, element_type, major_mode_size
    )
    if SHARED_MEMORY_SWIZZLE_BYTES == 128:
        return selected
    selected_name = getattr(selected, "name", str(selected).rsplit(".", 1)[-1])
    target_name = "MN_SW64" if selected_name.startswith("MN_") else "K_SW64"
    return getattr(type(selected), target_name)


def make_static_components(a_dtype, b_dtype, c_dtype, acc_dtype, a_layout, b_layout, c_layout):
    tiled_mma = sm90_utils.make_trivial_tiled_mma(
        a_dtype,
        b_dtype,
        a_layout.sm90_mma_major_mode(),
        b_layout.sm90_mma_major_mode(),
        acc_dtype,
        ATOM_LAYOUT_MNK,
        tiler_mn=(64, TILE_SHAPE_MNK[1]),
    )
    cta_layout_mnk = cute.make_layout((*CLUSTER_SHAPE_MN, 1))

    pinned_selector = sm90_utils.get_smem_layout_atom
    try:
        sm90_utils.get_smem_layout_atom = _select_layout_atom
        a_smem_layout_staged = sm90_utils.make_smem_layout_a(
            a_layout, TILE_SHAPE_MNK, a_dtype, MAINLOOP_STAGES
        )
        b_smem_layout_staged = sm90_utils.make_smem_layout_b(
            b_layout, TILE_SHAPE_MNK, b_dtype, MAINLOOP_STAGES
        )
        epi_smem_layout_staged = sm90_utils.make_smem_layout_epi(
            c_dtype, c_layout, EPILOGUE_TILE, EPILOGUE_STAGES
        )
    finally:
        sm90_utils.get_smem_layout_atom = pinned_selector

    return (
        tiled_mma,
        cta_layout_mnk,
        a_smem_layout_staged,
        b_smem_layout_staged,
        epi_smem_layout_staged,
    )


def make_tma_load(tensor, smem_layout_staged, smem_tile, multicast_count):
    op = (
        cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        if multicast_count == 1
        else cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp()
    )
    smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
    return cute.nvgpu.cpasync.make_tiled_tma_atom(
        op,
        tensor,
        smem_layout,
        smem_tile,
        num_multicast=multicast_count,
    )


def make_tma_store(tensor_c, epi_smem_layout_staged):
    epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
    return cute.nvgpu.cpasync.make_tiled_tma_atom(
        cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
        tensor_c,
        epi_smem_layout,
        EPILOGUE_TILE,
    )


def make_mainloop_pipeline(barrier_storage, tx_count):
    producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
    consumer_group = pipeline.CooperativeGroup(
        pipeline.Agent.Thread, THREADS_PER_CTA * {mainloop.cluster_m}
    )
    return pipeline.PipelineTmaAsync.create(
        barrier_storage=barrier_storage,
        num_stages=MAINLOOP_STAGES,
        producer_group=producer_group,
        consumer_group=consumer_group,
        tx_count=tx_count,
        cta_layout_vmnk=cute.make_layout((1, *CLUSTER_SHAPE_MN, 1)),
        defer_sync=True,
    )


def make_epilogue_pipeline():
    return pipeline.PipelineTmaStore.create(
        num_stages=EPILOGUE_STAGES,
        producer_group=pipeline.CooperativeGroup(
            pipeline.Agent.Thread, THREADS_PER_CTA
        ),
    )


def required_api_bindings():
    return (
        sm90_utils.make_trivial_tiled_mma,
        sm90_utils.make_smem_layout_a,
        sm90_utils.make_smem_layout_b,
        sm90_utils.make_smem_layout_epi,
        cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp,
        cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp,
        cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp,
        cute.nvgpu.cpasync.make_tiled_tma_atom,
        pipeline.PipelineTmaAsync.create,
        pipeline.PipelineTmaStore.create,
    )
'''
    return IndependentCuteLowering(
        configuration_hash=kernel.configuration_hash,
        source=source,
        implemented_components=(
            "wgmma_tiled_mma",
            "cluster_layout",
            "shared_memory_layouts",
            "tma_load_atoms",
            "tma_store_atom",
            "mainloop_pipeline_constructor",
            "epilogue_pipeline_constructor",
        ),
        missing_components=(
            "shared_storage_struct",
            "cta_and_cluster_coordinates",
            "multicast_masks",
            "tma_producer_loop",
            "wgmma_consumer_loop",
            "register_to_shared_epilogue",
            "tma_store_loop",
            "kernel_launch",
        ),
    )
