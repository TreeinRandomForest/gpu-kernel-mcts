from __future__ import annotations

from dataclasses import replace

import pytest

from kernel_mcts.cute_independent import make_independent_cute_gemm
from kernel_mcts.cute_independent_lowering import (
    lower_independent_cute_gemm_static,
)


def test_static_lowering_is_deterministic_and_tracks_completeness() -> None:
    kernel = make_independent_cute_gemm()

    first = lower_independent_cute_gemm_static(kernel)
    second = lower_independent_cute_gemm_static(kernel)

    assert first == second
    assert first.configuration_hash == kernel.configuration_hash
    assert first.source_hash == second.source_hash
    assert first.executable_kernel is False
    assert "wgmma_tiled_mma" in first.implemented_components
    assert "wgmma_consumer_loop" in first.missing_components
    compile(first.source, "independent_cute_static_lowering.py", "exec")


def test_lowering_uses_concrete_cutlass_451_apis_without_expert_kernel() -> None:
    source = lower_independent_cute_gemm_static(
        make_independent_cute_gemm()
    ).source

    assert "sm90_utils.make_trivial_tiled_mma(" in source
    assert "sm90_utils.make_smem_layout_a(" in source
    assert "CopyBulkTensorTileG2SMulticastOp()" in source
    assert "CopyBulkTensorTileS2GOp()" in source
    assert "pipeline.PipelineTmaAsync.create(" in source
    assert "pipeline.PipelineTmaStore.create(" in source
    assert "def required_api_bindings():" in source
    assert "HopperWgmmaGemmKernel" not in source
    assert "dense_gemm.py" not in source


def test_layout_variant_changes_source_and_configuration_identity() -> None:
    sw128 = lower_independent_cute_gemm_static(
        make_independent_cute_gemm(swizzle_bytes=128)
    )
    sw64 = lower_independent_cute_gemm_static(
        make_independent_cute_gemm(swizzle_bytes=64)
    )

    assert sw128.configuration_hash != sw64.configuration_hash
    assert sw128.source_hash != sw64.source_hash
    assert "SHARED_MEMORY_SWIZZLE_BYTES = 128" in sw128.source
    assert "SHARED_MEMORY_SWIZZLE_BYTES = 64" in sw64.source


def test_illegal_kernel_is_rejected_before_lowering() -> None:
    kernel = make_independent_cute_gemm()
    invalid = replace(
        kernel,
        consumer=replace(kernel.consumer, warp_groups_m=1),
    )

    with pytest.raises(ValueError, match="WGMMA ownership"):
        lower_independent_cute_gemm_static(invalid)
