from __future__ import annotations

from dataclasses import replace

import pytest

from kernel_mcts.cute_independent import make_independent_cute_gemm
from kernel_mcts.cute_independent_tma import render_independent_tma_copy_diagnostic


def test_tma_diagnostic_is_deterministic_and_typed() -> None:
    kernel = make_independent_cute_gemm()
    first = render_independent_tma_copy_diagnostic(kernel)
    second = render_independent_tma_copy_diagnostic(kernel)

    assert first == second
    assert first.configuration_hash == kernel.configuration_hash
    assert first.source_hash == second.source_hash
    compile(first.source, "independent_tma_diagnostic.py", "exec")


def test_tma_diagnostic_lowers_cluster_multicast_and_exact_round_trip() -> None:
    source = render_independent_tma_copy_diagnostic(
        make_independent_cute_gemm()
    ).source

    assert "CopyBulkTensorTileG2SMulticastOp()" in source
    assert "cute.make_layout_image_mask(" in source
    assert "mcast_mask=b_mcast_mask" in source
    assert "cute.arch.mbarrier_expect_tx(" in source
    assert "if warp_idx == 0:" in source
    assert "with cute.arch.elect_one():" in source
    assert "if tidx == 0:\n            cute.copy(" not in source
    assert source.index("cute.arch.cluster_wait()") < source.index(
        "mcast_mask=b_mcast_mask"
    )
    assert '"a_exact": a_equal' in source
    assert '"b_exact": b_equal' in source
    assert "HopperWgmmaGemmKernel" not in source


@pytest.mark.parametrize(
    ("stage", "cluster", "enable_b", "enable_multicast", "launch"),
    (
        ("compile_only", "(2, 1)", True, True, False),
        ("launch_empty", "(1, 1)", False, False, True),
        ("single_cta_a_load", "(1, 1)", False, False, True),
        ("single_cta_a", "(1, 1)", False, False, True),
        ("single_cta_ab", "(1, 1)", True, False, True),
        ("cluster_ab_no_multicast", "(2, 1)", True, False, True),
        ("cluster_ab_multicast", "(2, 1)", True, True, True),
    ),
)
def test_tma_debug_stages_render_bounded_progression(
    stage: str,
    cluster: str,
    enable_b: bool,
    enable_multicast: bool,
    launch: bool,
) -> None:
    source = render_independent_tma_copy_diagnostic(
        make_independent_cute_gemm(),
        debug_stage=stage,
    ).source

    assert f"DEBUG_STAGE = {stage!r}" in source
    assert f"CLUSTER_SHAPE_MN = {cluster}" in source
    assert f"ENABLE_B = {enable_b!r}" in source
    assert f"ENABLE_MULTICAST = {enable_multicast!r}" in source
    assert f"LAUNCH_KERNEL = {launch!r}" in source
    assert f"EMPTY_KERNEL = {(stage == 'launch_empty')!r}" in source
    assert f"LOAD_ONLY = {(stage == 'single_cta_a_load')!r}" in source
    compile(source, f"independent_tma_{stage}.py", "exec")


def test_tma_diagnostic_identity_changes_with_swizzle() -> None:
    sw128 = render_independent_tma_copy_diagnostic(
        make_independent_cute_gemm(swizzle_bytes=128)
    )
    sw64 = render_independent_tma_copy_diagnostic(
        make_independent_cute_gemm(swizzle_bytes=64)
    )

    assert sw128.configuration_hash != sw64.configuration_hash
    assert sw128.source_hash != sw64.source_hash


def test_tma_diagnostic_rejects_illegal_execution_contract() -> None:
    kernel = make_independent_cute_gemm()
    buffers = list(kernel.execution.buffers)
    buffers[0] = replace(buffers[0], stages=2)
    invalid = replace(
        kernel,
        execution=replace(kernel.execution, buffers=tuple(buffers)),
    )

    with pytest.raises(ValueError, match="buffer rings"):
        render_independent_tma_copy_diagnostic(invalid)
