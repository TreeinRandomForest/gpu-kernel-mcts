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
        ("compile_only", "(1, 1)", True, False, False),
        ("launch_empty", "(1, 1)", False, False, True),
        ("single_cta_a_load", "(1, 1)", False, False, True),
        ("single_cta_a", "(1, 1)", False, False, True),
        ("single_cta_ab", "(1, 1)", True, False, True),
        ("cluster_ab_no_multicast", "(2, 1)", True, False, True),
        ("cluster_ab_multicast", "(2, 1)", True, True, True),
        ("wgmma_compile_only", "(1, 1)", True, False, False),
        ("wgmma_issue_only", "(1, 1)", True, False, True),
        ("wgmma_r2s_first_tile", "(1, 1)", True, False, True),
        ("wgmma_r2s_two_tiles", "(1, 1)", True, False, True),
        ("wgmma_r2s_three_tiles", "(1, 1)", True, False, True),
        ("wgmma_r2s_four_reuse_zero", "(1, 1)", True, False, True),
        ("wgmma_r2s_four_padded", "(1, 1)", True, False, True),
        ("wgmma_r2s_four_tiles", "(1, 1)", True, False, True),
        ("wgmma_r2s_only", "(1, 1)", True, False, True),
        ("wgmma_one_group", "(1, 1)", True, False, True),
        ("wgmma_two_group", "(1, 1)", True, False, True),
        ("wgmma_full_k", "(1, 1)", True, False, True),
        ("wgmma_full_workload", "(1, 1)", True, False, True),
        ("wgmma_one_k_no_reuse", "(1, 1)", True, False, True),
        ("wgmma_one_k", "(1, 1)", True, False, True),
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
    assert f"ENABLE_WGMMA = {stage.startswith('wgmma_')!r}" in source
    assert f"WGMMA_ISSUE_ONLY = {(stage == 'wgmma_issue_only')!r}" in source
    expected_r2s_only = stage in (
        "wgmma_r2s_first_tile",
        "wgmma_r2s_two_tiles",
        "wgmma_r2s_three_tiles",
        "wgmma_r2s_four_reuse_zero",
        "wgmma_r2s_four_padded",
        "wgmma_r2s_four_tiles",
        "wgmma_r2s_only",
    )
    assert f"WGMMA_R2S_ONLY = {expected_r2s_only!r}" in source
    expected_first_tile = stage == "wgmma_r2s_first_tile"
    assert f"WGMMA_R2S_FIRST_TILE = {expected_first_tile!r}" in source
    expected_two_tiles = stage == "wgmma_r2s_two_tiles"
    assert f"WGMMA_R2S_TWO_TILES = {expected_two_tiles!r}" in source
    expected_three_tiles = stage == "wgmma_r2s_three_tiles"
    assert f"WGMMA_R2S_THREE_TILES = {expected_three_tiles!r}" in source
    expected_reuse_zero = stage == "wgmma_r2s_four_reuse_zero"
    assert f"WGMMA_R2S_FOUR_REUSE_ZERO = {expected_reuse_zero!r}" in source
    expected_padded = stage == "wgmma_r2s_four_padded"
    assert f"WGMMA_R2S_FOUR_PADDED = {expected_padded!r}" in source
    expected_four_tiles = stage == "wgmma_r2s_four_tiles"
    assert f"WGMMA_R2S_FOUR_TILES = {expected_four_tiles!r}" in source
    compile(source, f"independent_tma_{stage}.py", "exec")

    expected_epilogue_stages = 8 if stage == "wgmma_one_k_no_reuse" else 4
    assert f"EPILOGUE_STAGES = {expected_epilogue_stages}" in source


def test_wgmma_diagnostic_lowers_partition_gemm_and_tma_epilogue() -> None:
    source = render_independent_tma_copy_diagnostic(
        make_independent_cute_gemm(),
        debug_stage="wgmma_one_k",
    ).source

    assert "sm90_utils.make_trivial_tiled_mma(" in source
    assert "warp_group_thread_layout = cute.make_layout(" in source
    assert "warp_group_thread_layout(warp_group_idx)" in source
    assert "tCsA = thr_mma.partition_A(sA)" in source
    assert "tCsB = thr_mma.partition_B(sB)" in source
    assert "tCrA = tiled_mma.make_fragment_A(tCsA)" in source
    assert "tCrB = tiled_mma.make_fragment_B(tCsB)" in source
    assert "cute.make_rmem_tensor(tCgC.shape, self.acc_dtype)" in source
    assert "cute.nvgpu.warpgroup.fence()" in source
    assert "cute.nvgpu.warpgroup.commit_group()" in source
    assert "cute.nvgpu.warpgroup.wait_group(0)" in source
    assert "cute.arch.sync_threads()" in source
    assert "sm90_utils.sm90_get_smem_store_op(" in source
    assert "tiled_copy_r2s.retile(accumulators)" in source
    assert "epi_index * rC_size + value_index" in source
    assert "if cutlass.const_expr(WGMMA_R2S_FOUR_REUSE_ZERO):" in source
    assert "EPILOGUE_STORAGE_ELEMENTS = 20480" in source
    assert "self.c_dtype,\n                    EPILOGUE_STORAGE_ELEMENTS" in source
    assert "rC_out.store(rC.load().to(self.c_dtype))" in source
    assert "epi_tile_count = cute.size(gC_for_tma, mode=[1])" in source
    assert "pipeline.PipelineTmaStore.create(" in source
    assert "stride=(epi_tile_shape[1], 1)" in source
    assert "c_pipeline.producer_tail()" in source
    assert "if cutlass.const_expr(not WGMMA_R2S_ONLY):" in source
    assert '"output_nonzero": output_nonzero' in source
    assert '"cosine_similarity": cosine_similarity' in source
    assert '"best_tile_permutation": best_tile_permutation' in source
    assert '"best_transposed_tile_permutation": best_transposed_tile_permutation' in source
    assert "torch.matmul(a.float(), b.float().transpose(0, 1))" in source


def test_wgmma_no_reuse_diagnostic_allocates_eight_stages_plus_guard() -> None:
    source = render_independent_tma_copy_diagnostic(
        make_independent_cute_gemm(),
        debug_stage="wgmma_one_k_no_reuse",
    ).source

    assert "EPILOGUE_STAGES = 8" in source
    assert "EPILOGUE_STORAGE_ELEMENTS = 36864" in source


def test_wgmma_one_group_diagnostic_uses_64_row_tile() -> None:
    source = render_independent_tma_copy_diagnostic(
        make_independent_cute_gemm(),
        debug_stage="wgmma_one_group",
    ).source

    assert "TILE_SHAPE_MNK = (64, 256, 64)" in source
    assert "THREADS_PER_CTA = 128" in source
    assert "MMA_WARP_GROUPS = 1" in source


def test_wgmma_two_group_diagnostic_preserves_experimental_path() -> None:
    source = render_independent_tma_copy_diagnostic(
        make_independent_cute_gemm(),
        debug_stage="wgmma_two_group",
    ).source

    assert "TILE_SHAPE_MNK = (128, 256, 64)" in source
    assert "THREADS_PER_CTA = 256" in source
    assert "MMA_WARP_GROUPS = 2" in source


def test_wgmma_full_k_cycles_the_three_stage_mainloop_ring() -> None:
    source = render_independent_tma_copy_diagnostic(
        make_independent_cute_gemm(),
        debug_stage="wgmma_full_k",
    ).source

    assert "PROBLEM_K = 4096" in source
    assert "FULL_K = True" in source
    assert "FULL_K_TILE_COUNT = PROBLEM_K // TILE_SHAPE_MNK[2]" in source
    assert "stage = k_tile % MAINLOOP_STAGES" in source
    assert "phase = (k_tile // MAINLOOP_STAGES) % 2" in source
    assert "tAgA[(None, bidx, k_tile)]" in source
    assert "tCrA[(None, None, k_block, stage)]" in source


def test_wgmma_full_workload_launches_complete_mn_grid() -> None:
    source = render_independent_tma_copy_diagnostic(
        make_independent_cute_gemm(),
        debug_stage="wgmma_full_workload",
    ).source

    assert "PROBLEM_M = 4096" in source
    assert "PROBLEM_N = 4096" in source
    assert "PROBLEM_K = 4096" in source
    assert "FULL_WORKLOAD = True" in source
    assert "GRID_M = 64" in source
    assert "GRID_N = 16" in source
    assert "grid=(GRID_M, GRID_N, 1)" in source
    assert "(bidx, bidy)" in source
    assert "warmup_count = 0 if PROFILE_SINGLE_LAUNCH else 10" in source
    assert "measurement_count = 1 if PROFILE_SINGLE_LAUNCH else 30" in source
    assert '"median_us": float(statistics.median(timings_us))' in source


def test_two_stage_full_workload_renders_two_stage_barrier_ring() -> None:
    source = render_independent_tma_copy_diagnostic(
        make_independent_cute_gemm(pipeline_stages=2),
        debug_stage="wgmma_full_workload",
    ).source

    assert "MAINLOOP_STAGES = 2" in source
    assert "stage = k_tile % MAINLOOP_STAGES" in source
    assert "phase = (k_tile // MAINLOOP_STAGES) % 2" in source
    compile(source, "independent_two_stage_gemm.py", "exec")


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
