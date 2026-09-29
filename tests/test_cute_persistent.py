from __future__ import annotations

import pytest

from kernel_mcts.cute_persistent import (
    iter_persistent_stage_uses,
    render_persistent_barrier_diagnostic,
    render_persistent_tma_diagnostic,
    render_persistent_ownership_diagnostic,
    validate_persistent_stage_schedule,
)


def test_persistent_ownership_diagnostic_is_deterministic_and_inspectable() -> None:
    first = render_persistent_ownership_diagnostic()
    second = render_persistent_ownership_diagnostic()

    assert first == second
    assert first.source_hash == second.source_hash
    assert "GRID_M = 32" in first.source
    assert "GRID_N = 16" in first.source
    assert "utils.StaticPersistentTileScheduler.create(" in first.source
    assert "scheduler.advance_to_next_work()" in first.source
    assert '"exactly_once": exactly_once' in first.source
    compile(first.source, "persistent_ownership.py", "exec")


@pytest.mark.parametrize(("grid_m", "grid_n"), ((0, 16), (32, 0), (-1, 16)))
def test_persistent_ownership_diagnostic_rejects_invalid_grid(
    grid_m: int, grid_n: int
) -> None:
    with pytest.raises(ValueError, match="grid extents must be positive"):
        render_persistent_ownership_diagnostic(grid_m=grid_m, grid_n=grid_n)


def test_persistent_ownership_diagnostic_tracks_requested_grid() -> None:
    rendered = render_persistent_ownership_diagnostic(grid_m=7, grid_n=5)

    assert "GRID_M = 7" in rendered.source
    assert "GRID_N = 5" in rendered.source
    assert "LOGICAL_TILE_COUNT = GRID_M * GRID_N" in rendered.source


def test_persistent_stage_schedule_carries_phase_across_work_items() -> None:
    uses = list(
        iter_persistent_stage_uses(
            work_items=2,
            k_tiles_per_work=64,
            pipeline_stages=3,
        )
    )

    assert uses[0].global_k_tile == 0
    assert (uses[0].stage, uses[0].phase) == (0, 0)
    assert uses[63].global_k_tile == 63
    assert (uses[63].stage, uses[63].phase) == (0, 1)
    assert uses[64].work_index == 1
    assert uses[64].k_tile == 0
    assert uses[64].global_k_tile == 64
    assert (uses[64].stage, uses[64].phase) == (1, 1)
    assert validate_persistent_stage_schedule(
        work_items=4,
        k_tiles_per_work=64,
        pipeline_stages=3,
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"work_items": -1, "k_tiles_per_work": 64, "pipeline_stages": 3}, "work-item"),
        ({"work_items": 1, "k_tiles_per_work": 0, "pipeline_stages": 3}, "K-tile"),
        ({"work_items": 1, "k_tiles_per_work": 64, "pipeline_stages": 0}, "pipeline-stage"),
    ),
)
def test_persistent_stage_schedule_rejects_invalid_extents(
    kwargs: dict[str, int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        list(iter_persistent_stage_uses(**kwargs))


def test_persistent_barrier_diagnostic_carries_global_phase_in_lockstep() -> None:
    rendered = render_persistent_barrier_diagnostic()

    assert "K_TILES_PER_WORK = 64" in rendered.source
    assert "PIPELINE_STAGES = 3" in rendered.source
    assert rendered.source.count("StaticPersistentTileScheduler.create(") == 2
    assert "producer.num_tiles_executed * K_TILES_PER_WORK" in rendered.source
    assert "consumer.num_tiles_executed * K_TILES_PER_WORK" in rendered.source
    assert '"producer_consumer_lockstep": lockstep' in rendered.source
    compile(rendered.source, "persistent_barriers.py", "exec")


@pytest.mark.parametrize(
    "kwargs",
    (
        {"grid_m": 0},
        {"grid_n": 0},
        {"k_tiles_per_work": 0},
        {"pipeline_stages": 0},
    ),
)
def test_persistent_barrier_diagnostic_rejects_invalid_extents(
    kwargs: dict[str, int]
) -> None:
    with pytest.raises(ValueError):
        render_persistent_barrier_diagnostic(**kwargs)


def test_persistent_tma_diagnostic_verifies_coordinates_and_payloads() -> None:
    rendered = render_persistent_tma_diagnostic()

    assert "TILE_SHAPE_MNK = (128, 256, 64)" in rendered.source
    assert "CopyBulkTensorTileG2SOp()" in rendered.source
    assert "samples[(worker, slot, 0)] = sA[(0, 0, stage)]" in rendered.source
    assert '"payload_exact": payload_exact' in rendered.source
    compile(rendered.source, "persistent_tma.py", "exec")


def test_persistent_wgmma_issue_resets_accumulators_per_work_item() -> None:
    rendered = render_persistent_tma_diagnostic(enable_wgmma_issue=True)

    assert "ENABLE_WGMMA_ISSUE = True" in rendered.source
    assert "THREADS_PER_CTA = 384 if ENABLE_WGMMA_ISSUE else 256" in rendered.source
    assert "accumulators.fill(0.0)" in rendered.source
    assert "Field.ACCUMULATE, False" in rendered.source
    assert "cute.nvgpu.warpgroup.wait_group(0)" in rendered.source
    assert "consumer_barrier.arrive_and_wait()" in rendered.source
    assert '"wgmma_issued": ENABLE_WGMMA_ISSUE' in rendered.source
    compile(rendered.source, "persistent_wgmma_issue.py", "exec")


def test_persistent_epilogue_stores_all_wide_cta_accumulator_tiles() -> None:
    rendered = render_persistent_tma_diagnostic(
        enable_wgmma_issue=True, enable_epilogue=True
    )

    assert "ENABLE_EPILOGUE = True" in rendered.source
    assert "EPILOGUE_STAGES = 8" in rendered.source
    assert "epi_buffer = warp_group_idx * 4 + local_epi_index" in rendered.source
    assert "for epi_index in cutlass.range_constexpr(8)" in rendered.source
    assert '"numerical_correct": numerical_correct' in rendered.source
    compile(rendered.source, "persistent_epilogue.py", "exec")


def test_persistent_epilogue_requires_wgmma() -> None:
    with pytest.raises(ValueError, match="requires WGMMA"):
        render_persistent_tma_diagnostic(enable_epilogue=True)


def test_persistent_full_k_carries_global_stage_phase_and_accumulates() -> None:
    rendered = render_persistent_tma_diagnostic(
        enable_wgmma_issue=True, enable_epilogue=True, full_k=True
    )

    assert "FULL_K = True" in rendered.source
    assert "K_TILES_PER_WORK = 64 if FULL_K else 1" in rendered.source
    assert "producer.num_tiles_executed * K_TILES_PER_WORK" in rendered.source
    assert "global_k_tile = slot * K_TILES_PER_WORK + k_tile" in rendered.source
    assert "import tempfile" in rendered.source
    assert "import ctypes" in rendered.source
    assert "from pathlib import Path" in rendered.source
    assert "import statistics" in rendered.source
    assert "import subprocess" in rendered.source
    assert '"/usr/local/bin/kernel-mcts-bf16-inputs"' in rendered.source
    consumer_loop = rendered.source[rendered.source.index("accumulators.fill(0.0)") :]
    assert consumer_loop.index("accumulators.fill(0.0)") < consumer_loop.index(
        "for k_tile in cutlass.range("
    )
    assert '"full_k": FULL_K' in rendered.source
    compile(rendered.source, "persistent_full_k.py", "exec")


@pytest.mark.parametrize("kwargs", ({"grid_m": 0}, {"grid_n": 0}, {"pipeline_stages": 0}))
def test_persistent_tma_diagnostic_rejects_invalid_extents(
    kwargs: dict[str, int]
) -> None:
    with pytest.raises(ValueError):
        render_persistent_tma_diagnostic(**kwargs)
