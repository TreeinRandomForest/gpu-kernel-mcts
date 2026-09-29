from __future__ import annotations

import pytest

from kernel_mcts.cute_persistent import render_persistent_ownership_diagnostic


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
