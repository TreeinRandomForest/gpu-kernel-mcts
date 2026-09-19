from __future__ import annotations

import pytest

from kernel_mcts.cute_schedule import (
    DEFAULT_CUTE_SCHEDULE,
    CuteSchedule,
    cute_schedule_from_mapping,
    enumerate_cute_schedules,
    validate_cute_schedule,
)


def test_enumerates_small_deterministic_unique_space_including_baseline() -> None:
    first = enumerate_cute_schedules()
    second = enumerate_cute_schedules()

    assert first == second
    assert len(first) == 12
    assert len(set(first)) == len(first)
    assert DEFAULT_CUTE_SCHEDULE in first
    assert first[0] == CuteSchedule(64, 128, 1, 1)


def test_rejects_values_outside_declared_space_and_bad_workload_coverage() -> None:
    schedule = CuteSchedule(96, 128, 1, 1)

    assert validate_cute_schedule(schedule) == (
        "unsupported CTA tile (96, 128)",
        "CTA tile M does not divide the fixed workload M",
    )


def test_rejects_cluster_that_does_not_cover_tile_grid_cleanly() -> None:
    schedule = CuteSchedule(128, 256, 3, 1)

    assert validate_cute_schedule(schedule) == (
        "unsupported cluster shape (3, 1)",
        "M tile grid is not divisible by cluster M",
    )


def test_schedule_serialization_and_identity_are_stable() -> None:
    schedule = CuteSchedule(128, 256, 1, 1)

    assert schedule.as_dict() == {
        "tile_m": 128,
        "tile_n": 256,
        "cluster_m": 1,
        "cluster_n": 1,
    }
    assert schedule.configuration_id == CuteSchedule(128, 256, 1, 1).configuration_id
    assert schedule.configuration_id != CuteSchedule(128, 128, 1, 1).configuration_id


def test_mapping_requires_only_schedule_fields() -> None:
    with pytest.raises(ValueError, match="fields must be exactly"):
        cute_schedule_from_mapping(
            {
                "tile_m": 128,
                "tile_n": 256,
                "cluster_m": 1,
                "cluster_n": 1,
                "dtype": 16,
            }
        )
