from __future__ import annotations

from kernel_mcts.cute_schedule import DEFAULT_CUTE_SCHEDULE, CuteSchedule
from kernel_mcts.cute_tuning import run_cute_schedule_tuning
from kernel_mcts.domain import BenchmarkResult


def _cublas(median: float = 100.0):
    return {
        "correctness": {"success": True},
        "benchmark": BenchmarkResult((median, median), median),
    }


def _valid_result(median: float):
    return {
        "correctness": {"success": True},
        "benchmark": {"timings_us": [median, median], "median_us": median},
        "comparable_to_repository_baselines": True,
    }


def test_tunes_legal_schedules_in_order_and_reports_best_ratios() -> None:
    schedules = (
        DEFAULT_CUTE_SCHEDULE,
        CuteSchedule(128, 128, 1, 1),
        CuteSchedule(64, 128, 1, 1),
    )
    latencies = {
        schedules[0]: 120.0,
        schedules[1]: 105.0,
        schedules[2]: 110.0,
    }
    calls = []

    def runner(schedule):
        calls.append(schedule)
        return _valid_result(latencies[schedule])

    result = run_cute_schedule_tuning(
        runner, _cublas(), {"manifest_id": "worker-1"}, schedules=schedules
    )

    assert calls == list(schedules)
    assert result["b_tune"] == 3
    assert result["best"]["schedule"] == schedules[1].as_dict()
    assert result["default"]["median_us"] == 120.0
    assert result["comparison"] == {
        "best_latency_ratio_vs_cublas": 1.05,
        "best_speedup_vs_default": 120.0 / 105.0,
    }
    assert [trial["b_tune"] for trial in result["trials"]] == [1, 2, 3]


def test_failed_evaluation_consumes_budget_and_is_retained() -> None:
    schedules = (DEFAULT_CUTE_SCHEDULE, CuteSchedule(128, 128, 1, 1))

    def runner(schedule):
        if schedule == DEFAULT_CUTE_SCHEDULE:
            raise RuntimeError("JIT failed")
        return _valid_result(110.0)

    result = run_cute_schedule_tuning(
        runner, _cublas(), {}, schedules=schedules
    )

    assert result["b_tune"] == 2
    assert result["trials"][0] == {
        "ordinal": 1,
        "schedule": DEFAULT_CUTE_SCHEDULE.as_dict(),
        "schedule_id": DEFAULT_CUTE_SCHEDULE.configuration_id,
        "status": "FAILED",
        "charged_to_b_tune": True,
        "b_tune": 1,
        "error_type": "RuntimeError",
        "error": "JIT failed",
    }
    assert result["best"]["median_us"] == 110.0
    assert result["comparison"]["best_speedup_vs_default"] is None


def test_static_rejection_is_retained_without_consuming_budget() -> None:
    invalid = CuteSchedule(96, 128, 1, 1)
    calls = []

    result = run_cute_schedule_tuning(
        lambda schedule: calls.append(schedule),
        _cublas(),
        {},
        schedules=(invalid,),
    )

    assert calls == []
    assert result["b_tune"] == 0
    assert result["status"] == "no_valid_schedule"
    assert result["trials"][0]["status"] == "STATICALLY_REJECTED"
    assert result["trials"][0]["charged_to_b_tune"] is False
