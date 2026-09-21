from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from .cute_schedule import (
    DEFAULT_CUTE_SCHEDULE,
    CuteSchedule,
    enumerate_cute_schedules,
    validate_cute_schedule,
)
from .domain import BenchmarkResult
from .cute_program import (
    REFERENCE_CUTE_GEMM,
    CuteGemmProgram,
    validate_cute_gemm_program,
)
from .serialization import serialize_benchmark


ScheduleRunner = Callable[[CuteSchedule], Mapping[str, Any]]
ProgramRunner = Callable[[CuteGemmProgram], Mapping[str, Any]]


def pipeline_interaction_candidates() -> tuple[CuteGemmProgram, ...]:
    """Return the fixed nine-point tile/cluster/mainloop-stage interaction grid."""

    return tuple(
        CuteGemmProgram(
            128,
            256,
            cluster_m,
            cluster_n,
            pipeline_stages=pipeline_stages,
        )
        for cluster_m, cluster_n in ((1, 1), (1, 2), (2, 1))
        for pipeline_stages in (None, 2, 3)
    )


def run_cute_pipeline_interaction_tuning(
    runner: ProgramRunner,
    cublas: Mapping[str, object],
    environment_manifest: Mapping[str, Any],
    *,
    candidates: Sequence[CuteGemmProgram] | None = None,
) -> Mapping[str, Any]:
    """Evaluate the bounded cluster-by-pipeline interaction grid under B_tune."""

    programs = (
        tuple(candidates)
        if candidates is not None
        else pipeline_interaction_candidates()
    )
    trials: list[dict[str, Any]] = []
    attempted = 0
    best_trial: dict[str, Any] | None = None
    default_trial: dict[str, Any] | None = None
    for ordinal, program in enumerate(programs, 1):
        legality = validate_cute_gemm_program(program)
        base = {
            "ordinal": ordinal,
            "representation": program.as_dict(),
            "configuration_hash": program.configuration_hash,
        }
        if not legality.valid:
            trials.append(
                {
                    **base,
                    "status": "STATICALLY_REJECTED",
                    "charged_to_b_tune": False,
                    "b_tune": attempted,
                    "static_validation": legality.as_dict(),
                }
            )
            continue

        attempted += 1
        try:
            result = dict(runner(program))
            _require_valid_result(result)
        except Exception as error:
            trial = {
                **base,
                "status": "FAILED",
                "charged_to_b_tune": True,
                "b_tune": attempted,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        else:
            trial = {
                **base,
                "status": "VALID",
                "charged_to_b_tune": True,
                "b_tune": attempted,
                "result": result,
            }
            if best_trial is None or _median_us(trial) < _median_us(best_trial):
                best_trial = trial
        trials.append(trial)
        if program == REFERENCE_CUTE_GEMM:
            default_trial = trial

    cublas_result = _serialize_cublas(cublas)
    cublas_median = float(cublas_result["benchmark"]["median_us"])
    best_median = _median_us(best_trial) if best_trial is not None else None
    default_median = (
        _median_us(default_trial)
        if default_trial is not None and default_trial["status"] == "VALID"
        else None
    )
    return {
        "status": "ok" if best_trial is not None else "no_valid_configuration",
        "benchmark_id": "bf16_gemm_4096_h100",
        "method": "deterministic_pipeline_interaction_grid",
        "b_tune": attempted,
        "candidate_count": len(programs),
        "environment_manifest": dict(environment_manifest),
        "cublas": cublas_result,
        "trials": trials,
        "best": _program_trial_summary(best_trial),
        "default": _program_trial_summary(default_trial),
        "comparison": {
            "best_latency_ratio_vs_cublas": (
                best_median / cublas_median if best_median is not None else None
            ),
            "best_speedup_vs_default": (
                default_median / best_median
                if default_median is not None and best_median is not None
                else None
            ),
        },
    }


def run_cute_schedule_tuning(
    runner: ScheduleRunner,
    cublas: Mapping[str, object],
    environment_manifest: Mapping[str, Any],
    *,
    schedules: Sequence[CuteSchedule] | None = None,
) -> Mapping[str, Any]:
    """Evaluate a deterministic CuTe schedule grid under a separate B_tune."""

    candidates = tuple(schedules) if schedules is not None else enumerate_cute_schedules()
    trials: list[dict[str, Any]] = []
    attempted = 0
    best_trial: dict[str, Any] | None = None
    default_trial: dict[str, Any] | None = None

    for ordinal, schedule in enumerate(candidates, 1):
        reasons = validate_cute_schedule(schedule)
        base = {
            "ordinal": ordinal,
            "schedule": schedule.as_dict(),
            "schedule_id": schedule.configuration_id,
        }
        if reasons:
            trials.append(
                {
                    **base,
                    "status": "STATICALLY_REJECTED",
                    "charged_to_b_tune": False,
                    "b_tune": attempted,
                    "reasons": list(reasons),
                }
            )
            continue

        attempted += 1
        try:
            result = dict(runner(schedule))
            _require_valid_result(result)
        except Exception as error:  # A failed JIT/evaluation is a retained tuning trial.
            trial = {
                **base,
                "status": "FAILED",
                "charged_to_b_tune": True,
                "b_tune": attempted,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        else:
            trial = {
                **base,
                "status": "VALID",
                "charged_to_b_tune": True,
                "b_tune": attempted,
                "result": result,
            }
            if best_trial is None or _median_us(trial) < _median_us(best_trial):
                best_trial = trial
        trials.append(trial)
        if schedule == DEFAULT_CUTE_SCHEDULE:
            default_trial = trial

    cublas_result = _serialize_cublas(cublas)
    cublas_median = float(cublas_result["benchmark"]["median_us"])
    best_median = _median_us(best_trial) if best_trial is not None else None
    default_median = (
        _median_us(default_trial)
        if default_trial is not None and default_trial["status"] == "VALID"
        else None
    )
    return {
        "status": "ok" if best_trial is not None else "no_valid_schedule",
        "benchmark_id": "bf16_gemm_4096_h100",
        "method": "deterministic_grid",
        "b_tune": attempted,
        "candidate_count": len(candidates),
        "environment_manifest": dict(environment_manifest),
        "cublas": cublas_result,
        "trials": trials,
        "best": _trial_summary(best_trial),
        "default": _trial_summary(default_trial),
        "comparison": {
            "best_latency_ratio_vs_cublas": (
                best_median / cublas_median if best_median is not None else None
            ),
            "best_speedup_vs_default": (
                default_median / best_median
                if default_median is not None and best_median is not None
                else None
            ),
        },
    }


def _require_valid_result(result: Mapping[str, Any]) -> None:
    if not result.get("comparable_to_repository_baselines"):
        raise RuntimeError("CuTe schedule did not complete the comparable contract")
    correctness = result.get("correctness")
    if not isinstance(correctness, Mapping) or not correctness.get("success"):
        raise RuntimeError("CuTe schedule failed correctness")
    benchmark = result.get("benchmark")
    if not isinstance(benchmark, Mapping) or "median_us" not in benchmark:
        raise RuntimeError("CuTe schedule returned no median latency")


def _median_us(trial: Mapping[str, Any]) -> float:
    result = trial["result"]
    assert isinstance(result, Mapping)
    benchmark = result["benchmark"]
    assert isinstance(benchmark, Mapping)
    return float(benchmark["median_us"])


def _trial_summary(trial: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if trial is None:
        return None
    summary = {
        "status": trial["status"],
        "schedule": trial["schedule"],
        "schedule_id": trial["schedule_id"],
        "b_tune": trial["b_tune"],
    }
    if trial["status"] == "VALID":
        summary["median_us"] = _median_us(trial)
    return summary


def _program_trial_summary(
    trial: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if trial is None:
        return None
    summary = {
        "status": trial["status"],
        "representation": trial["representation"],
        "configuration_hash": trial["configuration_hash"],
        "b_tune": trial["b_tune"],
    }
    if trial["status"] == "VALID":
        summary["median_us"] = _median_us(trial)
    return summary


def _serialize_cublas(result: Mapping[str, object]) -> dict[str, object]:
    correctness = result.get("correctness")
    benchmark = result.get("benchmark")
    if not isinstance(correctness, Mapping) or not correctness.get("success"):
        raise RuntimeError("cuBLAS baseline failed correctness")
    if not isinstance(benchmark, BenchmarkResult):
        raise TypeError("cuBLAS baseline returned an invalid benchmark")
    return {
        "correctness": dict(correctness),
        "benchmark": serialize_benchmark(benchmark),
    }
