from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from .domain import BenchmarkResult, EvaluationResult, WorkloadContract
from .generation import GenerationResult
from .providers import EnvironmentManifest


def serialize_benchmark(benchmark: BenchmarkResult | None) -> dict[str, Any] | None:
    if benchmark is None:
        return None
    return {
        "timings_us": list(benchmark.timings_us),
        "median_us": benchmark.median_us,
        "per_shape_median_us": _json_value(benchmark.per_shape_median_us),
        "warmup_count": benchmark.warmup_count,
        "mean_us": benchmark.mean_us,
        "stddev_us": benchmark.stddev_us,
        "min_us": benchmark.min_us,
        "max_us": benchmark.max_us,
    }


def serialize_evaluation(evaluation: EvaluationResult) -> dict[str, Any]:
    return {
        "status": evaluation.status.value,
        "program": serialize_program(evaluation.program),
        "state_key": evaluation.state_key,
        "reward": evaluation.reward,
        "benchmark": serialize_benchmark(evaluation.benchmark),
        "invalid_reason": evaluation.invalid_reason.value if evaluation.invalid_reason else None,
        "compile_status": evaluation.compile_status.value,
        "correctness_status": evaluation.correctness_status.value,
        "worker_id": evaluation.worker_id,
        "environment_manifest_id": evaluation.environment_manifest_id,
        "source_hash": evaluation.source_hash,
        "binary_hash": evaluation.binary_hash,
        "launch_config": _json_value(evaluation.launch_config),
        "compilation": (
            {
                "artifact_id": evaluation.compilation.artifact_id,
                "stdout": evaluation.compilation.stdout,
                "stderr": evaluation.compilation.stderr,
                "duration_seconds": evaluation.compilation.duration_seconds,
                "artifact_paths": list(evaluation.compilation.artifact_paths),
            }
            if evaluation.compilation
            else None
        ),
        "correctness": (
            {
                "maximum_error": evaluation.correctness.maximum_error,
                "mean_error": evaluation.correctness.mean_error,
                "failed_test_id": evaluation.correctness.failed_test_id,
                "reference_metadata": _json_value(
                    evaluation.correctness.reference_metadata
                ),
            }
            if evaluation.correctness
            else None
        ),
        "metadata": _json_value(evaluation.metadata),
    }


def serialize_generation(generation: GenerationResult) -> dict[str, Any]:
    return {
        "generation_id": generation.generation_id,
        "raw_output": generation.raw_output,
        "program": serialize_program(generation.program),
        "prompt_hash": generation.prompt_hash,
        "input_tokens": generation.input_tokens,
        "output_tokens": generation.output_tokens,
        "latency_seconds": generation.latency_seconds,
        "metadata": _json_value(generation.metadata or {}),
    }


def serialize_program(program) -> dict[str, str] | None:
    if program is None:
        return None
    return {"source": program.source, "backend": program.backend}


def serialize_profile(profile: Mapping[str, object] | None) -> dict[str, Any] | None:
    if profile is None:
        return None
    value = _json_value(profile)
    assert isinstance(value, dict)
    return value


def serialize_workload(workload: WorkloadContract) -> dict[str, Any]:
    return {
        "benchmark_id": workload.benchmark_id,
        "operation": workload.operation,
        "dtype": workload.dtype,
        "shapes": [
            {"dimensions": _json_value(shape.dimensions), "weight": shape.weight}
            for shape in workload.shapes
        ],
        "rtol": workload.rtol,
        "atol": workload.atol,
        "metadata": _json_value(workload.metadata),
    }


def serialize_environment_manifest(manifest: EnvironmentManifest) -> dict[str, Any]:
    value = _json_value(manifest.as_dict())
    assert isinstance(value, dict)
    return value


def _json_value(value: object) -> Any:
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")
