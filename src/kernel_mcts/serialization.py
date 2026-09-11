from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from .domain import (
    BenchmarkResult,
    CompilationEvidence,
    CompileStatus,
    CorrectnessEvidence,
    CorrectnessStatus,
    EvaluationResult,
    InvalidReason,
    KernelProgram,
    ProposalStatus,
    ShapeCase,
    WorkloadContract,
)
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
        "prompt_text": generation.prompt_text,
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


def deserialize_workload(value: Mapping[str, Any]) -> WorkloadContract:
    return WorkloadContract(
        benchmark_id=str(value["benchmark_id"]),
        operation=str(value["operation"]),
        dtype=str(value["dtype"]),
        shapes=tuple(
            ShapeCase(
                {str(key): int(item) for key, item in shape["dimensions"].items()},
                float(shape["weight"]),
            )
            for shape in value["shapes"]
        ),
        rtol=float(value["rtol"]),
        atol=float(value["atol"]),
        metadata=value.get("metadata", {}),
    )


def deserialize_evaluation(value: Mapping[str, Any]) -> EvaluationResult:
    program_value = value.get("program")
    program = (
        KernelProgram(str(program_value["source"]), str(program_value["backend"]))
        if isinstance(program_value, Mapping)
        else None
    )
    benchmark_value = value.get("benchmark")
    benchmark = (
        BenchmarkResult(
            tuple(float(item) for item in benchmark_value["timings_us"]),
            float(benchmark_value["median_us"]),
            benchmark_value.get("per_shape_median_us", {}),
            benchmark_value.get("warmup_count"),
            benchmark_value.get("mean_us"),
            benchmark_value.get("stddev_us"),
            benchmark_value.get("min_us"),
            benchmark_value.get("max_us"),
        )
        if isinstance(benchmark_value, Mapping)
        else None
    )
    compilation_value = value.get("compilation")
    compilation = (
        CompilationEvidence(
            compilation_value.get("artifact_id"),
            str(compilation_value.get("stdout", "")),
            str(compilation_value.get("stderr", "")),
            compilation_value.get("duration_seconds"),
            tuple(compilation_value.get("artifact_paths", ())),
        )
        if isinstance(compilation_value, Mapping)
        else None
    )
    correctness_value = value.get("correctness")
    correctness = (
        CorrectnessEvidence(
            correctness_value.get("maximum_error"),
            correctness_value.get("mean_error"),
            correctness_value.get("failed_test_id"),
            correctness_value.get("reference_metadata", {}),
        )
        if isinstance(correctness_value, Mapping)
        else None
    )
    invalid_reason = value.get("invalid_reason")
    return EvaluationResult(
        status=ProposalStatus(value["status"]),
        program=program,
        state_key=value.get("state_key"),
        reward=value.get("reward"),
        benchmark=benchmark,
        invalid_reason=InvalidReason(invalid_reason) if invalid_reason else None,
        metadata=value.get("metadata", {}),
        compile_status=CompileStatus(value.get("compile_status", "NOT_ATTEMPTED")),
        correctness_status=CorrectnessStatus(
            value.get("correctness_status", "NOT_TESTED")
        ),
        worker_id=value.get("worker_id"),
        environment_manifest_id=value.get("environment_manifest_id"),
        source_hash=value.get("source_hash"),
        binary_hash=value.get("binary_hash"),
        launch_config=value.get("launch_config", {}),
        compilation=compilation,
        correctness=correctness,
    )


def deserialize_environment_manifest(value: Mapping[str, Any]) -> EnvironmentManifest:
    fields = dict(value)
    fields.pop("manifest_id", None)
    return EnvironmentManifest(**fields)


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
