from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Mapping

from .backends import KernelBackend
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
    WorkloadContract,
)
from .serialization import serialize_workload


class EvaluationInfrastructureError(RuntimeError):
    """A worker or tool failure unrelated to candidate validity."""


class CandidateBenchmarkError(RuntimeError):
    """A deterministic candidate launch or benchmark failure."""


class CandidateLaunchError(RuntimeError):
    """The compiled candidate could not launch successfully."""


class CandidateTimeoutError(RuntimeError):
    """A candidate exceeded a bounded compile or execution timeout."""


@dataclass(frozen=True, slots=True)
class EvaluationContext:
    worker_id: str
    environment_manifest_id: str
    launch_config: Mapping[str, object]
    hardware_toolchain: Mapping[str, object]


class BackendKernelEvaluator:
    """Run mandatory compile, correctness, and benchmark evaluation once."""

    def __init__(
        self,
        *,
        backend: KernelBackend,
        root_benchmark: BenchmarkResult,
        context: EvaluationContext,
    ) -> None:
        self.backend = backend
        self.root_benchmark = root_benchmark
        self.context = context

    def evaluate(
        self,
        program: KernelProgram,
        workload: WorkloadContract,
    ) -> EvaluationResult:
        source_hash = _sha256(self.backend.normalize_program(program))
        compile_status = CompileStatus.NOT_ATTEMPTED
        correctness_status = CorrectnessStatus.NOT_TESTED
        compilation: CompilationEvidence | None = None
        correctness: CorrectnessEvidence | None = None
        artifact = None

        try:
            compiled = self.backend.compile(program, workload)
            compilation = CompilationEvidence(
                artifact_id=(compiled.artifact.artifact_id if compiled.artifact else None),
                stdout=compiled.stdout,
                stderr=compiled.stderr,
                duration_seconds=getattr(compiled, "duration_seconds", None),
                artifact_paths=getattr(compiled, "artifact_paths", ()),
            )
            if not compiled.success or compiled.artifact is None:
                return self._result(
                    status=ProposalStatus.INVALID,
                    program=program,
                    source_hash=source_hash,
                    invalid_reason=(
                        InvalidReason.TIMEOUT
                        if getattr(compiled, "timed_out", False)
                        else InvalidReason.COMPILE_FAILURE
                    ),
                    compile_status=CompileStatus.FAIL,
                    correctness_status=correctness_status,
                    compilation=compilation,
                )

            artifact = compiled.artifact
            compile_status = CompileStatus.SUCCESS
            checked = self.backend.check_correctness(artifact, workload)
            correctness = CorrectnessEvidence(
                maximum_error=checked.maximum_error,
                mean_error=checked.mean_error,
                failed_test_id=getattr(checked, "failed_test_id", None),
                reference_metadata=getattr(checked, "reference_metadata", {}),
            )
            if not checked.success:
                return self._result(
                    status=ProposalStatus.INVALID,
                    program=program,
                    source_hash=source_hash,
                    invalid_reason=InvalidReason.CORRECTNESS_FAILURE,
                    compile_status=compile_status,
                    correctness_status=CorrectnessStatus.FAIL,
                    artifact=artifact,
                    compilation=compilation,
                    correctness=correctness,
                )

            correctness_status = CorrectnessStatus.PASS
            benchmark = self.backend.benchmark(artifact, workload)
            _validate_benchmark(benchmark, workload)
            reward = root_normalized_reward(self.root_benchmark, benchmark, workload)
            binary_hash = self.backend.binary_fingerprint(
                artifact,
                self.context.launch_config,
            )
            state_key = _state_key(
                binary_hash=binary_hash,
                workload=workload,
                context=self.context,
            )
            return self._result(
                status=ProposalStatus.VALID,
                program=program,
                source_hash=source_hash,
                state_key=state_key,
                reward=reward,
                benchmark=benchmark,
                binary_hash=binary_hash,
                compile_status=compile_status,
                correctness_status=correctness_status,
                artifact=artifact,
                compilation=compilation,
                correctness=correctness,
            )
        except (CandidateBenchmarkError, CandidateLaunchError, CandidateTimeoutError) as error:
            if isinstance(error, CandidateLaunchError):
                reason = InvalidReason.LAUNCH_FAILURE
            elif isinstance(error, CandidateTimeoutError):
                reason = InvalidReason.TIMEOUT
            else:
                reason = InvalidReason.BENCHMARK_FAILURE
            return self._result(
                status=ProposalStatus.INVALID,
                program=program,
                source_hash=source_hash,
                invalid_reason=reason,
                compile_status=compile_status,
                correctness_status=correctness_status,
                artifact=artifact,
                compilation=compilation,
                correctness=correctness,
                metadata={"error_type": type(error).__name__},
            )
        except EvaluationInfrastructureError as error:
            return self._result(
                status=ProposalStatus.INFRASTRUCTURE_FAILURE,
                program=program,
                source_hash=source_hash,
                compile_status=compile_status,
                correctness_status=correctness_status,
                artifact=artifact,
                compilation=compilation,
                correctness=correctness,
                metadata={"error_type": type(error).__name__},
            )

    def _result(self, **values: object) -> EvaluationResult:
        artifact = values.pop("artifact", None)
        return EvaluationResult(
            worker_id=self.context.worker_id,
            environment_manifest_id=self.context.environment_manifest_id,
            launch_config=self.context.launch_config,
            compiled_artifact=artifact,
            **values,
        )


def _validate_benchmark(
    benchmark: BenchmarkResult,
    workload: WorkloadContract,
) -> None:
    if not benchmark.timings_us or any(
        not math.isfinite(value) or value <= 0 for value in benchmark.timings_us
    ):
        raise CandidateBenchmarkError("benchmark timings must be finite and positive")
    if not math.isfinite(benchmark.median_us) or benchmark.median_us <= 0:
        raise CandidateBenchmarkError("benchmark median must be finite and positive")
    if len(workload.shapes) > 1:
        for shape in workload.shapes:
            key = _shape_key(shape.dimensions)
            value = benchmark.per_shape_median_us.get(key)
            if value is None or not math.isfinite(value) or value <= 0:
                raise CandidateBenchmarkError(f"missing valid timing for shape {key}")


def root_normalized_reward(
    root: BenchmarkResult,
    candidate: BenchmarkResult,
    workload: WorkloadContract,
) -> float:
    """Return the spec-defined log speedup against one fixed root benchmark."""
    if len(workload.shapes) == 1:
        if root.median_us <= 0 or not math.isfinite(root.median_us):
            raise ValueError("root benchmark median must be finite and positive")
        return math.log(root.median_us / candidate.median_us)

    reward = 0.0
    for shape in workload.shapes:
        key = _shape_key(shape.dimensions)
        root_time = root.per_shape_median_us.get(key)
        candidate_time = candidate.per_shape_median_us[key]
        if root_time is None or root_time <= 0 or not math.isfinite(root_time):
            raise ValueError(f"missing valid root timing for shape {key}")
        reward += shape.weight * math.log(root_time / candidate_time)
    return reward


def _shape_key(dimensions: Mapping[str, int]) -> str:
    return ",".join(f"{name}={value}" for name, value in sorted(dimensions.items()))


def _state_key(
    *,
    binary_hash: str,
    workload: WorkloadContract,
    context: EvaluationContext,
) -> str:
    identity = {
        "binary_hash": binary_hash,
        "workload": serialize_workload(workload),
        "launch_config": context.launch_config,
        "hardware_toolchain": context.hardware_toolchain,
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()
