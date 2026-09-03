from __future__ import annotations

from typing import Mapping, Protocol

from .domain import BenchmarkResult, KernelProgram, WorkloadContract


class KernelBackend(Protocol):
    """Backend boundary; search code must depend on this protocol, not CUDA tools."""

    name: str

    def normalize_program(self, program: KernelProgram) -> str: ...

    def compile(
        self, program: KernelProgram, workload: WorkloadContract
    ) -> "CompilationResult": ...

    def check_correctness(
        self, artifact: "CompiledArtifact", workload: WorkloadContract
    ) -> "CorrectnessResult": ...

    def benchmark(
        self, artifact: "CompiledArtifact", workload: WorkloadContract
    ) -> BenchmarkResult: ...

    def lightweight_profile(
        self, artifact: "CompiledArtifact", workload: WorkloadContract
    ) -> Mapping[str, object]: ...

    def full_profile(
        self, artifact: "CompiledArtifact", workload: WorkloadContract
    ) -> Mapping[str, object]: ...

    def binary_fingerprint(
        self, artifact: "CompiledArtifact", launch_config: Mapping[str, object]
    ) -> str: ...


class CompiledArtifact(Protocol):
    artifact_id: str


class CompilationResult(Protocol):
    success: bool
    artifact: CompiledArtifact | None
    stdout: str
    stderr: str


class CorrectnessResult(Protocol):
    success: bool
    maximum_error: float | None
    mean_error: float | None

