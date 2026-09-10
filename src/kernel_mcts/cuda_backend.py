from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from .domain import BenchmarkResult, KernelProgram, WorkloadContract
from .evaluation import (
    CandidateBenchmarkError,
    CandidateLaunchError,
    CandidateTimeoutError,
    EvaluationInfrastructureError,
)


class CommandRunner(Protocol):
    def __call__(
        self,
        command: Sequence[str],
        *,
        timeout: float,
        env: Mapping[str, str],
    ) -> subprocess.CompletedProcess[str]: ...


@dataclass(frozen=True, slots=True)
class CudaBackendConfig:
    artifact_root: Path
    nvcc: str = "nvcc"
    cuobjdump: str | None = "cuobjdump"
    architecture: str = "sm_90"
    compile_timeout_seconds: float = 120.0
    execution_timeout_seconds: float = 120.0
    warmup_count: int = 10
    measurement_count: int = 30
    seed: int = 0
    subprocess_environment: Mapping[str, str] = field(
        default_factory=lambda: {"PATH": os.defpath, "LANG": "C", "LC_ALL": "C"}
    )

    def __post_init__(self) -> None:
        if self.compile_timeout_seconds <= 0 or self.execution_timeout_seconds <= 0:
            raise ValueError("CUDA backend timeouts must be positive")
        if self.warmup_count < 0 or self.measurement_count < 1:
            raise ValueError("CUDA benchmark counts are invalid")


@dataclass(frozen=True, slots=True)
class CudaArtifact:
    artifact_id: str
    executable: Path


@dataclass(frozen=True, slots=True)
class CudaCompilationResult:
    success: bool
    artifact: CudaArtifact | None
    stdout: str
    stderr: str
    duration_seconds: float | None
    timed_out: bool = False

    @property
    def artifact_paths(self) -> tuple[str, ...]:
        return (str(self.artifact.executable),) if self.artifact else ()


@dataclass(frozen=True, slots=True)
class CudaCorrectnessResult:
    success: bool
    maximum_error: float | None
    mean_error: float | None
    failed_test_id: str | None = None
    reference_metadata: Mapping[str, object] = field(default_factory=dict)


class CudaCppBackend:
    name = "cuda_cpp"

    def __init__(
        self,
        config: CudaBackendConfig,
        *,
        runner: CommandRunner | None = None,
    ) -> None:
        self.config = config
        self._runner = runner or _run_command

    def normalize_program(self, program: KernelProgram) -> str:
        without_comments = re.sub(r"//[^\n]*|/\*.*?\*/", " ", program.source, flags=re.S)
        return " ".join(without_comments.split())

    def compile(
        self,
        program: KernelProgram,
        workload: WorkloadContract,
    ) -> CudaCompilationResult:
        self._validate_contract(program, workload)
        self.config.artifact_root.mkdir(parents=True, exist_ok=True)
        artifact_dir = Path(
            tempfile.mkdtemp(prefix="cuda-artifact-", dir=self.config.artifact_root)
        )
        candidate_path = artifact_dir / "candidate.cu"
        harness_path = artifact_dir / "harness.cu"
        executable = artifact_dir / "evaluate"
        candidate_path.write_text(program.source, encoding="utf-8")
        harness_path.write_text(
            files("kernel_mcts.benchmarks")
            .joinpath("kernels", "bf16_gemm_harness.cu")
            .read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        command = [
            _resolve_tool(self.config.nvcc),
            f"-arch={self.config.architecture}",
            "-std=c++17",
            "-O3",
            "-rdc=true",
            str(candidate_path),
            str(harness_path),
            "-lcublas",
            "-o",
            str(executable),
        ]
        started = time.monotonic()
        try:
            completed = self._runner(
                command,
                timeout=self.config.compile_timeout_seconds,
                env=self.config.subprocess_environment,
            )
        except subprocess.TimeoutExpired as error:
            return CudaCompilationResult(
                False,
                None,
                _bounded(error.stdout or ""),
                _bounded(error.stderr or ""),
                time.monotonic() - started,
                True,
            )
        except OSError as error:
            raise EvaluationInfrastructureError("CUDA compiler could not be executed") from error
        duration = time.monotonic() - started
        if completed.returncode != 0:
            return CudaCompilationResult(
                False,
                None,
                _bounded(completed.stdout),
                _bounded(completed.stderr),
                duration,
            )
        artifact_id = hashlib.sha256(executable.read_bytes()).hexdigest()
        return CudaCompilationResult(
            True,
            CudaArtifact(artifact_id, executable),
            _bounded(completed.stdout),
            _bounded(completed.stderr),
            duration,
        )

    def check_correctness(
        self,
        artifact: CudaArtifact,
        workload: WorkloadContract,
    ) -> CudaCorrectnessResult:
        payload = self._execute(artifact, workload, "correctness")
        return CudaCorrectnessResult(
            success=bool(payload["success"]),
            maximum_error=_optional_float(payload.get("maximum_error")),
            mean_error=_optional_float(payload.get("mean_error")),
            failed_test_id=payload.get("failed_test_id"),
            reference_metadata=payload.get("reference_metadata", {}),
        )

    def benchmark(
        self,
        artifact: CudaArtifact,
        workload: WorkloadContract,
    ) -> BenchmarkResult:
        payload = self._execute(artifact, workload, "benchmark")
        timings = tuple(float(value) for value in payload["timings_us"])
        if len(timings) != self.config.measurement_count:
            raise EvaluationInfrastructureError("worker returned an unexpected timing count")
        return BenchmarkResult(
            timings_us=timings,
            median_us=statistics.median(timings),
            warmup_count=self.config.warmup_count,
            mean_us=statistics.fmean(timings),
            stddev_us=statistics.pstdev(timings),
            min_us=min(timings),
            max_us=max(timings),
        )

    def binary_fingerprint(
        self,
        artifact: CudaArtifact,
        launch_config: Mapping[str, object],
    ) -> str:
        if self.config.cuobjdump:
            try:
                result = self._runner(
                    [
                        _resolve_tool(self.config.cuobjdump),
                        "--dump-sass",
                        str(artifact.executable),
                    ],
                    timeout=self.config.compile_timeout_seconds,
                    env=self.config.subprocess_environment,
                )
                if result.returncode == 0 and result.stdout.strip():
                    normalized = re.sub(r"/\*[^*]*\*/", "", result.stdout)
                    normalized = " ".join(normalized.split())
                    return hashlib.sha256(normalized.encode()).hexdigest()
            except (OSError, subprocess.TimeoutExpired):
                pass
        return hashlib.sha256(artifact.executable.read_bytes()).hexdigest()

    def lightweight_profile(self, artifact, workload):
        raise NotImplementedError("CUDA profiling is a later phase")

    def full_profile(self, artifact, workload):
        raise NotImplementedError("CUDA profiling is a later phase")

    def _execute(
        self,
        artifact: CudaArtifact,
        workload: WorkloadContract,
        mode: str,
    ) -> dict[str, object]:
        shape = workload.shapes[0].dimensions
        command = [
            str(artifact.executable),
            f"--mode={mode}",
            f"--M={shape['M']}",
            f"--N={shape['N']}",
            f"--K={shape['K']}",
            f"--rtol={workload.rtol}",
            f"--atol={workload.atol}",
            f"--seed={self.config.seed}",
            f"--warmups={self.config.warmup_count}",
            f"--measurements={self.config.measurement_count}",
        ]
        try:
            result = self._runner(
                command,
                timeout=self.config.execution_timeout_seconds,
                env=self.config.subprocess_environment,
            )
        except subprocess.TimeoutExpired as error:
            raise CandidateTimeoutError("candidate execution timed out") from error
        except OSError as error:
            raise EvaluationInfrastructureError("evaluation harness could not be executed") from error
        payload = _parse_payload(result.stdout)
        status = payload.get("status")
        if status == "launch_failure":
            raise CandidateLaunchError("candidate launch failed")
        if status == "benchmark_failure":
            raise CandidateBenchmarkError("candidate benchmark failed")
        if result.returncode != 0 or status != "ok":
            raise EvaluationInfrastructureError("evaluation harness failed")
        return payload

    @staticmethod
    def _validate_contract(program: KernelProgram, workload: WorkloadContract) -> None:
        if program.backend != "cuda_cpp":
            raise ValueError(f"unsupported program backend {program.backend!r}")
        if workload.operation != "gemm" or workload.dtype != "bfloat16":
            raise ValueError("CudaCppBackend currently supports only BF16 GEMM")
        if len(workload.shapes) != 1:
            raise ValueError("CudaCppBackend currently requires one fixed GEMM shape")
        if workload.metadata.get("entry_point") != "bf16_gemm_root":
            raise ValueError("BF16 GEMM workload must use the bf16_gemm_root entry point")


def _run_command(command, *, timeout, env):
    return subprocess.run(
        command,
        timeout=timeout,
        env=dict(env),
        text=True,
        capture_output=True,
        check=False,
    )


def _parse_payload(stdout: str) -> dict[str, object]:
    try:
        value = json.loads(stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as error:
        raise EvaluationInfrastructureError("worker returned malformed JSON") from error
    if not isinstance(value, dict):
        raise EvaluationInfrastructureError("worker JSON payload must be an object")
    return value


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)


def _bounded(value: str, limit: int = 64_000) -> str:
    return value[-limit:]


def _resolve_tool(command: str) -> str:
    if os.sep in command:
        return command
    return shutil.which(command) or command
