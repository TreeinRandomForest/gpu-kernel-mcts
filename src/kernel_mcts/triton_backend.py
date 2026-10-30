"""H100 Triton backend for the fixed BF16 GEMM integration experiment."""

from __future__ import annotations

import hashlib
import os
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from .compiler_context import MAX_PTX_ARTIFACT_BYTES
from .cuda_backend import _ncu_csv_output, _ncu_diagnostic, _parse_ncu_csv, _parse_payload, _resolve_tool
from .domain import BenchmarkResult, KernelProgram, WorkloadContract
from .evaluation import CandidateBenchmarkError, CandidateLaunchError, CandidateTimeoutError, EvaluationInfrastructureError
from .profiling import normalize_ncu_version, resolve_profile_metric_set


@dataclass(frozen=True, slots=True)
class TritonBackendConfig:
    artifact_root: Path
    python: str = sys.executable
    ncu: str = "ncu"
    architecture: str = "sm_90"
    ncu_version: str | None = None
    compile_timeout_seconds: float = 180.0
    execution_timeout_seconds: float = 180.0
    profile_timeout_seconds: float = 300.0
    warmup_count: int = 10
    measurement_count: int = 30
    seed: int = 0
    subprocess_environment: Mapping[str, str] = field(
        default_factory=lambda: {
            "PATH": f"/usr/local/cuda/bin:{os.defpath}", "LANG": "C", "LC_ALL": "C",
            **{name: os.environ[name] for name in (
                "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES",
                "TRITON_CACHE_DIR",
            ) if name in os.environ},
        }
    )

    def __post_init__(self) -> None:
        if min(self.compile_timeout_seconds, self.execution_timeout_seconds,
               self.profile_timeout_seconds) <= 0:
            raise ValueError("Triton backend timeouts must be positive")
        if self.warmup_count < 0 or self.measurement_count < 1:
            raise ValueError("Triton benchmark counts are invalid")


@dataclass(frozen=True, slots=True)
class TritonArtifact:
    artifact_id: str
    source_path: Path
    cubin_path: Path
    ptx_path: Path | None


@dataclass(frozen=True, slots=True)
class TritonCompilationResult:
    success: bool
    artifact: TritonArtifact | None
    stdout: str
    stderr: str
    duration_seconds: float | None
    timed_out: bool = False
    compiler_context: Mapping[str, object] | None = None

    @property
    def artifact_paths(self) -> tuple[str, ...]:
        if self.artifact is None:
            return ()
        paths = [self.artifact.source_path, self.artifact.cubin_path]
        if self.artifact.ptx_path is not None:
            paths.append(self.artifact.ptx_path)
        return tuple(map(str, paths))


@dataclass(frozen=True, slots=True)
class TritonCorrectnessResult:
    success: bool
    maximum_error: float | None
    mean_error: float | None
    failed_test_id: str | None
    reference_metadata: Mapping[str, object]


class TritonH100Backend:
    name = "triton_h100"

    def __init__(self, config: TritonBackendConfig, *, runner=None) -> None:
        self.config = config
        self._runner = runner or _run_command
        self._kernel_names: dict[str, str] = {}
        self._compiler_resources: dict[str, dict[str, object]] = {}

    def normalize_program(self, program: KernelProgram) -> str:
        # Python indentation and line breaks carry semantics.
        return program.source.replace("\r\n", "\n").strip()

    def compile(self, program: KernelProgram, workload: WorkloadContract) -> TritonCompilationResult:
        self._validate(program, workload)
        self.config.artifact_root.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix="triton-artifact-", dir=self.config.artifact_root))
        source = directory / "candidate.py"
        source.write_text(program.source, encoding="utf-8")
        started = time.monotonic()
        try:
            result = self._runner(self._command(source, workload, "compile"),
                                  timeout=self.config.compile_timeout_seconds,
                                  env=self.config.subprocess_environment)
        except subprocess.TimeoutExpired:
            return TritonCompilationResult(False, None, "", "compile timed out",
                                           time.monotonic() - started, True)
        except OSError as error:
            raise EvaluationInfrastructureError("Triton harness could not be executed") from error
        duration = time.monotonic() - started
        if result.returncode != 0:
            return TritonCompilationResult(False, None, result.stdout[-2000:],
                                           result.stderr[-2000:], duration)
        payload = _parse_payload(result.stdout)
        if payload.get("status") != "ok":
            return TritonCompilationResult(False, None, result.stdout[-2000:],
                                           result.stderr[-2000:], duration)
        cubin = directory / "kernel.cubin"
        if not cubin.is_file():
            raise EvaluationInfrastructureError("Triton harness did not save cubin")
        artifact_id = hashlib.sha256(cubin.read_bytes()).hexdigest()
        ptx = directory / "kernel.ptx"
        artifact = TritonArtifact(artifact_id, source, cubin, ptx if ptx.is_file() else None)
        ptx_artifacts = []
        if artifact.ptx_path is not None:
            ptx_bytes = artifact.ptx_path.read_bytes()
            ptx_artifacts.append({
                "path": str(artifact.ptx_path),
                "sha256": hashlib.sha256(ptx_bytes).hexdigest(),
                "size_bytes": len(ptx_bytes),
                "storage_status": (
                    "eligible" if len(ptx_bytes) <= MAX_PTX_ARTIFACT_BYTES else "too_large"
                ),
            })
        name = payload.get("kernel_name")
        if isinstance(name, str):
            self._kernel_names[artifact_id] = name
        context: dict[str, object] = {
            "schema_version": 1,
            "backend": self.name,
            "architecture": self.config.architecture,
            "resources": {
                "registers_per_thread": payload.get("registers_per_thread"),
                "static_shared_memory_bytes": payload.get("shared_memory_bytes"),
                "spills": payload.get("spills"),
            },
            "ptx": {"status": "captured" if ptx_artifacts else "unavailable",
                    "artifacts": ptx_artifacts},
            "sass": {"status": "unavailable"},
        }
        self._compiler_resources[artifact_id] = dict(context["resources"])
        return TritonCompilationResult(True, artifact, result.stdout[-2000:],
                                       result.stderr[-2000:], duration, compiler_context=context)

    def check_correctness(self, artifact: TritonArtifact,
                          workload: WorkloadContract) -> TritonCorrectnessResult:
        payload = self._execute(artifact, workload, "correctness")
        return TritonCorrectnessResult(
            bool(payload["success"]), payload.get("maximum_error"), payload.get("mean_error"),
            payload.get("failed_test_id"), payload.get("reference_metadata", {}),
        )

    def benchmark(self, artifact: TritonArtifact, workload: WorkloadContract) -> BenchmarkResult:
        payload = self._execute(artifact, workload, "benchmark")
        timings = tuple(float(value) for value in payload["timings_us"])
        if len(timings) != self.config.measurement_count or any(value <= 0 for value in timings):
            raise EvaluationInfrastructureError("Triton harness returned invalid timings")
        return BenchmarkResult(
            timings_us=timings, median_us=statistics.median(timings),
            warmup_count=self.config.warmup_count, mean_us=statistics.fmean(timings),
            stddev_us=statistics.pstdev(timings), min_us=min(timings), max_us=max(timings),
        )

    def binary_fingerprint(self, artifact: TritonArtifact,
                           launch_config: Mapping[str, object]) -> str:
        # launch() may change the grid without changing device code; retain
        # source identity until launcher parameters are extracted separately.
        digest = hashlib.sha256()
        digest.update(artifact.cubin_path.read_bytes())
        digest.update(artifact.source_path.read_bytes())
        return digest.hexdigest()

    def lightweight_profile(self, artifact: TritonArtifact,
                            workload: WorkloadContract,
                            metric_set: str = "lightweight_v1") -> Mapping[str, object]:
        if metric_set not in {"lightweight_v1", "diagnostic_v2"}:
            raise EvaluationInfrastructureError(
                "Triton H100 supports lightweight_v1 and diagnostic_v2 NCU metrics"
            )
        try:
            definition = resolve_profile_metric_set(metric_set, self.config.architecture,
                                                    self.config.ncu_version)
        except ValueError as error:
            raise EvaluationInfrastructureError(str(error)) from error
        command = [
            _resolve_tool(self.config.ncu), "--csv", "--page", "raw", "--log-file", "stdout",
            "--kernel-name-base", "function", "--kernel-name", "regex:.*bf16_gemm_triton.*",
            "--launch-count", "1", "--metrics", ",".join(definition.metrics),
            *self._command(artifact.source_path, workload, "profile"),
        ]
        try:
            result = self._runner(command, timeout=self.config.profile_timeout_seconds,
                                  env=self.config.subprocess_environment)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise EvaluationInfrastructureError("Triton NCU profiling could not be completed") from error
        if result.returncode != 0:
            diagnostic = _ncu_diagnostic(result.stdout, result.stderr)
            raise EvaluationInfrastructureError(f"Triton NCU profiling failed: {diagnostic}")
        metrics = _parse_ncu_csv(
            _ncu_csv_output(result.stdout, result.stderr, definition.metrics), definition.metrics
        )
        summary = {alias: metrics[name]["value"] for name, alias in definition.aliases.items()}
        if metric_set == "diagnostic_v2":
            resources = self._compiler_resources.get(artifact.artifact_id)
            if resources is not None:
                summary["compiler_spills"] = resources.get("spills")
                summary["compiler_static_shared_memory_bytes"] = resources.get(
                    "static_shared_memory_bytes"
                )
        return {
            "schema_version": definition.schema_version, "profiler": "ncu",
            "metric_set": definition.id,
            "target": {"architecture": self.config.architecture,
                       "ncu_version": normalize_ncu_version(self.config.ncu_version)},
            "resolved_metrics": list(definition.metrics), "artifact_id": artifact.artifact_id,
            "summary": summary,
            "metrics": metrics,
        }

    def full_profile(self, artifact: TritonArtifact,
                     workload: WorkloadContract) -> Mapping[str, object]:
        raise NotImplementedError("Triton full profiling is outside the integration milestone")

    def _command(self, source: Path, workload: WorkloadContract, mode: str) -> list[str]:
        shape = workload.shapes[0].dimensions
        return [self.config.python, "-m", "kernel_mcts.triton_harness",
                "--candidate", str(source), "--artifact-dir", str(source.parent),
                "--mode", mode, f"--M={shape['M']}", f"--N={shape['N']}",
                f"--K={shape['K']}", f"--rtol={workload.rtol}", f"--atol={workload.atol}",
                f"--seed={self.config.seed}", f"--warmups={self.config.warmup_count}",
                f"--measurements={self.config.measurement_count}"]

    def _execute(self, artifact: TritonArtifact, workload: WorkloadContract,
                 mode: str) -> dict[str, object]:
        try:
            result = self._runner(self._command(artifact.source_path, workload, mode),
                                  timeout=self.config.execution_timeout_seconds,
                                  env=self.config.subprocess_environment)
        except subprocess.TimeoutExpired as error:
            raise CandidateTimeoutError("Triton candidate timed out") from error
        except OSError as error:
            raise EvaluationInfrastructureError("Triton harness could not be executed") from error
        payload = _parse_payload(result.stdout)
        if result.returncode != 0 or payload.get("status") != "ok":
            if mode == "benchmark":
                raise CandidateBenchmarkError("Triton candidate benchmark failed")
            raise CandidateLaunchError("Triton candidate launch failed")
        return payload

    @staticmethod
    def _validate(program: KernelProgram, workload: WorkloadContract) -> None:
        if program.backend != "triton_h100" or workload.benchmark_id != "bf16_gemm_4096_h100":
            raise ValueError("TritonH100Backend requires its backend and fixed H100 GEMM workload")


def _run_command(command, *, timeout, env):
    return subprocess.run(command, timeout=timeout, env=dict(env), text=True,
                          capture_output=True, check=False)
