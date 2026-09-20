from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .cute_program import (
    CuteGemmProgram,
    PinnedCuteGemmRenderer,
    cute_gemm_program_from_source,
    validate_cute_gemm_program,
)
from .cute_diagnostics import select_runtime_fingerprint
from .cuda_backend import (
    _ncu_csv_output,
    _ncu_diagnostic,
    _parse_ncu_csv,
    _resolve_tool,
)
from .domain import BenchmarkResult, KernelProgram, WorkloadContract
from .evaluation import EvaluationInfrastructureError
from .profiling import (
    NCU_QUERY_VALIDATION_EXEMPT_METRICS,
    ProfileMetricSet,
    normalize_ncu_version,
    resolve_profile_metric_set,
)


class CuteExecutor(Protocol):
    def __call__(self, program: CuteGemmProgram) -> Mapping[str, Any]: ...


class TelemetryReader(Protocol):
    def __call__(self) -> Mapping[str, object]: ...


class ProfileRunner(Protocol):
    def __call__(
        self,
        command: Sequence[str],
        *,
        timeout: float,
        env: Mapping[str, str],
    ) -> subprocess.CompletedProcess[str]: ...


@dataclass(frozen=True, slots=True)
class CuteBackendConfig:
    artifact_root: Path
    python_executable: str = sys.executable
    ncu: str = "ncu"
    architecture: str = "sm_90"
    ncu_version: str | None = None
    compile_timeout_seconds: float = 600.0
    profile_timeout_seconds: float = 600.0
    warmup_count: int = 10
    measurement_count: int = 30
    subprocess_environment: Mapping[str, str] = field(
        default_factory=lambda: {
            key: os.environ[key]
            for key in (
                "PATH",
                "LANG",
                "LC_ALL",
                "HOME",
                "LD_LIBRARY_PATH",
                "CUDA_VISIBLE_DEVICES",
                "PYTHONPATH",
            )
            if key in os.environ
        }
    )

    def __post_init__(self) -> None:
        if self.compile_timeout_seconds <= 0 or self.profile_timeout_seconds <= 0:
            raise ValueError("CuTe timeouts must be positive")
        if self.warmup_count < 0 or self.measurement_count < 1:
            raise ValueError("CuTe benchmark counts are invalid")


@dataclass(frozen=True, slots=True)
class CuteArtifact:
    artifact_id: str
    source_path: Path
    representation: CuteGemmProgram
    result: Mapping[str, Any]
    operating_state: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class CuteCompilationResult:
    success: bool
    artifact: CuteArtifact | None
    stdout: str
    stderr: str
    duration_seconds: float | None
    timed_out: bool = False

    @property
    def artifact_paths(self) -> tuple[str, ...]:
        return (str(self.artifact.source_path),) if self.artifact else ()


@dataclass(frozen=True, slots=True)
class CuteCorrectnessResult:
    success: bool
    maximum_error: float | None
    mean_error: float | None
    failed_test_id: str | None = None
    reference_metadata: Mapping[str, object] = field(default_factory=dict)


class CuTeDSLBackend:
    """Pinned CuTe DSL backend with one cached JIT/evaluation per rendered state."""

    name = "cute_dsl"

    def __init__(
        self,
        config: CuteBackendConfig,
        *,
        executor: CuteExecutor | None = None,
        telemetry: TelemetryReader | None = None,
        renderer: PinnedCuteGemmRenderer | None = None,
        profile_runner: ProfileRunner | None = None,
    ) -> None:
        self.config = config
        self._executor = executor or self._execute_pinned_program
        self._telemetry = telemetry or _gpu_operating_state
        self._renderer = renderer or PinnedCuteGemmRenderer()
        self._profile_runner = profile_runner or _run_command
        self._artifacts: dict[str, CuteArtifact] = {}
        self._profiles: dict[tuple[str, str], Mapping[str, object]] = {}
        self._validated_profile_metric_sets: set[str] = set()

    def normalize_program(self, program: KernelProgram) -> str:
        self._require_backend(program)
        try:
            tree = ast.parse(program.source)
        except SyntaxError:
            return program.source
        return ast.dump(tree, annotate_fields=True, include_attributes=False)

    def compile(
        self,
        program: KernelProgram,
        workload: WorkloadContract,
    ) -> CuteCompilationResult:
        started = time.monotonic()
        try:
            self._validate_contract(program, workload)
            representation = cute_gemm_program_from_source(program.source)
            legality = validate_cute_gemm_program(representation)
            if not legality.valid:
                diagnostic = "; ".join(item.message for item in legality.violations)
                return CuteCompilationResult(
                    False, None, "", diagnostic, time.monotonic() - started
                )
            if self._renderer.render(representation) != program:
                return CuteCompilationResult(
                    False,
                    None,
                    "",
                    "program is not the deterministic rendering of its representation",
                    time.monotonic() - started,
                )
        except (SyntaxError, TypeError, ValueError) as error:
            return CuteCompilationResult(
                False, None, "", str(error), time.monotonic() - started
            )

        artifact_id = hashlib.sha256(program.source.encode("utf-8")).hexdigest()
        cached = self._artifacts.get(artifact_id)
        if cached is not None:
            return CuteCompilationResult(
                True,
                cached,
                "reused cached CuTe JIT artifact",
                "",
                time.monotonic() - started,
            )

        self.config.artifact_root.mkdir(parents=True, exist_ok=True)
        artifact_dir = Path(
            tempfile.mkdtemp(prefix="cute-artifact-", dir=self.config.artifact_root)
        )
        source_path = artifact_dir / "candidate.py"
        source_path.write_text(program.source, encoding="utf-8")
        before = dict(self._telemetry())
        try:
            result = dict(self._executor(representation))
        except TimeoutError:
            return CuteCompilationResult(
                False,
                None,
                "",
                "CuTe JIT/evaluation timed out",
                time.monotonic() - started,
                True,
            )
        except EvaluationInfrastructureError:
            raise
        except Exception as error:
            return CuteCompilationResult(
                False,
                None,
                "",
                f"{type(error).__name__}: {error}",
                time.monotonic() - started,
            )
        after = dict(self._telemetry())
        _validate_execution_payload(result, self.config.measurement_count)
        artifact = CuteArtifact(
            artifact_id,
            source_path,
            representation,
            result,
            {"before": before, "after": after},
        )
        self._artifacts[artifact_id] = artifact
        return CuteCompilationResult(
            True,
            artifact,
            "CuTe JIT/evaluation completed and cached",
            "",
            time.monotonic() - started,
        )

    def check_correctness(
        self,
        artifact: CuteArtifact,
        workload: WorkloadContract,
    ) -> CuteCorrectnessResult:
        self._validate_workload(workload)
        correctness = artifact.result["correctness"]
        assert isinstance(correctness, Mapping)
        return CuteCorrectnessResult(
            bool(correctness["success"]),
            _optional_float(correctness.get("maximum_error")),
            _optional_float(correctness.get("mean_error")),
            correctness.get("failed_test_id"),
            correctness.get("reference_metadata", {}),
        )

    def benchmark(
        self,
        artifact: CuteArtifact,
        workload: WorkloadContract,
    ) -> BenchmarkResult:
        self._validate_workload(workload)
        benchmark = artifact.result["benchmark"]
        assert isinstance(benchmark, Mapping)
        timings = tuple(float(value) for value in benchmark["timings_us"])
        return BenchmarkResult(
            timings,
            statistics.median(timings),
            warmup_count=self.config.warmup_count,
            mean_us=statistics.fmean(timings),
            stddev_us=statistics.pstdev(timings),
            min_us=min(timings),
            max_us=max(timings),
            gpu_operating_state=artifact.operating_state,
        )

    def binary_fingerprint(
        self,
        artifact: CuteArtifact,
        launch_config: Mapping[str, object],
    ) -> str:
        diagnostics = artifact.result.get("jit_diagnostics")
        selected = (
            select_runtime_fingerprint(diagnostics)
            if isinstance(diagnostics, Mapping)
            else None
        )
        identity = (
            {
                "runtime_artifact_kind": selected.get("kind"),
                "runtime_artifact_sha256": selected.get("sha256"),
                "launch_config": launch_config,
            }
            if selected is not None and selected.get("sha256")
            else {
                "rendered_source_sha256": artifact.artifact_id,
                "representation_hash": artifact.representation.configuration_hash,
                "pinned_example_sha256": artifact.result.get("example_sha256"),
                "launch_config": launch_config,
            }
        )
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def evaluation_metadata(artifact: CuteArtifact) -> Mapping[str, object]:
        diagnostics = artifact.result.get("jit_diagnostics")
        selected = (
            select_runtime_fingerprint(diagnostics)
            if isinstance(diagnostics, Mapping)
            else None
        )
        value = {
            "representation": artifact.representation.as_dict(),
            "representation_schema_version": artifact.representation.schema_version,
            "configuration_hash": artifact.representation.configuration_hash,
            "artifact_fingerprint_kind": (
                selected.get("kind")
                if selected is not None
                else "rendered_source_and_pinned_template"
            ),
            "pinned_example_sha256": artifact.result.get("example_sha256"),
        }
        if selected is not None:
            value["runtime_fingerprint"] = dict(selected)
        if isinstance(diagnostics, Mapping):
            mlir = diagnostics.get("mlir")
            if isinstance(mlir, Mapping):
                value["compiler_ir"] = dict(mlir)
            kernel_names = diagnostics.get("kernel_names")
            if isinstance(kernel_names, Sequence) and not isinstance(
                kernel_names, (str, bytes)
            ):
                value["kernel_names"] = [str(name) for name in kernel_names]
        return value

    def lightweight_profile(
        self,
        artifact: CuteArtifact,
        workload: WorkloadContract,
        metric_set: str = "lightweight_v1",
    ) -> Mapping[str, object]:
        cache_key = (artifact.artifact_id, metric_set)
        cached = self._profiles.get(cache_key)
        if cached is not None:
            return cached
        try:
            definition = resolve_profile_metric_set(
                metric_set,
                self.config.architecture,
                self.config.ncu_version,
            )
        except ValueError as error:
            raise EvaluationInfrastructureError(str(error)) from error
        self._validate_profile_metric_set(definition)
        diagnostics = artifact.result.get("jit_diagnostics")
        kernel_names = (
            diagnostics.get("kernel_names", ())
            if isinstance(diagnostics, Mapping)
            else ()
        )
        selected_kernel = _select_profile_kernel_name(kernel_names)
        command = [
            _resolve_tool(self.config.ncu),
            "--csv",
            "--page",
            "raw",
            "--log-file",
            "stdout",
            "--target-processes",
            "all",
            "--kernel-name-base",
            "function",
            "--kernel-name",
            selected_kernel,
            "--launch-count",
            "1",
            "--metrics",
            ",".join(definition.metrics),
            self.config.python_executable,
            "-m",
            "kernel_mcts.cute_backend_runner",
            "--mode",
            "profile",
            "--representation-json",
            artifact.representation.canonical_json(),
        ]
        try:
            result = self._profile_runner(
                command,
                timeout=self.config.profile_timeout_seconds,
                env=self.config.subprocess_environment,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise EvaluationInfrastructureError(
                "Nsight Compute profiling could not be completed"
            ) from error
        if result.returncode != 0:
            diagnostic = _ncu_diagnostic(result.stdout, result.stderr)
            suffix = f": {diagnostic}" if diagnostic else ""
            raise EvaluationInfrastructureError(
                f"Nsight Compute profiling failed{suffix}"
            )
        metrics = _parse_ncu_csv(
            _ncu_csv_output(result.stdout, result.stderr, definition.metrics),
            definition.metrics,
        )
        profile = {
            "schema_version": definition.schema_version,
            "profiler": "ncu",
            "metric_set": definition.id,
            "target": {
                "architecture": self.config.architecture,
                "ncu_version": normalize_ncu_version(self.config.ncu_version),
                "kernel_name": selected_kernel,
                "available_kernel_names": [str(name) for name in kernel_names],
            },
            "resolved_metrics": list(definition.metrics),
            "artifact_id": artifact.artifact_id,
            "summary": {
                alias: metrics[name]["value"]
                for name, alias in definition.aliases.items()
            },
            "metrics": metrics,
        }
        self._profiles[cache_key] = profile
        return profile

    def full_profile(
        self,
        artifact: CuteArtifact,
        workload: WorkloadContract,
    ) -> Mapping[str, object]:
        raise EvaluationInfrastructureError(
            "CuTe DSL full profiling is not implemented"
        )

    def _validate_profile_metric_set(self, definition: ProfileMetricSet) -> None:
        if (
            definition.id == "lightweight_v1"
            or definition.id in self._validated_profile_metric_sets
        ):
            return
        try:
            result = self._profile_runner(
                [
                    _resolve_tool(self.config.ncu),
                    "--query-metrics-mode",
                    "all",
                    "--devices",
                    "0",
                ],
                timeout=self.config.profile_timeout_seconds,
                env=self.config.subprocess_environment,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise EvaluationInfrastructureError(
                "Nsight Compute metric validation could not be completed"
            ) from error
        if result.returncode != 0:
            raise EvaluationInfrastructureError(
                "Nsight Compute metric validation failed"
            )
        available = {
            token.strip(" ,:\"'")
            for token in f"{result.stdout}\n{result.stderr}".split()
            if "__" in token
        }
        missing = [
            metric
            for metric in definition.metrics
            if metric not in available
            and metric not in NCU_QUERY_VALIDATION_EXEMPT_METRICS
        ]
        if missing:
            raise EvaluationInfrastructureError(
                f"Nsight Compute metric set {definition.id!r} is unavailable; "
                f"missing: {', '.join(missing[:8])}"
            )
        self._validated_profile_metric_sets.add(definition.id)

    def _execute_pinned_program(self, program: CuteGemmProgram) -> Mapping[str, Any]:
        command = [
            self.config.python_executable,
            "-m",
            "kernel_mcts.cute_backend_runner",
            "--representation-json",
            program.canonical_json(),
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.config.compile_timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise TimeoutError("CuTe JIT/evaluation timed out") from error
        except OSError as error:
            raise EvaluationInfrastructureError(
                "CuTe evaluation subprocess could not be executed"
            ) from error
        try:
            payload = json.loads(completed.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError) as error:
            raise EvaluationInfrastructureError(
                "CuTe evaluation subprocess returned malformed JSON"
            ) from error
        if completed.returncode != 0 or not isinstance(payload, Mapping):
            raise EvaluationInfrastructureError("CuTe evaluation subprocess failed")
        return payload

    @staticmethod
    def _require_backend(program: KernelProgram) -> None:
        if program.backend != "cute_dsl":
            raise ValueError(f"unsupported program backend {program.backend!r}")

    def _validate_contract(
        self, program: KernelProgram, workload: WorkloadContract
    ) -> None:
        self._require_backend(program)
        self._validate_workload(workload)

    @staticmethod
    def _validate_workload(workload: WorkloadContract) -> None:
        if workload.benchmark_id != "bf16_gemm_4096_h100":
            raise ValueError("CuTeDSLBackend supports only bf16_gemm_4096_h100")
        if workload.operation != "gemm" or workload.dtype != "bfloat16":
            raise ValueError("CuTeDSLBackend supports only BF16 GEMM")
        if len(workload.shapes) != 1:
            raise ValueError("CuTeDSLBackend requires one fixed GEMM shape")
        shape = workload.shapes[0].dimensions
        if any(shape.get(name) != 4096 for name in ("M", "N", "K")):
            raise ValueError("CuTeDSLBackend requires M=N=K=4096")


def _validate_execution_payload(payload: Mapping[str, Any], count: int) -> None:
    if not payload.get("comparable_to_repository_baselines"):
        raise EvaluationInfrastructureError(
            "CuTe executor did not complete the repository evaluation contract"
        )
    correctness = payload.get("correctness")
    benchmark = payload.get("benchmark")
    if not isinstance(correctness, Mapping) or "success" not in correctness:
        raise EvaluationInfrastructureError("CuTe executor omitted correctness")
    if not isinstance(benchmark, Mapping):
        raise EvaluationInfrastructureError("CuTe executor omitted benchmark")
    timings = benchmark.get("timings_us")
    if not isinstance(timings, Sequence) or isinstance(timings, (str, bytes)):
        raise EvaluationInfrastructureError("CuTe executor returned invalid timings")
    if len(timings) != count:
        raise EvaluationInfrastructureError("CuTe executor returned unexpected timing count")
    try:
        numeric_timings = tuple(float(value) for value in timings)
    except (TypeError, ValueError) as error:
        raise EvaluationInfrastructureError("CuTe executor returned invalid timings") from error
    if any(not math.isfinite(value) or value <= 0 for value in numeric_timings):
        raise EvaluationInfrastructureError("CuTe executor returned invalid timings")


def _optional_float(value: object) -> float | None:
    return float(value) if value is not None else None


def _gpu_operating_state() -> Mapping[str, object]:
    fields = (
        "temperature.gpu",
        "clocks.sm",
        "clocks.mem",
        "power.draw",
        "power.limit",
        "pstate",
        "utilization.gpu",
        "utilization.memory",
        "clocks_event_reasons.active",
    )
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={','.join(fields)}",
                "--format=csv,noheader,nounits",
                "--id=0",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            env={"PATH": os.defpath, "LANG": "C", "LC_ALL": "C"},
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"status": "unavailable"}
    if completed.returncode != 0:
        return {"status": "unavailable"}
    values = [value.strip() for value in completed.stdout.strip().split(",")]
    if len(values) != len(fields):
        return {"status": "unavailable"}
    return {"status": "observed", **dict(zip(fields, values))}


def _select_profile_kernel_name(value: object) -> str:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise EvaluationInfrastructureError("CuTe JIT did not report kernel names")
    names = [str(name) for name in value if str(name)]
    if not names:
        raise EvaluationInfrastructureError("CuTe JIT did not report kernel names")
    return min(
        names,
        key=lambda name: (
            "gemm" not in name.casefold(),
            "copy" in name.casefold(),
            len(name),
            name,
        ),
    )


def _run_command(command, *, timeout, env):
    return subprocess.run(
        command,
        timeout=timeout,
        env=dict(env),
        text=True,
        capture_output=True,
        check=False,
    )
