from __future__ import annotations

import os
import csv
import io
import shutil
import subprocess
from pathlib import Path

import pytest

from kernel_mcts.benchmarks import BF16_GEMM_WORKLOAD, load_bf16_gemm_root
from kernel_mcts.cuda_backend import CudaBackendConfig, CudaCppBackend
from kernel_mcts.evaluation import (
    BackendKernelEvaluator,
    CandidateLaunchError,
    CandidateTimeoutError,
    EvaluationContext,
    EvaluationInfrastructureError,
)
from kernel_mcts.domain import BenchmarkResult
from kernel_mcts.profiling import resolve_profile_metric_set


class FakeRunner:
    def __init__(self) -> None:
        self.calls = []
        self.compile_returncode = 0
        self.compile_stdout = ""
        self.compile_stderr = ""
        self.execution_stdout = '{"status":"ok","success":true,"maximum_error":0.01,"mean_error":0.001,"failed_test_id":null,"reference_metadata":{"implementation":"cuBLAS"}}'
        self.execution_returncode = 0
        self.ncu_stdout = ""
        self.ncu_stderr = ""
        self.ncu_returncode = 0
        self.ncu_query_stdout = ""
        self.timeout_stage = None

    def __call__(self, command, *, timeout, env):
        command = list(command)
        self.calls.append((command, timeout, dict(env)))
        if "-o" in command:
            if self.timeout_stage == "compile":
                raise subprocess.TimeoutExpired(command, timeout)
            output = Path(command[command.index("-o") + 1])
            if self.compile_returncode == 0:
                output.write_bytes(b"compiled executable")
            return subprocess.CompletedProcess(
                command,
                self.compile_returncode,
                self.compile_stdout,
                self.compile_stderr,
            )
        if "--dump-sass" in command:
            return subprocess.CompletedProcess(
                command,
                0,
                "/*0000*/ HMMA; /*0010*/ EXIT;",
                "",
            )
        if command[0] == "/opt/cuda/bin/ncu":
            if self.timeout_stage == "profile":
                raise subprocess.TimeoutExpired(command, timeout)
            if "--query-metrics-mode" in command:
                return subprocess.CompletedProcess(
                    command,
                    self.ncu_returncode,
                    self.ncu_query_stdout,
                    "",
                )
            return subprocess.CompletedProcess(
                command,
                self.ncu_returncode,
                self.ncu_stdout,
                self.ncu_stderr,
            )
        if self.timeout_stage == "execute":
            raise subprocess.TimeoutExpired(command, timeout)
        return subprocess.CompletedProcess(
            command,
            self.execution_returncode,
            self.execution_stdout,
            "",
        )


def backend(tmp_path, runner, **changes):
    config = {
        "artifact_root": tmp_path,
        "nvcc": "/opt/cuda/bin/nvcc",
        "cuobjdump": "/opt/cuda/bin/cuobjdump",
        "ncu": "/opt/cuda/bin/ncu",
        "warmup_count": 2,
        "measurement_count": 3,
        **changes,
    }
    return CudaCppBackend(CudaBackendConfig(**config), runner=runner)


def ncu_csv(metrics) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        ["ID", "Kernel Name", "Metric Name", "Metric Unit", "Metric Value"]
    )
    for index, metric in enumerate(metrics):
        writer.writerow([index, "bf16_gemm_root", metric, "%", f"{index + 1},234.5"])
    return output.getvalue()


def ncu_wide_csv(metrics) -> str:
    output = io.StringIO()
    writer = csv.writer(output, quoting=csv.QUOTE_ALL)
    writer.writerow(["ID", "Kernel Name", *metrics])
    writer.writerow(["", "", *("%" for _ in metrics)])
    writer.writerow(
        [
            0,
            "bf16_gemm_root",
            *(f"{index + 1},234.5" for index, _ in enumerate(metrics)),
        ]
    )
    return output.getvalue()


def test_compile_uses_shell_safe_arguments_and_sanitized_environment(tmp_path) -> None:
    runner = FakeRunner()
    subject = backend(tmp_path, runner)

    result = subject.compile(load_bf16_gemm_root(), BF16_GEMM_WORKLOAD)

    assert result.success
    assert result.artifact is not None
    command, timeout, environment = runner.calls[0]
    assert command[0] == "/opt/cuda/bin/nvcc"
    assert "-arch=sm_90" in command
    assert "-rdc=true" in command
    assert "-lcublas" in command
    assert timeout == 120.0
    assert environment == {"PATH": os.defpath, "LANG": "C", "LC_ALL": "C"}
    assert "RUNPOD_API_KEY" not in environment
    assert result.artifact.executable.read_bytes() == b"compiled executable"


def test_compile_failure_and_timeout_are_structured(tmp_path) -> None:
    failed_runner = FakeRunner()
    failed_runner.compile_returncode = 1
    failed_runner.compile_stderr = "syntax error"
    failed = backend(tmp_path / "failed", failed_runner).compile(
        load_bf16_gemm_root(), BF16_GEMM_WORKLOAD
    )

    timeout_runner = FakeRunner()
    timeout_runner.timeout_stage = "compile"
    timed_out = backend(tmp_path / "timeout", timeout_runner).compile(
        load_bf16_gemm_root(), BF16_GEMM_WORKLOAD
    )

    assert not failed.success
    assert failed.stderr == "syntax error"
    assert not failed.timed_out
    assert not timed_out.success
    assert timed_out.timed_out


def test_correctness_and_benchmark_payloads_are_preserved(tmp_path) -> None:
    runner = FakeRunner()
    subject = backend(tmp_path, runner)
    compilation = subject.compile(load_bf16_gemm_root(), BF16_GEMM_WORKLOAD)
    assert compilation.artifact is not None

    correctness = subject.check_correctness(compilation.artifact, BF16_GEMM_WORKLOAD)
    runner.execution_stdout = '{"status":"ok","timings_us":[10.0,12.0,11.0]}'
    benchmark = subject.benchmark(compilation.artifact, BF16_GEMM_WORKLOAD)

    assert correctness.success
    assert correctness.maximum_error == 0.01
    assert correctness.reference_metadata == {"implementation": "cuBLAS"}
    assert benchmark.timings_us == (10.0, 12.0, 11.0)
    assert benchmark.median_us == 11.0
    assert benchmark.mean_us == 11.0
    assert benchmark.stddev_us == pytest.approx(0.8164965809)
    assert benchmark.min_us == 10.0
    assert benchmark.max_us == 12.0
    assert benchmark.warmup_count == 2


def test_lightweight_profile_extracts_versioned_numeric_ncu_metrics(tmp_path) -> None:
    runner = FakeRunner()
    subject = backend(tmp_path, runner)
    compilation = subject.compile(load_bf16_gemm_root(), BF16_GEMM_WORKLOAD)
    assert compilation.artifact is not None
    runner.ncu_stdout = '{"status":"ok","timings_us":[10.0]}'
    runner.ncu_stderr = ncu_csv(subject.LIGHTWEIGHT_METRICS)

    profile = subject.lightweight_profile(compilation.artifact, BF16_GEMM_WORKLOAD)

    command, timeout, environment = runner.calls[-1]
    assert command[:9] == [
        "/opt/cuda/bin/ncu",
        "--csv",
        "--page",
        "raw",
        "--log-file",
        "stdout",
        "--kernel-name-base",
        "function",
        "--kernel-name",
    ]
    assert "--launch-count" in command
    assert command[command.index("--launch-count") + 1] == "1"
    assert command[command.index("--metrics") + 1] == ",".join(
        subject.LIGHTWEIGHT_METRICS
    )
    assert command[-2:] == ["--warmups=0", "--measurements=1"]
    assert timeout == 300.0
    assert environment == {"PATH": os.defpath, "LANG": "C", "LC_ALL": "C"}
    assert profile["schema_version"] == 1
    assert profile["profiler"] == "ncu"
    assert profile["metric_set"] == "lightweight_v1"
    assert profile["artifact_id"] == compilation.artifact.artifact_id
    assert profile["summary"]["registers_per_thread"] == 1234.5
    assert profile["summary"]["instructions_executed"] == 9234.5
    assert list(profile["metrics"]) == list(subject.LIGHTWEIGHT_METRICS)
    assert profile["metrics"][subject.LIGHTWEIGHT_METRICS[0]] == {
        "value": 1234.5,
        "unit": "%",
    }


def test_diagnostic_profile_validates_and_collects_sm90_metrics(tmp_path) -> None:
    runner = FakeRunner()
    subject = backend(tmp_path, runner, ncu_version="Version 2025.1.1.0")
    compilation = subject.compile(load_bf16_gemm_root(), BF16_GEMM_WORKLOAD)
    assert compilation.artifact is not None
    definition = resolve_profile_metric_set(
        "diagnostic_v2",
        "sm_90",
        "Version 2025.1.1.0",
    )
    runner.ncu_query_stdout = "\n".join(definition.metrics)
    runner.ncu_stderr = ncu_csv(definition.metrics)

    profile = subject.lightweight_profile(
        compilation.artifact,
        BF16_GEMM_WORKLOAD,
        "diagnostic_v2",
    )

    query_command = runner.calls[-2][0]
    profile_command = runner.calls[-1][0]
    assert query_command[1:] == [
        "--query-metrics-mode",
        "all",
        "--devices",
        "0",
    ]
    assert profile_command[profile_command.index("--metrics") + 1] == ",".join(
        definition.metrics
    )
    assert profile["schema_version"] == 2
    assert profile["metric_set"] == "diagnostic_v2"
    assert profile["target"] == {
        "architecture": "sm_90",
        "ncu_version": "2025.1",
    }
    assert profile["resolved_metrics"] == list(definition.metrics)
    assert "shared_bank_conflicts" in profile["summary"]


def test_diagnostic_profile_rejects_missing_ncu_metrics(tmp_path) -> None:
    runner = FakeRunner()
    subject = backend(tmp_path, runner, ncu_version="2025.1")
    compilation = subject.compile(load_bf16_gemm_root(), BF16_GEMM_WORKLOAD)
    assert compilation.artifact is not None
    runner.ncu_query_stdout = "launch__registers_per_thread"

    with pytest.raises(EvaluationInfrastructureError, match="missing:"):
        subject.lightweight_profile(
            compilation.artifact,
            BF16_GEMM_WORKLOAD,
            "diagnostic_v2",
        )


def test_diagnostic_profile_accepts_launch_metric_absent_from_query(tmp_path) -> None:
    runner = FakeRunner()
    subject = backend(tmp_path, runner, ncu_version="2025.1")
    compilation = subject.compile(load_bf16_gemm_root(), BF16_GEMM_WORKLOAD)
    assert compilation.artifact is not None
    definition = resolve_profile_metric_set("diagnostic_v2", "sm_90", "2025.1")
    runner.ncu_query_stdout = "\n".join(
        metric
        for metric in definition.metrics
        if metric != "launch__registers_per_thread"
    )
    runner.ncu_stderr = ncu_csv(definition.metrics)

    profile = subject.lightweight_profile(
        compilation.artifact,
        BF16_GEMM_WORKLOAD,
        "diagnostic_v2",
    )

    assert profile["summary"]["registers_per_thread"] == 1234.5


def test_lightweight_profile_accepts_ncu_csv_on_stdout(tmp_path) -> None:
    runner = FakeRunner()
    subject = backend(tmp_path, runner)
    compilation = subject.compile(load_bf16_gemm_root(), BF16_GEMM_WORKLOAD)
    assert compilation.artifact is not None
    runner.ncu_stdout = ncu_csv(subject.LIGHTWEIGHT_METRICS)

    profile = subject.lightweight_profile(compilation.artifact, BF16_GEMM_WORKLOAD)

    assert list(profile["metrics"]) == list(subject.LIGHTWEIGHT_METRICS)


def test_lightweight_profile_accepts_ncu_wide_csv(tmp_path) -> None:
    runner = FakeRunner()
    subject = backend(tmp_path, runner)
    compilation = subject.compile(load_bf16_gemm_root(), BF16_GEMM_WORKLOAD)
    assert compilation.artifact is not None
    runner.ncu_stdout = ncu_wide_csv(subject.LIGHTWEIGHT_METRICS)

    profile = subject.lightweight_profile(compilation.artifact, BF16_GEMM_WORKLOAD)

    assert profile["summary"]["registers_per_thread"] == 1234.5
    assert profile["summary"]["instructions_executed"] == 9234.5
    assert profile["metrics"][subject.LIGHTWEIGHT_METRICS[0]]["unit"] == "%"


def test_lightweight_profile_reports_bounded_malformed_output(tmp_path) -> None:
    runner = FakeRunner()
    subject = backend(tmp_path, runner)
    compilation = subject.compile(load_bf16_gemm_root(), BF16_GEMM_WORKLOAD)
    assert compilation.artifact is not None
    runner.ncu_stdout = "unexpected profiler output"

    with pytest.raises(
        EvaluationInfrastructureError,
        match="malformed CSV: unexpected profiler output",
    ):
        subject.lightweight_profile(compilation.artifact, BF16_GEMM_WORKLOAD)


def test_lightweight_profile_rejects_missing_metrics_and_tool_failures(tmp_path) -> None:
    missing_runner = FakeRunner()
    missing = backend(tmp_path / "missing", missing_runner)
    compilation = missing.compile(load_bf16_gemm_root(), BF16_GEMM_WORKLOAD)
    assert compilation.artifact is not None
    missing_runner.ncu_stdout = ncu_csv(missing.LIGHTWEIGHT_METRICS[:-1])

    with pytest.raises(EvaluationInfrastructureError, match="required.*metrics"):
        missing.lightweight_profile(compilation.artifact, BF16_GEMM_WORKLOAD)

    failed_runner = FakeRunner()
    failed = backend(tmp_path / "failed-profile", failed_runner)
    compilation = failed.compile(load_bf16_gemm_root(), BF16_GEMM_WORKLOAD)
    assert compilation.artifact is not None
    failed_runner.ncu_returncode = 1
    failed_runner.ncu_stderr = "==ERROR== ERR_NVGPUCTRPERM"
    with pytest.raises(
        EvaluationInfrastructureError,
        match="profiling failed:.*ERR_NVGPUCTRPERM",
    ):
        failed.lightweight_profile(compilation.artifact, BF16_GEMM_WORKLOAD)


def test_malformed_worker_output_is_infrastructure_failure(tmp_path) -> None:
    runner = FakeRunner()
    subject = backend(tmp_path, runner)
    compilation = subject.compile(load_bf16_gemm_root(), BF16_GEMM_WORKLOAD)
    assert compilation.artifact is not None
    runner.execution_stdout = "not json"

    with pytest.raises(EvaluationInfrastructureError, match="malformed JSON"):
        subject.check_correctness(compilation.artifact, BF16_GEMM_WORKLOAD)


def test_launch_failure_is_not_misclassified_as_infrastructure(tmp_path) -> None:
    runner = FakeRunner()
    subject = backend(tmp_path, runner)
    compilation = subject.compile(load_bf16_gemm_root(), BF16_GEMM_WORKLOAD)
    assert compilation.artifact is not None
    runner.execution_stdout = '{"status":"launch_failure"}'
    runner.execution_returncode = 1

    with pytest.raises(CandidateLaunchError):
        subject.check_correctness(compilation.artifact, BF16_GEMM_WORKLOAD)


def test_execution_timeout_is_a_candidate_timeout(tmp_path) -> None:
    runner = FakeRunner()
    subject = backend(tmp_path, runner)
    compilation = subject.compile(load_bf16_gemm_root(), BF16_GEMM_WORKLOAD)
    assert compilation.artifact is not None
    runner.timeout_stage = "execute"

    with pytest.raises(CandidateTimeoutError):
        subject.check_correctness(compilation.artifact, BF16_GEMM_WORKLOAD)


def test_sass_fingerprint_ignores_instruction_addresses(tmp_path) -> None:
    runner = FakeRunner()
    subject = backend(tmp_path, runner)
    compilation = subject.compile(load_bf16_gemm_root(), BF16_GEMM_WORKLOAD)
    assert compilation.artifact is not None

    first = subject.binary_fingerprint(compilation.artifact, {})
    runner.calls.clear()
    original_call = runner.__call__

    def relocated(command, *, timeout, env):
        if "--dump-sass" in command:
            return subprocess.CompletedProcess(command, 0, "/*1230*/ HMMA; /*ABC0*/ EXIT;", "")
        return original_call(command, timeout=timeout, env=env)

    subject._runner = relocated
    second = subject.binary_fingerprint(compilation.artifact, {})

    assert first == second


def test_backend_integrates_with_tier_zero_failure_semantics(tmp_path) -> None:
    runner = FakeRunner()
    subject = backend(tmp_path, runner)
    runner.execution_stdout = '{"status":"launch_failure"}'
    runner.execution_returncode = 1
    evaluator = BackendKernelEvaluator(
        backend=subject,
        root_benchmark=BenchmarkResult((1.0,), 1.0),
        context=EvaluationContext(
            "worker",
            "manifest",
            BF16_GEMM_WORKLOAD.metadata["abi"]["launch"],
            {"gpu": "H100 SXM", "cuda": "12.4"},
        ),
    )
    # The launch failure occurs before the root benchmark is read.
    result = evaluator.evaluate(load_bf16_gemm_root(), BF16_GEMM_WORKLOAD)

    assert result.invalid_reason.value == "LAUNCH_FAILURE"


@pytest.mark.skipif(
    os.environ.get("RUN_H100_INTEGRATION") != "1"
    or shutil.which("nvcc") is None
    or shutil.which("nvidia-smi") is None,
    reason="set RUN_H100_INTEGRATION=1 on a CUDA H100 worker",
)
def test_packaged_root_on_h100(tmp_path) -> None:
    subject = CudaCppBackend(CudaBackendConfig(artifact_root=tmp_path))
    compilation = subject.compile(load_bf16_gemm_root(), BF16_GEMM_WORKLOAD)

    assert compilation.success
    assert compilation.artifact is not None
    correctness = subject.check_correctness(compilation.artifact, BF16_GEMM_WORKLOAD)
    assert correctness.success
    benchmark = subject.benchmark(compilation.artifact, BF16_GEMM_WORKLOAD)
    assert benchmark.median_us > 0
