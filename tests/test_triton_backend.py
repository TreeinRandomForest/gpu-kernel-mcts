from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from kernel_mcts.benchmarks import (
    BF16_GEMM_WORKLOAD,
    load_triton_bf16_gemm_gluon_reference,
    load_triton_bf16_gemm_root,
)
from kernel_mcts.domain import BenchmarkResult, KernelProgram
from kernel_mcts.evaluation import (
    BackendKernelEvaluator, EvaluationContext, EvaluationInfrastructureError,
)
from kernel_mcts.profiling import (
    LIGHTWEIGHT_V1_METRICS, DIAGNOSTIC_V2_ADDITIONAL_METRICS,
    SPILL_V1_ADDITIONAL_METRICS,
    REUSE_V1_ADDITIONAL_METRICS,
)
from kernel_mcts.triton_backend import TritonBackendConfig, TritonH100Backend


class FakeRunner:
    def __init__(self) -> None:
        self.calls = []
        self.correct = True
        self.compile_success = True
        self.kernel_name = "bf16_gemm_triton"
        self.profiled_kernel_name = None

    def __call__(self, command, *, timeout, env):
        self.calls.append((list(command), timeout, dict(env)))
        if command[0] == "ncu":
            lines = ['"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"']
            name = self.profiled_kernel_name or self.kernel_name
            lines += [f'0,"{name}","{metric}","%","1.0"'
                      for metric in command[command.index("--metrics") + 1].split(",")]
            return subprocess.CompletedProcess(command, 0, "\n".join(lines), "")
        mode = command[command.index("--mode") + 1]
        if mode == "compile":
            if not self.compile_success:
                return subprocess.CompletedProcess(command, 1, '{"status":"candidate_failure"}', "")
            directory = Path(command[command.index("--artifact-dir") + 1])
            (directory / "kernel.cubin").write_bytes(b"cubin")
            (directory / "kernel.ptx").write_text(f".entry {self.kernel_name}() {{}}")
            payload = {"status": "ok", "kernel_name": self.kernel_name,
                       "registers_per_thread": 64, "shared_memory_bytes": 2048, "spills": 0}
        elif mode == "correctness":
            payload = {"status": "ok", "success": self.correct,
                       "maximum_error": 0.0, "mean_error": 0.0,
                       "failed_test_id": None if self.correct else "gemm"}
        elif mode == "benchmark":
            payload = {"status": "ok", "timings_us": [5.0, 6.0, 5.5]}
        else:
            raise AssertionError(command)
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")


def _backend(tmp_path, runner):
    return TritonH100Backend(
        TritonBackendConfig(artifact_root=tmp_path, python="python", ncu="ncu",
                            measurement_count=3), runner=runner
    )


def test_triton_valid_candidate_keeps_artifacts_and_profiles_lazily(tmp_path) -> None:
    runner = FakeRunner()
    backend = _backend(tmp_path, runner)
    evaluator = BackendKernelEvaluator(
        backend=backend, root_benchmark=BenchmarkResult((10.0, 10.0, 10.0), 10.0),
        context=EvaluationContext("worker", "manifest", {"dynamic": True}, {"gpu": "H100"}),
    )

    result = evaluator.evaluate(load_triton_bf16_gemm_root(), BF16_GEMM_WORKLOAD)

    assert result.status.value == "VALID"
    assert result.benchmark.median_us == 5.5
    assert result.metadata["compiler_context"]["resources"]["registers_per_thread"] == 64
    assert result.metadata["compiler_context"]["ptx"]["artifacts"][0]["storage_status"] == "eligible"
    assert len(runner.calls) == 3
    assert result.compilation.artifact_paths[-1].endswith("kernel.ptx")
    profile = evaluator.lightweight_profile(result, BF16_GEMM_WORKLOAD)
    assert profile["summary"]["registers_per_thread"] == 1.0
    assert runner.calls[-1][0][0] == "ncu"
    assert runner.calls[-1][0][runner.calls[-1][0].index("--kernel-name") + 1] == "bf16_gemm_triton"
    assert runner.calls[-1][0][runner.calls[-1][0].index("--nvtx-include") + 1] == "KERNEL_MCTS_TARGET/"
    assert profile["kernel_name"] == "bf16_gemm_triton"
    assert "RUNPOD_API_KEY" not in runner.calls[0][2]


def test_triton_diagnostic_profile_exposes_local_traffic_and_compiler_spills(tmp_path) -> None:
    runner = FakeRunner()
    backend = TritonH100Backend(
        TritonBackendConfig(artifact_root=tmp_path, python="python", ncu="ncu",
                            ncu_version="2025.1.1.0", measurement_count=3), runner=runner
    )
    compilation = backend.compile(load_triton_bf16_gemm_root(), BF16_GEMM_WORKLOAD)

    profile = backend.lightweight_profile(
        compilation.artifact, BF16_GEMM_WORKLOAD, "diagnostic_v2"
    )

    assert profile["metric_set"] == "diagnostic_v2"
    assert profile["summary"]["local_load_sectors"] == 1.0
    assert profile["summary"]["local_store_sectors"] == 1.0
    assert profile["summary"]["compiler_spills"] == 0
    assert profile["summary"]["compiler_static_shared_memory_bytes"] == 2048
    assert set(runner.calls[-1][0][runner.calls[-1][0].index("--metrics") + 1].split(",")) == (
        set(LIGHTWEIGHT_V1_METRICS) | set(DIAGNOSTIC_V2_ADDITIONAL_METRICS)
    )

    spill_profile = backend.lightweight_profile(
        compilation.artifact, BF16_GEMM_WORKLOAD, "spill_v1"
    )
    assert spill_profile["summary"]["local_load_l1_hit_sectors"] == 1.0
    assert spill_profile["summary"]["local_load_l1_miss_sectors"] == 1.0
    assert spill_profile["summary"]["compiler_spills"] == 0
    assert set(runner.calls[-1][0][runner.calls[-1][0].index("--metrics") + 1].split(",")) == (
        set(LIGHTWEIGHT_V1_METRICS) | set(DIAGNOSTIC_V2_ADDITIONAL_METRICS)
        | set(SPILL_V1_ADDITIONAL_METRICS)
    )

    reuse_profile = backend.lightweight_profile(
        compilation.artifact, BF16_GEMM_WORKLOAD, "reuse_v1"
    )
    assert reuse_profile["summary"]["l2_tex_read_sectors"] == 1.0
    assert reuse_profile["summary"]["l2_tex_read_lookup_hit_sectors"] == 1.0
    assert reuse_profile["summary"]["l2_tex_read_lookup_miss_sectors"] == 1.0
    assert reuse_profile["summary"]["dram_bytes_read"] == 1.0
    assert reuse_profile["summary"]["local_load_l1_hit_sectors"] == 1.0
    assert reuse_profile["summary"]["local_load_l1_miss_sectors"] == 1.0
    assert reuse_profile["summary"]["compiler_spills"] == 0
    assert reuse_profile["cache_control"] == "all"
    assert reuse_profile["replay_mode"] == "kernel"
    assert any("not equivalent to DRAM" in limit
               for limit in reuse_profile["interpretation_limits"])
    assert runner.calls[-1][0][1:5] == [
        "--cache-control", "all", "--replay-mode", "kernel"
    ]
    assert set(runner.calls[-1][0][runner.calls[-1][0].index("--metrics") + 1].split(",")) == (
        set(LIGHTWEIGHT_V1_METRICS) | set(DIAGNOSTIC_V2_ADDITIONAL_METRICS)
        | set(SPILL_V1_ADDITIONAL_METRICS) | set(REUSE_V1_ADDITIONAL_METRICS)
    )


def test_gluon_kernel_name_from_compile_selects_exact_ncu_target(tmp_path) -> None:
    runner = FakeRunner()
    runner.kernel_name = "gemm"
    backend = _backend(tmp_path, runner)
    compilation = backend.compile(load_triton_bf16_gemm_gluon_reference(), BF16_GEMM_WORKLOAD)
    assert compilation.success and compilation.artifact is not None
    assert compilation.artifact.kernel_name == "gemm"
    assert compilation.compiler_context["kernel_name"] == "gemm"

    profile = backend.lightweight_profile(compilation.artifact, BF16_GEMM_WORKLOAD)
    command = runner.calls[-1][0]
    assert command[command.index("--kernel-name") + 1] == "gemm"
    assert profile["selection"]["nvtx_range"] == "KERNEL_MCTS_TARGET"
    assert profile["kernel_name"] == "gemm"


def test_triton_profile_rejects_different_reported_kernel(tmp_path) -> None:
    runner = FakeRunner()
    runner.kernel_name = "gemm"
    runner.profiled_kernel_name = "unrelated"
    backend = _backend(tmp_path, runner)
    compilation = backend.compile(load_triton_bf16_gemm_gluon_reference(), BF16_GEMM_WORKLOAD)
    assert compilation.artifact is not None
    with pytest.raises(EvaluationInfrastructureError, match="did not match"):
        backend.lightweight_profile(compilation.artifact, BF16_GEMM_WORKLOAD)


def test_triton_compile_rejects_unnamed_returned_kernel(tmp_path) -> None:
    runner = FakeRunner()
    runner.kernel_name = None
    result = _backend(tmp_path, runner).compile(
        load_triton_bf16_gemm_root(), BF16_GEMM_WORKLOAD
    )
    assert not result.success
    assert result.artifact is None
    assert "named compiled kernel" in result.stderr


def test_triton_compile_and_correctness_failures_do_not_make_nodes(tmp_path) -> None:
    runner = FakeRunner()
    backend = _backend(tmp_path, runner)
    evaluator = BackendKernelEvaluator(
        backend=backend, root_benchmark=BenchmarkResult((10.0,), 10.0),
        context=EvaluationContext("worker", "manifest", {}, {}),
    )
    runner.compile_success = False
    compile_result = evaluator.evaluate(load_triton_bf16_gemm_root(), BF16_GEMM_WORKLOAD)
    assert compile_result.status.value == "INVALID"
    assert len(runner.calls) == 1

    runner.compile_success = True
    runner.correct = False
    incorrect = evaluator.evaluate(load_triton_bf16_gemm_root(), BF16_GEMM_WORKLOAD)
    assert incorrect.status.value == "INVALID"
    assert incorrect.benchmark is None
    assert len(runner.calls) == 3


def test_triton_launcher_source_participates_in_state_identity(tmp_path) -> None:
    runner = FakeRunner()
    backend = _backend(tmp_path, runner)
    root = load_triton_bf16_gemm_root()
    other = KernelProgram(root.source + "\n# different launch configuration\n", "triton_h100")
    first = backend.compile(root, BF16_GEMM_WORKLOAD).artifact
    second = backend.compile(other, BF16_GEMM_WORKLOAD).artifact
    assert first.artifact_id == second.artifact_id
    assert backend.binary_fingerprint(first, {}) != backend.binary_fingerprint(second, {})
    assert backend.normalize_program(KernelProgram("if True:\n    pass", "triton_h100")) != backend.normalize_program(
        KernelProgram("if True:\npass", "triton_h100")
    )
