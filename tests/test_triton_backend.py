from __future__ import annotations

import json
import subprocess
from pathlib import Path

from kernel_mcts.benchmarks import BF16_GEMM_WORKLOAD, load_triton_bf16_gemm_root
from kernel_mcts.domain import BenchmarkResult, KernelProgram
from kernel_mcts.evaluation import BackendKernelEvaluator, EvaluationContext
from kernel_mcts.profiling import LIGHTWEIGHT_V1_METRICS
from kernel_mcts.triton_backend import TritonBackendConfig, TritonH100Backend


class FakeRunner:
    def __init__(self) -> None:
        self.calls = []
        self.correct = True
        self.compile_success = True

    def __call__(self, command, *, timeout, env):
        self.calls.append((list(command), timeout, dict(env)))
        if command[0] == "ncu":
            lines = ['"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"']
            lines += [f'0,"bf16_gemm_triton","{name}","%","1.0"'
                      for name in LIGHTWEIGHT_V1_METRICS]
            return subprocess.CompletedProcess(command, 0, "\n".join(lines), "")
        mode = command[command.index("--mode") + 1]
        if mode == "compile":
            if not self.compile_success:
                return subprocess.CompletedProcess(command, 1, '{"status":"candidate_failure"}', "")
            directory = Path(command[command.index("--artifact-dir") + 1])
            (directory / "kernel.cubin").write_bytes(b"cubin")
            (directory / "kernel.ptx").write_text(".entry bf16_gemm_triton() {}")
            payload = {"status": "ok", "kernel_name": "bf16_gemm_triton",
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
    assert "regex:.*bf16_gemm_triton.*" in runner.calls[-1][0]
    assert "RUNPOD_API_KEY" not in runner.calls[0][2]


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
