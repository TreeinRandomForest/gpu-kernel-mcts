from __future__ import annotations

import math

from kernel_mcts.benchmarks import BF16_GEMM_WORKLOAD
from kernel_mcts.cute_backend import CuteBackendConfig, CuTeDSLBackend
from kernel_mcts.cute_program import PinnedCuteGemmRenderer, REFERENCE_CUTE_GEMM
from kernel_mcts.domain import BenchmarkResult, CompileStatus, CorrectnessStatus
from kernel_mcts.evaluation import BackendKernelEvaluator, EvaluationContext


def _payload(median: float = 100.0):
    timings = [median] * 30
    return {
        "correctness": {
            "success": True,
            "maximum_error": 0.0,
            "mean_error": 0.0,
            "failed_test_id": None,
            "reference_metadata": {"implementation": "cuBLAS"},
        },
        "benchmark": {"timings_us": timings, "median_us": median},
        "comparable_to_repository_baselines": True,
        "example_sha256": "example-hash",
    }


def test_backend_evaluates_once_and_reuses_cached_artifact(tmp_path) -> None:
    executions = []
    telemetry_calls = []

    def execute(program):
        executions.append(program)
        return _payload()

    def telemetry():
        telemetry_calls.append(True)
        return {"status": "observed", "sample": len(telemetry_calls)}

    backend = CuTeDSLBackend(
        CuteBackendConfig(tmp_path), executor=execute, telemetry=telemetry
    )
    rendered = PinnedCuteGemmRenderer().render(REFERENCE_CUTE_GEMM)

    first = backend.compile(rendered, BF16_GEMM_WORKLOAD)
    second = backend.compile(rendered, BF16_GEMM_WORKLOAD)

    assert first.success is True
    assert first.artifact is not None
    assert second.artifact is first.artifact
    assert "reused cached" in second.stdout
    assert executions == [REFERENCE_CUTE_GEMM]
    assert len(telemetry_calls) == 2
    correctness = backend.check_correctness(first.artifact, BF16_GEMM_WORKLOAD)
    benchmark = backend.benchmark(first.artifact, BF16_GEMM_WORKLOAD)
    assert correctness.success is True
    assert correctness.reference_metadata == {"implementation": "cuBLAS"}
    assert benchmark.median_us == 100.0
    assert benchmark.gpu_operating_state["before"]["sample"] == 1
    assert benchmark.gpu_operating_state["after"]["sample"] == 2


def test_backend_rejects_source_not_matching_deterministic_rendering(tmp_path) -> None:
    executions = []
    backend = CuTeDSLBackend(
        CuteBackendConfig(tmp_path),
        executor=lambda program: executions.append(program),
        telemetry=lambda: {},
    )
    rendered = PinnedCuteGemmRenderer().render(REFERENCE_CUTE_GEMM)
    tampered = type(rendered)(rendered.source + "\n# mutation\n", rendered.backend)

    result = backend.compile(tampered, BF16_GEMM_WORKLOAD)

    assert result.success is False
    assert "not the deterministic rendering" in result.stderr
    assert executions == []


def test_backend_kernel_evaluator_returns_normal_cached_evaluation(tmp_path) -> None:
    executions = []

    def execute(program):
        executions.append(program)
        return _payload(100.0)

    backend = CuTeDSLBackend(
        CuteBackendConfig(tmp_path), executor=execute, telemetry=lambda: {}
    )
    evaluator = BackendKernelEvaluator(
        backend=backend,
        root_benchmark=BenchmarkResult((200.0,), 200.0),
        context=EvaluationContext(
            worker_id="worker",
            environment_manifest_id="manifest",
            launch_config={"tile_shape_mn": [128, 256], "cluster_shape_mn": [1, 1]},
            hardware_toolchain={"gpu": "H100", "cutlass": "4.5.1"},
        ),
    )
    rendered = PinnedCuteGemmRenderer().render(REFERENCE_CUTE_GEMM)

    first = evaluator.evaluate(rendered, BF16_GEMM_WORKLOAD)
    second = evaluator.evaluate(rendered, BF16_GEMM_WORKLOAD)

    assert first.status.value == "VALID"
    assert first.compile_status == CompileStatus.SUCCESS
    assert first.correctness_status == CorrectnessStatus.PASS
    assert first.reward == math.log(2.0)
    assert first.state_key == second.state_key
    assert first.binary_hash == second.binary_hash
    assert first.compiled_artifact is second.compiled_artifact
    assert first.metadata["representation"] == REFERENCE_CUTE_GEMM.as_dict()
    assert first.metadata["configuration_hash"] == REFERENCE_CUTE_GEMM.configuration_hash
    assert (
        first.metadata["artifact_fingerprint_kind"]
        == "rendered_source_and_pinned_template"
    )
    assert executions == [REFERENCE_CUTE_GEMM]


def test_fingerprint_changes_with_launch_context(tmp_path) -> None:
    backend = CuTeDSLBackend(
        CuteBackendConfig(tmp_path), executor=lambda program: _payload(), telemetry=lambda: {}
    )
    rendered = PinnedCuteGemmRenderer().render(REFERENCE_CUTE_GEMM)
    compilation = backend.compile(rendered, BF16_GEMM_WORKLOAD)
    assert compilation.artifact is not None

    first = backend.binary_fingerprint(compilation.artifact, {"cluster": [1, 1]})
    second = backend.binary_fingerprint(compilation.artifact, {"cluster": [2, 1]})

    assert first != second


def test_correctness_failure_remains_distinct_from_jit_success(tmp_path) -> None:
    payload = _payload()
    payload["correctness"] = {
        "success": False,
        "maximum_error": 1.0,
        "mean_error": 0.1,
        "failed_test_id": "fixed_shape",
    }
    backend = CuTeDSLBackend(
        CuteBackendConfig(tmp_path), executor=lambda program: payload, telemetry=lambda: {}
    )
    evaluator = BackendKernelEvaluator(
        backend=backend,
        root_benchmark=BenchmarkResult((200.0,), 200.0),
        context=EvaluationContext("worker", "manifest", {}, {"gpu": "H100"}),
    )

    result = evaluator.evaluate(
        PinnedCuteGemmRenderer().render(REFERENCE_CUTE_GEMM),
        BF16_GEMM_WORKLOAD,
    )

    assert result.status.value == "INVALID"
    assert result.compile_status == CompileStatus.SUCCESS
    assert result.correctness_status == CorrectnessStatus.FAIL
    assert result.invalid_reason.value == "CORRECTNESS_FAILURE"
