from __future__ import annotations

import math
import subprocess

from kernel_mcts.benchmarks import BF16_GEMM_WORKLOAD
from kernel_mcts.cute_backend import CuteBackendConfig, CuTeDSLBackend
from kernel_mcts.cute_program import PinnedCuteGemmRenderer, REFERENCE_CUTE_GEMM
from kernel_mcts.cute_independent import make_independent_cute_gemm
from kernel_mcts.cute_independent_program import IndependentCuteGemmRenderer
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
        "jit_diagnostics": {
            "kernel_names": ["kernel_mcts_hopper_gemm"],
            "runtime_artifacts": [],
            "mlir": {
                "normalized_sha256": "a" * 64,
                "normalized_bytes": 100,
                "normalized_text": "module @gemm {}\n",
            },
        },
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


def test_backend_accepts_and_caches_independent_typed_root(tmp_path) -> None:
    executions = []

    def execute(program):
        executions.append(program)
        payload = _payload(665.0)
        payload["implementation"] = "independent_cute_gemm_v1"
        payload.pop("example_sha256")
        return payload

    backend = CuTeDSLBackend(
        CuteBackendConfig(tmp_path), executor=execute, telemetry=lambda: {}
    )
    representation = make_independent_cute_gemm()
    rendered = IndependentCuteGemmRenderer().render(representation)

    first = backend.compile(rendered, BF16_GEMM_WORKLOAD)
    second = backend.compile(rendered, BF16_GEMM_WORKLOAD)

    assert first.success is True
    assert first.artifact is not None
    assert second.artifact is first.artifact
    assert executions == [representation]
    metadata = backend.evaluation_metadata(first.artifact)
    assert metadata["implementation"] == "independent_cute_gemm_v1"
    assert metadata["configuration_hash"] == representation.configuration_hash


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
        == "normalized_mlir"
    )
    assert executions == [REFERENCE_CUTE_GEMM]


def test_effective_fingerprint_includes_launch_context(tmp_path) -> None:
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


def test_fingerprint_prefers_normalized_mlir_and_records_compiler_ir(tmp_path) -> None:
    backend = CuTeDSLBackend(
        CuteBackendConfig(tmp_path), executor=lambda program: _payload(), telemetry=lambda: {}
    )
    compilation = backend.compile(
        PinnedCuteGemmRenderer().render(REFERENCE_CUTE_GEMM),
        BF16_GEMM_WORKLOAD,
    )
    assert compilation.artifact is not None

    fingerprint = backend.binary_fingerprint(compilation.artifact, {})
    metadata = backend.evaluation_metadata(compilation.artifact)

    assert fingerprint != "a" * 64
    assert len(fingerprint) == 64
    assert metadata["artifact_fingerprint_kind"] == "normalized_mlir"
    assert metadata["runtime_fingerprint"]["sha256"] == "a" * 64
    assert metadata["compiler_ir"]["normalized_text"] == "module @gemm {}\n"


def test_lightweight_profile_targets_reported_kernel_and_is_cached(tmp_path) -> None:
    calls = []

    def profile_runner(command, *, timeout, env):
        calls.append(command)
        from kernel_mcts.profiling import LIGHTWEIGHT_V1_METRICS

        rows = ['"Metric Name","Metric Unit","Metric Value"']
        rows.extend(f'"{name}","%","1.5"' for name in LIGHTWEIGHT_V1_METRICS)
        return subprocess.CompletedProcess(command, 0, "\n".join(rows), "")

    backend = CuTeDSLBackend(
        CuteBackendConfig(tmp_path),
        executor=lambda program: _payload(),
        telemetry=lambda: {},
        profile_runner=profile_runner,
    )
    compilation = backend.compile(
        PinnedCuteGemmRenderer().render(REFERENCE_CUTE_GEMM),
        BF16_GEMM_WORKLOAD,
    )
    assert compilation.artifact is not None

    first = backend.lightweight_profile(compilation.artifact, BF16_GEMM_WORKLOAD)
    second = backend.lightweight_profile(compilation.artifact, BF16_GEMM_WORKLOAD)

    assert first is second
    assert len(calls) == 1
    command = calls[0]
    assert command[command.index("--kernel-name") + 1] == "kernel_mcts_hopper_gemm"
    assert command[command.index("--mode") + 1] == "profile"
    assert first["target"]["kernel_name"] == "kernel_mcts_hopper_gemm"
