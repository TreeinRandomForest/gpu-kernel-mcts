from __future__ import annotations

import math
from dataclasses import dataclass, replace

import pytest

from kernel_mcts.backends import CompiledArtifact
from kernel_mcts.domain import (
    BenchmarkResult,
    CompileStatus,
    CorrectnessStatus,
    InvalidReason,
    KernelProgram,
    ProposalStatus,
    ShapeCase,
    WorkloadContract,
)
from kernel_mcts.evaluation import (
    BackendKernelEvaluator,
    CandidateBenchmarkError,
    EvaluationContext,
    EvaluationInfrastructureError,
)


WORKLOAD = WorkloadContract(
    "gemm",
    "gemm",
    "bfloat16",
    (ShapeCase({"M": 1, "N": 1, "K": 1}, 1.0),),
    0.02,
    0.02,
)
PROGRAM = KernelProgram("kernel source")
ROOT_BENCHMARK = BenchmarkResult((10.0, 12.0), 11.0)
CONTEXT = EvaluationContext(
    "worker-1",
    "manifest-1",
    {"block": [16, 16, 1]},
    {"gpu": "H100 SXM", "cuda": "12.4"},
)


@dataclass
class Artifact:
    artifact_id: str = "artifact-1"


@dataclass
class Compilation:
    success: bool
    artifact: CompiledArtifact | None
    stdout: str = "compiler output"
    stderr: str = ""


@dataclass
class Correctness:
    success: bool
    maximum_error: float | None = 0.01
    mean_error: float | None = 0.001


class FakeBackend:
    name = "fake"

    def __init__(self) -> None:
        self.calls = []
        self.compilation = Compilation(True, Artifact())
        self.correctness = Correctness(True)
        self.benchmark_result = BenchmarkResult((5.0, 6.0), 5.5)
        self.compile_error = None
        self.correctness_error = None
        self.benchmark_error = None
        self.binary_hash = "binary-1"

    def normalize_program(self, program):
        self.calls.append("normalize")
        return " ".join(program.source.split())

    def compile(self, program, workload):
        self.calls.append("compile")
        if self.compile_error:
            raise self.compile_error
        return self.compilation

    def check_correctness(self, artifact, workload):
        self.calls.append("correctness")
        if self.correctness_error:
            raise self.correctness_error
        return self.correctness

    def benchmark(self, artifact, workload):
        self.calls.append("benchmark")
        if self.benchmark_error:
            raise self.benchmark_error
        return self.benchmark_result

    def binary_fingerprint(self, artifact, launch_config):
        self.calls.append("fingerprint")
        return self.binary_hash

    def lightweight_profile(self, artifact, workload):
        raise AssertionError("tier-zero evaluation must not profile")

    def full_profile(self, artifact, workload):
        raise AssertionError("tier-zero evaluation must not profile")


def evaluator(backend: FakeBackend, context: EvaluationContext = CONTEXT):
    return BackendKernelEvaluator(
        backend=backend,
        root_benchmark=ROOT_BENCHMARK,
        context=context,
    )


def test_valid_evaluation_preserves_all_tier_zero_evidence() -> None:
    backend = FakeBackend()

    result = evaluator(backend).evaluate(PROGRAM, WORKLOAD)

    assert backend.calls == [
        "normalize",
        "compile",
        "correctness",
        "benchmark",
        "fingerprint",
    ]
    assert result.status == ProposalStatus.VALID
    assert result.compile_status == CompileStatus.SUCCESS
    assert result.correctness_status == CorrectnessStatus.PASS
    assert result.program == PROGRAM
    assert result.compiled_artifact is backend.compilation.artifact
    assert result.compilation is not None
    assert result.compilation.artifact_id == "artifact-1"
    assert result.compilation.stdout == "compiler output"
    assert result.correctness is not None
    assert result.correctness.maximum_error == 0.01
    assert result.benchmark == backend.benchmark_result
    assert result.reward == pytest.approx(math.log(2.0))
    assert result.source_hash is not None
    assert result.binary_hash == "binary-1"
    assert result.state_key is not None
    assert result.worker_id == "worker-1"
    assert result.environment_manifest_id == "manifest-1"
    assert result.launch_config == {"block": [16, 16, 1]}


def test_compile_failure_stops_the_pipeline() -> None:
    backend = FakeBackend()
    backend.compilation = Compilation(False, None, stderr="compile failed")

    result = evaluator(backend).evaluate(PROGRAM, WORKLOAD)

    assert backend.calls == ["normalize", "compile"]
    assert result.status == ProposalStatus.INVALID
    assert result.invalid_reason == InvalidReason.COMPILE_FAILURE
    assert result.compile_status == CompileStatus.FAIL
    assert result.correctness_status == CorrectnessStatus.NOT_TESTED
    assert result.compilation is not None
    assert result.compilation.stderr == "compile failed"


def test_correctness_failure_stops_before_benchmark() -> None:
    backend = FakeBackend()
    backend.correctness = Correctness(False, 2.0, 1.0)

    result = evaluator(backend).evaluate(PROGRAM, WORKLOAD)

    assert backend.calls == ["normalize", "compile", "correctness"]
    assert result.status == ProposalStatus.INVALID
    assert result.invalid_reason == InvalidReason.CORRECTNESS_FAILURE
    assert result.compile_status == CompileStatus.SUCCESS
    assert result.correctness_status == CorrectnessStatus.FAIL
    assert result.compiled_artifact is backend.compilation.artifact
    assert result.correctness is not None
    assert result.correctness.maximum_error == 2.0


@pytest.mark.parametrize(
    "failure",
    [CandidateBenchmarkError("launch failed"), None],
)
def test_benchmark_failure_is_invalid(failure) -> None:
    backend = FakeBackend()
    if failure is None:
        backend.benchmark_result = BenchmarkResult((0.0,), 0.0)
    else:
        backend.benchmark_error = failure

    result = evaluator(backend).evaluate(PROGRAM, WORKLOAD)

    assert result.status == ProposalStatus.INVALID
    assert result.invalid_reason == InvalidReason.BENCHMARK_FAILURE
    assert result.compile_status == CompileStatus.SUCCESS
    assert result.correctness_status == CorrectnessStatus.PASS


@pytest.mark.parametrize("stage", ["compile", "correctness", "benchmark"])
def test_infrastructure_failure_preserves_completed_stage_status(stage: str) -> None:
    backend = FakeBackend()
    setattr(backend, f"{stage}_error", EvaluationInfrastructureError("worker lost"))

    result = evaluator(backend).evaluate(PROGRAM, WORKLOAD)

    assert result.status == ProposalStatus.INFRASTRUCTURE_FAILURE
    assert result.invalid_reason is None
    assert result.metadata == {"error_type": "EvaluationInfrastructureError"}
    if stage == "compile":
        assert result.compile_status == CompileStatus.NOT_ATTEMPTED
        assert result.correctness_status == CorrectnessStatus.NOT_TESTED
    elif stage == "correctness":
        assert result.compile_status == CompileStatus.SUCCESS
        assert result.correctness_status == CorrectnessStatus.NOT_TESTED
    else:
        assert result.compile_status == CompileStatus.SUCCESS
        assert result.correctness_status == CorrectnessStatus.PASS


def test_unexpected_backend_errors_are_not_misclassified() -> None:
    backend = FakeBackend()
    backend.compile_error = ValueError("bad configuration")

    with pytest.raises(ValueError, match="bad configuration"):
        evaluator(backend).evaluate(PROGRAM, WORKLOAD)


def test_valid_slower_candidate_has_negative_reward() -> None:
    backend = FakeBackend()
    backend.benchmark_result = BenchmarkResult((22.0,), 22.0)

    result = evaluator(backend).evaluate(PROGRAM, WORKLOAD)

    assert result.status == ProposalStatus.VALID
    assert result.reward == pytest.approx(math.log(0.5))
    assert result.reward < 0


def test_weighted_reward_uses_each_shape_timing() -> None:
    workload = WorkloadContract(
        "weighted",
        "gemm",
        "bfloat16",
        (
            ShapeCase({"M": 1, "N": 2, "K": 3}, 0.25),
            ShapeCase({"M": 4, "N": 5, "K": 6}, 0.75),
        ),
        0.02,
        0.02,
    )
    root = BenchmarkResult(
        (10.0, 20.0),
        15.0,
        {"K=3,M=1,N=2": 10.0, "K=6,M=4,N=5": 20.0},
    )
    backend = FakeBackend()
    backend.benchmark_result = BenchmarkResult(
        (5.0, 40.0),
        22.5,
        {"K=3,M=1,N=2": 5.0, "K=6,M=4,N=5": 40.0},
    )
    subject = BackendKernelEvaluator(backend=backend, root_benchmark=root, context=CONTEXT)

    result = subject.evaluate(PROGRAM, workload)

    expected = 0.25 * math.log(2.0) + 0.75 * math.log(0.5)
    assert result.reward == pytest.approx(expected)


def test_state_key_deduplicates_identical_binary_across_source_and_worker() -> None:
    backend = FakeBackend()
    subject = evaluator(backend)
    first = subject.evaluate(PROGRAM, WORKLOAD)
    second = subject.evaluate(PROGRAM, WORKLOAD)

    changed_source = subject.evaluate(KernelProgram("different source"), WORKLOAD)
    changed_worker = evaluator(
        FakeBackend(),
        replace(CONTEXT, worker_id="worker-2", environment_manifest_id="manifest-2"),
    ).evaluate(PROGRAM, WORKLOAD)

    assert first.state_key == second.state_key
    assert changed_source.source_hash != first.source_hash
    assert changed_source.state_key == first.state_key
    assert changed_worker.state_key == first.state_key


def test_state_key_includes_binary_launch_workload_and_hardware_context() -> None:
    first = evaluator(FakeBackend()).evaluate(PROGRAM, WORKLOAD)
    changed_launch = evaluator(
        FakeBackend(),
        replace(CONTEXT, launch_config={"block": [32, 8, 1]}),
    ).evaluate(PROGRAM, WORKLOAD)
    changed_hardware = evaluator(
        FakeBackend(),
        replace(CONTEXT, hardware_toolchain={"gpu": "H100 PCIe", "cuda": "12.4"}),
    ).evaluate(PROGRAM, WORKLOAD)
    changed_binary_backend = FakeBackend()
    changed_binary_backend.binary_hash = "binary-2"
    changed_binary = evaluator(changed_binary_backend).evaluate(PROGRAM, WORKLOAD)
    changed_workload = evaluator(FakeBackend()).evaluate(
        PROGRAM,
        replace(WORKLOAD, benchmark_id="another-workload"),
    )

    assert len({
        first.state_key,
        changed_launch.state_key,
        changed_hardware.state_key,
        changed_binary.state_key,
        changed_workload.state_key,
    }) == 5


def test_returned_evaluation_is_cached_data_not_a_reevaluation_hook() -> None:
    backend = FakeBackend()
    result = evaluator(backend).evaluate(PROGRAM, WORKLOAD)
    calls_after_evaluation = list(backend.calls)

    assert result.benchmark is not None
    assert result.compiled_artifact is backend.compilation.artifact
    assert result.reward is not None
    assert backend.calls == calls_after_evaluation
