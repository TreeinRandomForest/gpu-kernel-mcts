from __future__ import annotations

import json
import math
import sqlite3

import pytest

from kernel_mcts.domain import (
    BenchmarkResult,
    CompileStatus,
    CorrectnessStatus,
    EvaluationResult,
    InvalidReason,
    KernelProgram,
    ProposalStatus,
    ShapeCase,
    Strategy,
    WorkloadContract,
)
from kernel_mcts.generation import GenerationResult
from kernel_mcts.orchestration import run_candidate_evaluation, run_mcts_search
from kernel_mcts.persistence import SQLiteTraceStore
from kernel_mcts.priors import UniformStrategyPrior
from kernel_mcts.providers import EnvironmentManifest, HardwareSpec
from kernel_mcts.search import MCTSConfig


WORKLOAD = WorkloadContract(
    "mock-kernel",
    "mock-operation",
    "fp32",
    (ShapeCase({"n": 1}, 1.0),),
    0.0,
    0.0,
)
HARDWARE = HardwareSpec("Mock GPU", form_factor="SXM")
STRATEGY = Strategy("optimize", "optimize", {"cuda_cpp": "optimize"})
MANIFEST = EnvironmentManifest(
    worker_id="worker-1",
    provider="mock",
    pod_id="pod-1",
    gpu_model="Mock GPU",
    compute_capability="9.0",
    form_factor="SXM",
    captured_at="2026-09-11T00:00:00+00:00",
)


def valid(program: KernelProgram, reward: float) -> EvaluationResult:
    return EvaluationResult(
        ProposalStatus.VALID,
        program,
        f"state:{program.source}",
        reward,
        BenchmarkResult((100.0 / (reward + 1.0),), 100.0 / (reward + 1.0)),
        compile_status=CompileStatus.SUCCESS,
        correctness_status=CorrectnessStatus.PASS,
        source_hash=f"source:{program.source}",
        binary_hash=f"binary:{program.source}",
        launch_config={"block": [1, 1, 1]},
        worker_id=MANIFEST.worker_id,
        environment_manifest_id=MANIFEST.manifest_id,
    )


class MockWorker:
    worker_id = "worker-1"

    def __init__(self) -> None:
        self.calls = []
        self.attempts = {}

    def get_environment_manifest(self):
        return MANIFEST

    def evaluate(self, evaluation_id, program, workload, profile_level):
        self.calls.append((evaluation_id, program.source, workload, profile_level))
        self.attempts[evaluation_id] = self.attempts.get(evaluation_id, 0) + 1
        if program.source == "root":
            # Simulate a worker reward computed against an earlier calibration.
            return valid(program, 0.25)
        if program.source == "candidate-1" and self.attempts[evaluation_id] == 1:
            return EvaluationResult(ProposalStatus.INFRASTRUCTURE_FAILURE)
        if program.source == "candidate-1":
            return valid(program, 1.0)
        if program.source == "invalid":
            return EvaluationResult(
                ProposalStatus.INVALID,
                program=program,
                invalid_reason=InvalidReason.COMPILE_FAILURE,
                compile_status=CompileStatus.FAIL,
                worker_id=MANIFEST.worker_id,
                environment_manifest_id=MANIFEST.manifest_id,
            )
        return valid(program, 2.0)


class MockProvider:
    def __init__(self, worker=None) -> None:
        self.worker = worker or MockWorker()
        self.acquisitions = 0
        self.releases = 0

    def acquire_worker(self, hardware):
        assert hardware == HARDWARE
        self.acquisitions += 1
        return self.worker

    def release_worker(self, worker):
        assert worker is self.worker
        self.releases += 1


class MockGenerator:
    def __init__(self) -> None:
        self.sources = iter(("candidate-1", "invalid", "candidate-2", "candidate-2"))
        self.calls = 0

    def generate(self, request):
        self.calls += 1
        source = next(self.sources)
        return GenerationResult(
            f"generation-{self.calls}",
            source,
            KernelProgram(source),
            f"prompt-{self.calls}",
        )


class OneCandidateGenerator:
    def generate(self, request):
        return GenerationResult(
            "generation-1",
            "candidate",
            KernelProgram("candidate"),
            "prompt-hash",
        )


class TransientRootWorker(MockWorker):
    def evaluate(self, evaluation_id, program, workload, profile_level):
        self.calls.append((evaluation_id, program.source, workload, profile_level))
        self.attempts[evaluation_id] = self.attempts.get(evaluation_id, 0) + 1
        if program.source == "root" and self.attempts[evaluation_id] == 1:
            return EvaluationResult(
                ProposalStatus.INFRASTRUCTURE_FAILURE,
                metadata={"error_type": "WorkerProtocolError"},
            )
        if program.source == "root":
            return valid(program, 0.25)
        return valid(program, 1.0)


class FailedRootWorker(MockWorker):
    def evaluate(self, evaluation_id, program, workload, profile_level):
        self.calls.append((evaluation_id, program.source, workload, profile_level))
        return EvaluationResult(
            ProposalStatus.INFRASTRUCTURE_FAILURE,
            metadata={"error_type": "WorkerProtocolError"},
        )


def test_mock_run_wires_worker_search_budget_and_sqlite(tmp_path) -> None:
    provider = MockProvider()
    generator = MockGenerator()
    database = tmp_path / "trace.sqlite"

    with SQLiteTraceStore(database) as trace:
        execution = run_mcts_search(
            provider=provider,
            hardware=HARDWARE,
            workload=WORKLOAD,
            root_program=KernelProgram("root"),
            strategies=(STRATEGY,),
            generator=generator,
            prior_provider=UniformStrategyPrior(),
            generation_budget=4,
            trace=trace,
            mcts_config=MCTSConfig(
                k_max=1,
                max_repairs=1,
                max_infrastructure_retries=1,
            ),
            run_id="mock-run",
        )

    assert execution.result.generations == 4
    assert execution.result.iterations == 3
    assert execution.result.root.reward == 0.0
    assert execution.result.best.program.source == "candidate-2"
    candidate_1 = next(
        node for node in execution.result.nodes if node.program.source == "candidate-1"
    )
    candidate_2 = next(
        node for node in execution.result.nodes if node.program.source == "candidate-2"
    )
    assert candidate_1.reward == pytest.approx(math.log(1.6))
    assert candidate_2.reward == pytest.approx(math.log(2.4))
    assert len(execution.result.nodes) == 3
    assert provider.acquisitions == provider.releases == 1
    assert generator.calls == 4
    candidate_ids = [call[0] for call in provider.worker.calls if call[1] == "candidate-1"]
    assert len(candidate_ids) == 2
    assert candidate_ids[0] == candidate_ids[1]
    transposed_ids = [call[0] for call in provider.worker.calls if call[1] == "candidate-2"]
    assert len(transposed_ids) == 2
    assert transposed_ids[0] == transposed_ids[1]
    assert {call[3] for call in provider.worker.calls} == {"tier0"}

    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT count(*) FROM environment_manifests").fetchone() == (1,)
        assert connection.execute("SELECT count(*) FROM generations").fetchone() == (4,)
        assert connection.execute("SELECT count(*) FROM nodes").fetchone() == (3,)
        assert connection.execute("SELECT count(*) FROM iterations").fetchone() == (3,)
        assert connection.execute(
            "SELECT proposal_status FROM generations ORDER BY b_gen"
        ).fetchall() == [("VALID",), ("INVALID",), ("VALID",), ("VALID",)]
        assert connection.execute(
            "SELECT reused_node FROM generations ORDER BY b_gen"
        ).fetchall() == [(0,), (0,), (0,), (1,)]
        assert connection.execute(
            "SELECT DISTINCT worker_id, environment_manifest_id FROM generations"
        ).fetchall() == [(MANIFEST.worker_id, MANIFEST.manifest_id)]
        assert connection.execute(
            "SELECT final_b_gen, final_iterations, environment_manifest_id "
            "FROM search_runs WHERE run_id = 'mock-run'"
        ).fetchone() == (4, 3, MANIFEST.manifest_id)
        assert connection.execute(
            "SELECT reward FROM nodes WHERE run_id = ? AND node_id = ?",
            (execution.run_id, execution.result.root.id),
        ).fetchone() == (0.0,)


def test_mock_run_releases_worker_when_search_raises(tmp_path) -> None:
    provider = MockProvider()

    class FailingGenerator:
        def generate(self, request):
            raise RuntimeError("generation failed")

    with SQLiteTraceStore(tmp_path / "failed.sqlite") as trace:
        with pytest.raises(RuntimeError, match="generation failed"):
            run_mcts_search(
                provider=provider,
                hardware=HARDWARE,
                workload=WORKLOAD,
                root_program=KernelProgram("root"),
                strategies=(STRATEGY,),
                generator=FailingGenerator(),
                prior_provider=UniformStrategyPrior(),
                generation_budget=1,
                trace=trace,
                run_id="failed-run",
            )
        row = trace.connection.execute(
            "SELECT ended_at, final_b_gen, final_iterations "
            "FROM search_runs WHERE run_id = 'failed-run'"
        ).fetchone()

    assert provider.acquisitions == provider.releases == 1
    assert row[0] is not None
    assert row[1:] == (1, 1)


def test_candidate_evaluation_retries_infrastructure_and_normalizes_reward() -> None:
    provider = MockProvider()

    execution = run_candidate_evaluation(
        provider=provider,
        hardware=HARDWARE,
        workload=WORKLOAD,
        root_program=KernelProgram("root"),
        candidate_program=KernelProgram("candidate-1"),
        run_id="candidate-run",
        max_infrastructure_retries=1,
    )

    assert execution.run_id == "candidate-run"
    assert execution.root.reward == 0.0
    assert execution.candidate.status == ProposalStatus.VALID
    assert execution.candidate.reward == pytest.approx(math.log(1.6))
    assert execution.environment_manifest["manifest_id"] == MANIFEST.manifest_id
    assert [call[1] for call in provider.worker.calls] == [
        "root",
        "candidate-1",
        "candidate-1",
    ]
    assert provider.acquisitions == provider.releases == 1


def test_mcts_root_retries_infrastructure_with_same_evaluation_id(tmp_path) -> None:
    provider = MockProvider(TransientRootWorker())
    database = tmp_path / "root-retry.sqlite"

    with SQLiteTraceStore(database) as trace:
        execution = run_mcts_search(
            provider=provider,
            hardware=HARDWARE,
            workload=WORKLOAD,
            root_program=KernelProgram("root"),
            strategies=(STRATEGY,),
            generator=OneCandidateGenerator(),
            prior_provider=UniformStrategyPrior(),
            generation_budget=1,
            trace=trace,
            mcts_config=MCTSConfig(max_infrastructure_retries=1),
            run_id="root-retry",
        )

    root_calls = [call for call in provider.worker.calls if call[1] == "root"]
    assert len(root_calls) == 2
    assert root_calls[0][0] == root_calls[1][0]
    assert execution.result.generations == 1
    assert execution.result.root.reward == 0.0
    assert provider.acquisitions == provider.releases == 1
    with sqlite3.connect(database) as connection:
        attempts = connection.execute(
            "SELECT payload_json FROM search_events "
            "WHERE run_id = 'root-retry' AND event_type = 'root_evaluation_attempt' "
            "ORDER BY id"
        ).fetchall()
    payloads = [json.loads(row[0]) for row in attempts]
    assert [item["attempt"] for item in payloads] == [1, 2]
    assert [item["evaluation"]["status"] for item in payloads] == [
        "INFRASTRUCTURE_FAILURE",
        "VALID",
    ]


def test_mcts_root_exhausted_retries_are_traced_without_generation(tmp_path) -> None:
    provider = MockProvider(FailedRootWorker())
    database = tmp_path / "root-failed.sqlite"

    with SQLiteTraceStore(database) as trace:
        with pytest.raises(
            RuntimeError,
            match=(
                "after 2 attempt.*INFRASTRUCTURE_FAILURE.*"
                "error_type=WorkerProtocolError"
            ),
        ):
            run_mcts_search(
                provider=provider,
                hardware=HARDWARE,
                workload=WORKLOAD,
                root_program=KernelProgram("root"),
                strategies=(STRATEGY,),
                generator=OneCandidateGenerator(),
                prior_provider=UniformStrategyPrior(),
                generation_budget=50,
                trace=trace,
                mcts_config=MCTSConfig(max_infrastructure_retries=1),
                run_id="root-failed",
            )

    assert len(provider.worker.calls) == 2
    assert provider.worker.calls[0][0] == provider.worker.calls[1][0]
    assert provider.acquisitions == provider.releases == 1
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT count(*) FROM generations WHERE run_id = 'root-failed'"
        ).fetchone() == (0,)
        row = connection.execute(
            "SELECT final_b_gen, final_iterations FROM search_runs "
            "WHERE run_id = 'root-failed'"
        ).fetchone()
        failed_payload = json.loads(
            connection.execute(
                "SELECT payload_json FROM search_events "
                "WHERE run_id = 'root-failed' AND event_type = 'run_failed'"
            ).fetchone()[0]
        )
    assert row == (0, 0)
    assert failed_payload["root_evaluation_attempts"] == 2
    assert failed_payload["root_evaluation_status"] == "INFRASTRUCTURE_FAILURE"
    assert failed_payload["root_error_type"] == "WorkerProtocolError"
