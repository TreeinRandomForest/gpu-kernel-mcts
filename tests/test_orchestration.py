from __future__ import annotations

import json
import hashlib
import math
import sqlite3
from dataclasses import replace

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
from kernel_mcts.generation import GenerationResult, ProposalBudgetKind
from kernel_mcts.llm import LLMCompletion
from kernel_mcts.llm_generation import LLMKernelGenerator
from kernel_mcts.orchestration import (
    WorkerKernelEvaluator,
    run_candidate_evaluation,
    run_mcts_search,
    run_sequential_search,
)
from kernel_mcts.persistence import SQLiteTraceStore
from kernel_mcts.priors import UniformStrategyPrior
from kernel_mcts.providers import EnvironmentManifest, HardwareSpec
from kernel_mcts.search import MCTSConfig
from kernel_mcts.sequential import SequentialConfig
from kernel_mcts.serialization import serialize_evaluation


def test_worker_evaluator_persists_ptx_before_node_event(tmp_path) -> None:
    raw = b".visible .entry root() {}"
    sha256 = hashlib.sha256(raw).hexdigest()

    class ArtifactWorker(MockWorker):
        def evaluate(self, evaluation_id, program, workload, profile_level):
            result = super().evaluate(evaluation_id, program, workload, profile_level)
            return replace(result, metadata={"compiler_context": {
                "ptx": {"artifacts": [{
                    "sha256": sha256, "size_bytes": len(raw),
                    "storage_status": "eligible",
                }]},
            }})

        def fetch_compiler_artifact(self, evaluation_id, requested_sha256):
            assert requested_sha256 == sha256
            return raw

    path = tmp_path / "trace.sqlite"
    with SQLiteTraceStore(path) as trace:
        trace.start_run("run", "mock-kernel", "mcts", {})
        trace.emit("run_started", {"workload": {}, "hardware": {}})
        result = WorkerKernelEvaluator(ArtifactWorker(), "run", trace).evaluate(
            KernelProgram("root", "cuda_cpp"), WORKLOAD
        )
        assert result.metadata["compiler_context"]["ptx"]["artifacts"][0][
            "storage_status"
        ] == "stored"
        trace.emit("node_created", {
            "node_id": "root", "state_key": "root", "reward": 0.0,
            "evaluation": serialize_evaluation(result),
        })
    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT content FROM compiler_artifacts WHERE sha256 = ?", (sha256,)
        ).fetchone()[0] == raw
        assert connection.execute(
            "SELECT node_id FROM node_compiler_artifacts WHERE sha256 = ?", (sha256,)
        ).fetchone()[0] == "root"


def test_search_prompt_reads_recorded_parent_ptx_graph(tmp_path) -> None:
    raw = (b".visible .entry kernel(.param .u64 p0) {\n"
           b"ld.param.u64 %rd1, [p0];\n"
           b"ld.global.u32 %r1, [%rd1];\nret;\n}\n")
    sha256 = hashlib.sha256(raw).hexdigest()

    class ArtifactWorker(MockWorker):
        def evaluate(self, evaluation_id, program, workload, profile_level):
            result = super().evaluate(evaluation_id, program, workload, profile_level)
            if result.status != ProposalStatus.VALID:
                return result
            return replace(result, metadata={"compiler_context": {
                "entry_point": "kernel",
                "ptx": {"artifacts": [{"sha256": sha256, "storage_status": "eligible"}]},
            }})

        def fetch_compiler_artifact(self, evaluation_id, requested_sha256):
            assert requested_sha256 == sha256
            return raw

    class RecordingClient:
        def __init__(self):
            self.prompts = []

        def complete(self, prompt):
            self.prompts.append(prompt)
            return LLMCompletion("generation-1", "candidate", "fake-model", metadata={})

    client = RecordingClient()
    database = tmp_path / "dataflow-prompt.sqlite"
    with SQLiteTraceStore(database) as trace:
        run_mcts_search(
            provider=MockProvider(ArtifactWorker()), hardware=HARDWARE,
            workload=WORKLOAD, root_program=KernelProgram("root"),
            strategies=(STRATEGY,), generator=LLMKernelGenerator(client),
            prior_provider=UniformStrategyPrior(), generation_budget=1,
            trace=trace, run_id="dataflow-prompt",
            mcts_config=MCTSConfig(k_max=1, max_repairs=0, dataflow_prompt_view="full"),
        )

    with sqlite3.connect(database) as connection:
        prompt_text, prompt_hash, metadata_text = connection.execute(
            "SELECT prompt_text, prompt_hash, metadata_json FROM generations"
        ).fetchone()
        config = json.loads(connection.execute("SELECT config_json FROM search_runs").fetchone()[0])
    assert prompt_text == client.prompts[0]
    assert prompt_hash == hashlib.sha256(prompt_text.encode()).hexdigest()
    view = json.loads(prompt_text)["parent_dataflow"]
    assert view["view"] == "full"
    assert view["ptx_sha256"] == sha256
    assert view["adjacency"]["sites"]
    assert any(edge[2] == "parameter_provenance" for edge in view["adjacency"]["edges"])
    assert json.loads(metadata_text)["generation"]["parent_dataflow"] == view
    assert config["mcts"]["dataflow_prompt_view"] == "full"


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
        self.profile_calls = []
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

    def profile(
        self,
        evaluation_id,
        profile_level,
        metric_set="lightweight_v1",
    ):
        self.profile_calls.append((evaluation_id, profile_level, metric_set))
        return {
            "schema_version": 1,
            "profiler": "ncu",
            "metric_set": "test",
            "metrics": {"occupancy": {"value": 50.0, "unit": "%"}},
        }


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
        self.requests = []

    def generate(self, request):
        self.calls += 1
        self.requests.append(request)
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
                profile_metric_set="diagnostic_v2",
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
    assert generator.requests[0].profile["profiler"] == "ncu"
    candidate_ids = [call[0] for call in provider.worker.calls if call[1] == "candidate-1"]
    assert len(candidate_ids) == 2
    assert candidate_ids[0] == candidate_ids[1]
    transposed_ids = [call[0] for call in provider.worker.calls if call[1] == "candidate-2"]
    assert len(transposed_ids) == 2
    assert transposed_ids[0] == transposed_ids[1]
    assert {call[3] for call in provider.worker.calls} == {"tier0"}
    assert provider.worker.profile_calls
    assert len(provider.worker.profile_calls) == len(set(provider.worker.profile_calls))
    assert {call[2] for call in provider.worker.profile_calls} == {"diagnostic_v2"}

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
            "SELECT final_b_gen, final_iterations, final_profile_calls, environment_manifest_id "
            "FROM search_runs WHERE run_id = 'mock-run'"
        ).fetchone() == (4, 3, execution.result.profile_calls, MANIFEST.manifest_id)
        assert connection.execute(
            "SELECT json_extract(config_json, '$.mcts.profile_metric_set') "
            "FROM search_runs WHERE run_id = 'mock-run'"
        ).fetchone() == ("diagnostic_v2",)
        assert connection.execute(
            "SELECT reward FROM nodes WHERE run_id = ? AND node_id = ?",
            (execution.run_id, execution.result.root.id),
        ).fetchone() == (0.0,)


def test_mock_run_passes_mutation_budget_into_search_and_trace(tmp_path) -> None:
    class MutationGenerator:
        def __init__(self):
            self.issued = False

        @staticmethod
        def proposal_budget_kind(_request):
            return ProposalBudgetKind.MUTATION

        def can_generate(self, _request):
            return not self.issued

        def generate(self, _request):
            self.issued = True
            return GenerationResult(
                "mutation:1",
                "candidate-mutation",
                KernelProgram("candidate-mutation"),
                "mutation",
                metadata={"proposal_mechanism": "typed_mutation"},
            )

    database = tmp_path / "mutation.sqlite"
    with SQLiteTraceStore(database) as trace:
        execution = run_mcts_search(
            provider=MockProvider(),
            hardware=HARDWARE,
            workload=WORKLOAD,
            root_program=KernelProgram("root"),
            strategies=(STRATEGY,),
            generator=MutationGenerator(),
            prior_provider=UniformStrategyPrior(),
            generation_budget=0,
            mutation_budget=1,
            trace=trace,
            run_id="mutation-run",
        )

    assert execution.result.generations == 0
    assert execution.result.mutations == 1
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT generation_budget, mutation_budget, final_b_gen, final_b_mut "
            "FROM search_runs WHERE run_id = 'mutation-run'"
        ).fetchone() == (0, 1, 0, 1)


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


def test_sequential_orchestration_reuses_worker_and_releases_it(tmp_path):
    provider = MockProvider()
    generator = MockGenerator()
    path = tmp_path / "sequential-worker.sqlite"
    with SQLiteTraceStore(path) as trace:
        execution = run_sequential_search(
            provider=provider, hardware=HARDWARE, workload=WORKLOAD,
            root_program=KernelProgram("root"), generator=generator,
            generation_budget=3, trace=trace,
            config=SequentialConfig(max_repairs=0, dataflow_view="none"),
            run_id="sequential-worker",
        )
    assert provider.acquisitions == provider.releases == 1
    assert execution.result.generations == 3
    assert execution.result.current.evaluation.program.source == "candidate-2"
    assert len(provider.worker.profile_calls) == 3
    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT algorithm, final_b_gen, final_profile_calls FROM search_runs"
        ).fetchone() == ("sequential", 3, 3)


def test_sequential_next_prompt_reads_recorded_ptx_graph(tmp_path):
    raw = (b".visible .entry kernel(.param .u64 p0) {\n"
           b"ld.param.u64 %rd1, [p0];\n"
           b"ld.global.u32 %r1, [%rd1];\nret;\n}\n")
    sha256 = hashlib.sha256(raw).hexdigest()

    class ArtifactWorker(MockWorker):
        def evaluate(self, evaluation_id, program, workload, profile_level):
            result = super().evaluate(evaluation_id, program, workload, profile_level)
            if result.status != ProposalStatus.VALID:
                return result
            return replace(result, metadata={"compiler_context": {
                "schema_version": 2, "entry_point": "kernel", "architecture": "sm_90",
                "ptx": {"artifacts": [{"sha256": sha256, "storage_status": "eligible"}]},
            }})

        def fetch_compiler_artifact(self, evaluation_id, requested_sha256):
            assert requested_sha256 == sha256
            return raw

    provider = MockProvider(ArtifactWorker())
    generator = MockGenerator()
    path = tmp_path / "sequential-dataflow.sqlite"
    with SQLiteTraceStore(path) as trace:
        run_sequential_search(
            provider=provider, hardware=HARDWARE, workload=WORKLOAD,
            root_program=KernelProgram("root"), generator=generator,
            generation_budget=2, trace=trace,
            config=SequentialConfig(max_repairs=0, dataflow_view="paths"),
            run_id="sequential-dataflow",
        )
    assert generator.requests[0].dataflow_view["status"] != "unavailable"
    assert generator.requests[1].dataflow_view["view"] == "paths"
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT count(*) FROM node_dataflow_graphs").fetchone()[0] == 2
