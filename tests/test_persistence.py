import sqlite3

import pytest

from kernel_mcts.budget import GenerationBudget
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
from kernel_mcts.persistence import SQLiteTraceStore
from kernel_mcts.priors import UniformStrategyPrior
from kernel_mcts.providers import EnvironmentManifest
from kernel_mcts.search import MCTS, MCTSConfig
from kernel_mcts.serialization import serialize_environment_manifest


def test_trace_store_records_run_and_event(tmp_path) -> None:
    path = tmp_path / "trace.sqlite"
    with SQLiteTraceStore(path) as store:
        store.start_run("run", "toy", "mcts", {"budget": 1})
        store.emit("proposal", {"status": "VALID"})
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT count(*) FROM search_events").fetchone()[0] == 1


def test_trace_store_records_run_model_name(tmp_path) -> None:
    path = tmp_path / "trace.sqlite"
    with SQLiteTraceStore(path) as store:
        store.start_run(
            "run",
            "toy",
            "mcts",
            {"generation_budget": 2, "model_name": "test-model"},
        )

    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT model_name FROM search_runs WHERE run_id = 'run'"
        ).fetchone() == ("test-model",)


def test_trace_store_creates_versioned_structured_schema(tmp_path) -> None:
    path = tmp_path / "trace.sqlite"
    with SQLiteTraceStore(path) as store:
        assert store.connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1

    with sqlite3.connect(path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        assert {
            "search_runs",
            "search_events",
            "environment_manifests",
            "generations",
            "nodes",
            "strategy_edges",
            "realization_edges",
            "iterations",
            "iteration_steps",
        } <= tables
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 3


def test_trace_store_additively_migrates_existing_search_runs(tmp_path) -> None:
    path = tmp_path / "trace.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """CREATE TABLE search_runs (
                run_id TEXT PRIMARY KEY,
                benchmark_id TEXT NOT NULL,
                algorithm TEXT NOT NULL,
                config_json TEXT NOT NULL,
                started_at TEXT NOT NULL
            )"""
        )
        connection.execute(
            "INSERT INTO search_runs VALUES ('old', 'toy', 'mcts', '{}', 'timestamp')"
        )

    with SQLiteTraceStore(path):
        pass

    with sqlite3.connect(path) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(search_runs)").fetchall()
        }
        assert {
            "seed",
            "generation_budget",
            "final_b_gen",
            "final_b_prior",
            "workload_json",
            "hardware_json",
            "toolchain_json",
        } <= columns
        assert connection.execute(
            "SELECT benchmark_id FROM search_runs WHERE run_id = 'old'"
        ).fetchone() == ("toy",)


def test_trace_store_rejects_newer_schema(tmp_path) -> None:
    path = tmp_path / "trace.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA user_version = 999")

    try:
        SQLiteTraceStore(path)
    except RuntimeError as error:
        assert "newer than supported" in str(error)
    else:
        raise AssertionError("expected newer schema to be rejected")


def test_mcts_events_materialize_complete_search_trace(tmp_path) -> None:
    workload = WorkloadContract(
        "toy",
        "increment",
        "fp32",
        (ShapeCase({"n": 1}, 1.0),),
        0.0,
        0.0,
    )
    strategy = Strategy("increment", "increment", {"cuda_cpp": "increment"})

    class Generator:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, request):
            self.calls += 1
            value = int(request.parent.source) + 1
            return GenerationResult(
                f"generation:{self.calls}",
                str(value),
                KernelProgram(str(value)),
                "prompt-hash",
                prompt_text="full generation prompt",
                input_tokens=10,
                output_tokens=5,
            )

    class Evaluator:
        def evaluate(self, program, workload):
            value = int(program.source)
            return EvaluationResult(
                ProposalStatus.VALID,
                program,
                f"state:{value}",
                float(value),
                BenchmarkResult((float(value),), float(value)),
                compile_status=CompileStatus.SUCCESS,
                correctness_status=CorrectnessStatus.PASS,
                source_hash=f"source:{value}",
                binary_hash=f"binary:{value}",
                launch_config={"block": [32, 1, 1]},
            )

    class Profiler:
        def lightweight_profile(self, evaluation, workload):
            return {"occupancy": 0.75}

    root_evaluation = EvaluationResult(
        ProposalStatus.VALID,
        KernelProgram("0"),
        "state:0",
        0.0,
        BenchmarkResult((1.0,), 1.0),
        compile_status=CompileStatus.SUCCESS,
        correctness_status=CorrectnessStatus.PASS,
    )
    path = tmp_path / "trace.sqlite"
    with SQLiteTraceStore(path) as store:
        store.start_run("run", "toy", "mcts", {"generation_budget": 2})
        result = MCTS(
            strategies=(strategy,),
            workload=workload,
            generator=Generator(),
            evaluator=Evaluator(),
            prior_provider=UniformStrategyPrior(),
            budget=GenerationBudget(2),
            config=MCTSConfig(k_max=1),
            profiler=Profiler(),
            events=store,
        ).run(root_evaluation)

    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT count(*) FROM generations").fetchone() == (2,)
        assert connection.execute("SELECT count(*) FROM nodes").fetchone() == (3,)
        assert connection.execute("SELECT count(*) FROM iterations").fetchone() == (2,)
        assert connection.execute("SELECT count(*) FROM iteration_steps").fetchone() == (3,)
        assert connection.execute("SELECT count(*) FROM strategy_edges").fetchone() == (2,)
        assert connection.execute("SELECT count(*) FROM realization_edges").fetchone() == (2,)
        assert connection.execute(
            "SELECT final_b_gen, final_b_prior, final_profile_calls, "
            "final_iterations, best_node_id "
            "FROM search_runs WHERE run_id = 'run'"
        ).fetchone() == (2, 0, result.profile_calls, 2, result.best.id)
        assert connection.execute(
            "SELECT selection_mode FROM iteration_steps "
            "WHERE iteration = 2 ORDER BY step"
        ).fetchall() == [("UCB",), ("EXPAND",)]
        assert connection.execute(
            "SELECT q_mean, q_max FROM strategy_edges "
            "WHERE parent_node_id = ? AND strategy_id = 'increment'",
            (result.root.id,),
        ).fetchone() == (1.5, 2.0)
        assert connection.execute(
            "SELECT compile_status, correctness_status, input_tokens, output_tokens, "
            "prompt_text "
            "FROM generations ORDER BY b_gen LIMIT 1"
        ).fetchone() == ("SUCCESS", "PASS", 10, 5, "full generation prompt")
        assert connection.execute(
            "SELECT profile_json FROM nodes WHERE node_id = ?",
            (result.root.id,),
        ).fetchone() == ('{"occupancy": 0.75}',)
        assert connection.execute(
            "SELECT count(*) FROM search_events WHERE event_type = 'node_profiled'"
        ).fetchone() == (result.profile_calls,)


def test_final_snapshot_persists_invalid_proposal_counters(tmp_path) -> None:
    workload = WorkloadContract(
        "toy",
        "invalid",
        "fp32",
        (ShapeCase({"n": 1}, 1.0),),
        0.0,
        0.0,
    )
    strategy = Strategy("invalid", "invalid", {"cuda_cpp": "invalid"})

    class Generator:
        def generate(self, request):
            return GenerationResult(
                "generation",
                "invalid",
                KernelProgram("invalid"),
                "prompt-hash",
            )

    class Evaluator:
        def evaluate(self, program, workload):
            return EvaluationResult(
                ProposalStatus.INVALID,
                program=program,
                invalid_reason=InvalidReason.COMPILE_FAILURE,
                compile_status=CompileStatus.FAIL,
            )

    root = EvaluationResult(
        ProposalStatus.VALID,
        KernelProgram("root"),
        "root",
        0.0,
        BenchmarkResult((1.0,), 1.0),
        compile_status=CompileStatus.SUCCESS,
        correctness_status=CorrectnessStatus.PASS,
    )
    path = tmp_path / "trace.sqlite"
    with SQLiteTraceStore(path) as store:
        store.start_run("run", "toy", "mcts", {})
        result = MCTS(
            strategies=(strategy,),
            workload=workload,
            generator=Generator(),
            evaluator=Evaluator(),
            prior_provider=UniformStrategyPrior(),
            budget=GenerationBudget(1),
            config=MCTSConfig(max_repairs=0),
            events=store,
        ).run(root)

    with sqlite3.connect(path) as connection:
        assert connection.execute(
            """SELECT visits, q_max, proposal_count, generation_attempt_count,
                      valid_proposal_count, invalid_proposal_count
               FROM strategy_edges
               WHERE parent_node_id = ? AND strategy_id = 'invalid'""",
            (result.root.id,),
        ).fetchone() == (0, None, 1, 1, 0, 1)


def test_materialization_failure_rolls_back_raw_event(tmp_path) -> None:
    path = tmp_path / "trace.sqlite"
    with SQLiteTraceStore(path) as store:
        store.start_run("run", "toy", "mcts", {})
        with pytest.raises((KeyError, ValueError)):
            store.emit("node_created", {"node_id": "incomplete"})
        assert store.connection.execute("SELECT count(*) FROM search_events").fetchone() == (0,)


def test_environment_manifest_event_is_persisted_and_linked_to_run(tmp_path) -> None:
    manifest = EnvironmentManifest(
        worker_id="worker",
        provider="runpod",
        pod_id="pod",
        gpu_model="NVIDIA H100",
        gpu_uuid="GPU-123",
        compute_capability="9.0",
        form_factor="SXM",
        captured_at="2026-09-09T12:00:00+00:00",
        toolchain_versions={"cuda_toolkit": "12.4"},
    )
    path = tmp_path / "trace.sqlite"
    with SQLiteTraceStore(path) as store:
        store.start_run("run", "toy", "mcts", {})
        store.emit("environment_manifest", serialize_environment_manifest(manifest))

    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT worker_id, gpu_model, form_factor FROM environment_manifests"
        ).fetchone() == ("worker", "NVIDIA H100", "SXM")
        assert connection.execute(
            "SELECT environment_manifest_id FROM search_runs WHERE run_id = 'run'"
        ).fetchone() == (manifest.manifest_id,)
