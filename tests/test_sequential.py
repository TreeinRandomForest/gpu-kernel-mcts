from __future__ import annotations

import json
import sqlite3
from dataclasses import replace

from kernel_mcts.domain import (
    BenchmarkResult, CompileStatus, CorrectnessStatus, EvaluationResult,
    InvalidReason, KernelProgram, ProposalStatus, ShapeCase, WorkloadContract,
)
from kernel_mcts.generation import GenerationResult
from kernel_mcts.llm import LLMCompletion
from kernel_mcts.persistence import SQLiteTraceStore
from kernel_mcts.sequential import SequentialConfig, _visible_profile, run_sequential_loop
from kernel_mcts.sequential_generation import SequentialKernelGenerator, build_sequential_prompt
from kernel_mcts.resource_graph import h100_sxm_resource_graph


WORKLOAD = WorkloadContract("test", "elementwise", "fp32", (ShapeCase({"n": 1}, 1.0),), 0, 0)


def valid(source: str, latency: float) -> EvaluationResult:
    return EvaluationResult(
        ProposalStatus.VALID, KernelProgram(source), f"state:{source}",
        0.0 if source == "root" else (100.0 - latency) / 100.0,
        BenchmarkResult((latency,), latency),
        compile_status=CompileStatus.SUCCESS,
        correctness_status=CorrectnessStatus.PASS,
    )


class Generator:
    def __init__(self, sources):
        self.sources = iter(sources)
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        source = next(self.sources)
        return GenerationResult(
            f"generation-{len(self.requests)}", source, KernelProgram(source),
            f"hash-{len(self.requests)}", prompt_text=source,
        )


class Evaluator:
    def evaluate(self, program, workload):
        if program.source == "invalid":
            return EvaluationResult(
                ProposalStatus.INVALID, program,
                invalid_reason=InvalidReason.COMPILE_FAILURE,
                compile_status=CompileStatus.FAIL,
            )
        return valid(program.source, {"fast": 80.0, "slow": 120.0}[program.source])


class Profiler:
    def __init__(self):
        self.sources = []

    def lightweight_profile(self, evaluation, workload):
        self.sources.append(evaluation.program.source)
        return {"metric_set": "lightweight_v1", "summary": {"source": evaluation.program.source}}


def test_sequential_accepts_slower_valid_candidate_and_keeps_best(tmp_path):
    generator = Generator(["fast", "invalid", "slow"])
    profiler = Profiler()
    path = tmp_path / "sequential.sqlite"
    with SQLiteTraceStore(path) as trace:
        trace.start_run("run", WORKLOAD.benchmark_id, "sequential", {})
        result = run_sequential_loop(
            root_evaluation=valid("root", 100.0), workload=WORKLOAD,
            hardware={"gpu_model": "H100"}, generator=generator,
            evaluator=Evaluator(), profiler=profiler, trace=trace,
            generation_budget=3, config=SequentialConfig(max_repairs=0),
        )
    assert [request.parent.source for request in generator.requests] == ["root", "fast", "fast"]
    assert result.current.evaluation.program.source == "slow"
    assert result.best.evaluation.program.source == "fast"
    assert result.accepted_depth == 2
    assert result.generations == 3
    assert result.profile_calls == 3
    assert profiler.sources == ["root", "fast", "slow"]
    assert generator.requests[1].recent_feedback["accepted"] is True
    assert generator.requests[2].recent_feedback["accepted"] is False
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT count(*) FROM nodes").fetchone()[0] == 3
        assert connection.execute("SELECT count(*) FROM generations").fetchone()[0] == 3
        assert connection.execute("SELECT algorithm, final_b_gen, final_profile_calls FROM search_runs").fetchone() == ("sequential", 3, 3)
        assert connection.execute("SELECT count(*) FROM search_events WHERE event_type='sequential_step'").fetchone()[0] == 3


def test_repairs_consume_budget_and_invalid_does_not_advance(tmp_path):
    generator = Generator(["invalid", "fast"])
    with SQLiteTraceStore(tmp_path / "repair.sqlite") as trace:
        trace.start_run("run", WORKLOAD.benchmark_id, "sequential", {})
        result = run_sequential_loop(
            root_evaluation=valid("root", 100.0), workload=WORKLOAD,
            hardware={}, generator=generator, evaluator=Evaluator(),
            profiler=Profiler(), trace=trace, generation_budget=2,
            config=SequentialConfig(max_repairs=1, dataflow_view="none"),
        )
    assert result.generations == 2
    assert result.iterations == 1
    assert result.current.evaluation.program.source == "fast"
    assert [request.parent.source for request in generator.requests] == ["root", "root"]
    assert generator.requests[1].previous_result.invalid_reason == InvalidReason.COMPILE_FAILURE


def test_unconstrained_prompt_and_strategy_menu():
    from kernel_mcts.domain import Strategy
    from kernel_mcts.generation import GenerationRequest

    request = GenerationRequest(
        parent=KernelProgram("root"), strategy=Strategy("freeform", "", {"cuda_cpp": ""}),
        workload=WORKLOAD, hardware={}, profile=None,
        current_benchmark={"median_us": 100.0},
    )
    plain = json.loads(build_sequential_prompt(request))
    menu = json.loads(build_sequential_prompt(request, ("tile: change tile",)))
    assert "strategy" not in plain and "strategy_menu" not in plain
    assert menu["strategy_menu"] == ["tile: change tile"]
    assert plain["current_benchmark"]["median_us"] == 100.0


def test_best_so_far_context_is_opt_in_and_only_shown_away_from_best(tmp_path):
    generator = Generator(["fast", "slow", "fast", "slow"])
    with SQLiteTraceStore(tmp_path / "best-context.sqlite") as trace:
        trace.start_run("run", WORKLOAD.benchmark_id, "sequential", {})
        result = run_sequential_loop(
            root_evaluation=valid("root", 100.0), workload=WORKLOAD,
            hardware={}, generator=generator, evaluator=Evaluator(),
            profiler=Profiler(), trace=trace, generation_budget=4,
            config=SequentialConfig(
                max_repairs=0, dataflow_view="none", include_best_so_far=True,
            ),
        )
    assert [request.best_so_far for request in generator.requests] == [
        None, None, {"source": "fast", "median_us": 80.0}, None,
    ]
    assert "best_so_far" not in json.loads(build_sequential_prompt(generator.requests[1]))
    assert json.loads(build_sequential_prompt(generator.requests[2]))["best_so_far"] == {
        "source": "fast", "median_us": 80.0,
    }
    assert result.best.evaluation.program.source == "fast"
    assert result.current.evaluation.program.source == "slow"
    assert result.generations == 4
    assert result.profile_calls == 3


def test_hardware_graph_context_is_opt_in_and_recorded_in_prompt(tmp_path):
    graph = h100_sxm_resource_graph({
        "manifest_id": "h100-test", "gpu_model": "H100 80 GB",
        "form_factor": "SXM", "compute_capability": "9.0",
        "gpu_count": 1, "mig_configuration": {},
    })
    assert graph is not None
    generator = Generator(["fast"])
    with SQLiteTraceStore(tmp_path / "hardware-context.sqlite") as trace:
        trace.start_run("run", WORKLOAD.benchmark_id, "sequential", {})
        run_sequential_loop(
            root_evaluation=valid("root", 100.0), workload=WORKLOAD,
            hardware={}, generator=generator, evaluator=Evaluator(),
            profiler=Profiler(), trace=trace, generation_budget=1,
            config=SequentialConfig(
                max_repairs=0, dataflow_view="none", include_hardware_graph=True,
            ), resource_graph=graph,
        )
    payload = json.loads(build_sequential_prompt(generator.requests[0]))
    assert payload["hardware_resource_graph"]["manifest_id"] == "h100-test"
    assert payload["hardware_resource_graph"]["constraints"]
    assert payload["hardware_resource_graph"]["links"]


def test_generator_requires_hypothesis_and_complete_source():
    from kernel_mcts.domain import Strategy
    from kernel_mcts.generation import GenerationRequest

    class Client:
        def complete(self, prompt):
            return LLMCompletion("r1", json.dumps({"hypothesis": "fuse loads", "source": "kernel"}), "fake")

    request = GenerationRequest(KernelProgram("root"), Strategy("freeform", "", {"cuda_cpp": ""}), WORKLOAD, {}, None)
    result = SequentialKernelGenerator(Client()).generate(request)
    assert result.program.source == "kernel"
    assert result.metadata["hypothesis"] == "fuse loads"


def test_context_visibility_does_not_change_profile_collection(tmp_path):
    generator = Generator(["fast"])
    profiler = Profiler()
    root = replace(valid("root", 100.0), metadata={"compiler_context": {
        "schema_version": 2, "architecture": "sm_90",
        "ptxas": {"status": "parsed", "resources": {"registers_per_thread": 64}},
        "sass": {"status": "parsed", "categories": {"global_load": 4}},
    }})
    with SQLiteTraceStore(tmp_path / "visibility.sqlite") as trace:
        trace.start_run("run", WORKLOAD.benchmark_id, "sequential", {})
        run_sequential_loop(
            root_evaluation=root, workload=WORKLOAD, hardware={},
            generator=generator, evaluator=Evaluator(), profiler=profiler,
            trace=trace, generation_budget=1,
            config=SequentialConfig(
                max_repairs=0, profile_visibility="none",
                compiler_detail="resources", dataflow_view="none",
            ),
        )
    assert profiler.sources == ["root", "fast"]
    assert generator.requests[0].profile is None
    assert generator.requests[0].compiler_summary["ptxas_resources"]["registers_per_thread"] == 64
    assert "sass_static_instruction_sites" not in generator.requests[0].compiler_summary
    assert generator.requests[0].dataflow_view is None


def test_lightweight_prompt_view_filters_diagnostic_metrics():
    profile = {
        "metric_set": "diagnostic_v2",
        "summary": {"registers_per_thread": 64, "dram_bytes_read": 1000},
        "metrics": {"launch__registers_per_thread": {"value": 64}, "dram__bytes_read.sum": {"value": 1000}},
        "resolved_metrics": ["launch__registers_per_thread", "dram__bytes_read.sum"],
    }
    visible = _visible_profile(profile, "lightweight")
    assert visible["metric_set"] == "diagnostic_v2"
    assert visible["summary"] == {"registers_per_thread": 64}
    assert list(visible["metrics"]) == ["launch__registers_per_thread"]
    assert len(profile["metrics"]) == 2
