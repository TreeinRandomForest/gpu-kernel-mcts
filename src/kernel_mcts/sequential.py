"""Linear, single-session CUDA refinement; independent of MCTS selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Mapping
from uuid import uuid4

from .budget import GenerationBudget
from .compiler_context import prompt_compiler_summary
from .dataflow_prompt import VIEWS, prompt_dataflow_view
from .domain import EvaluationResult, ProposalStatus, Strategy, WorkloadContract
from .generation import GenerationRequest, KernelGenerator
from .interfaces import KernelEvaluator, NodeProfiler
from .llm_generation import _repair_evidence
from .orchestration import SearchTraceStore
from .profiling import (
    LIGHTWEIGHT_V1_ALIASES, LIGHTWEIGHT_V1_METRICS, PROFILE_METRIC_SET_IDS,
)
from .proposals import run_proposal
from .resource_graph import prompt_h100_resource_graph
from .serialization import serialize_benchmark, serialize_evaluation, serialize_generation, serialize_workload


@dataclass(frozen=True, slots=True)
class SequentialConfig:
    max_repairs: int = 1
    max_infrastructure_retries: int = 1
    profile_metric_set: str = "lightweight_v1"
    profile_visibility: str = "full"  # none, lightweight, or full; collection is independent
    compiler_detail: str = "full"  # none, resources, or full
    dataflow_view: str = "paths"
    dataflow_description: str = "limits"  # none or limits
    include_best_so_far: bool = False
    include_hardware_graph: bool = False

    def __post_init__(self) -> None:
        if self.max_repairs < 0 or self.max_infrastructure_retries < 0:
            raise ValueError("retry limits cannot be negative")
        if self.profile_metric_set not in PROFILE_METRIC_SET_IDS:
            raise ValueError("unknown profile metric set")
        if self.profile_visibility not in {"none", "lightweight", "full"}:
            raise ValueError("unknown profile visibility")
        if self.compiler_detail not in {"none", "resources", "full"}:
            raise ValueError("unknown compiler detail")
        if self.dataflow_view not in VIEWS:
            raise ValueError("unknown dataflow view")
        if self.dataflow_description not in {"none", "limits"}:
            raise ValueError("unknown dataflow description")


@dataclass(slots=True)
class SequentialNode:
    id: str
    evaluation: EvaluationResult
    profile: Mapping[str, object] | None = None
    parent_node_id: str | None = None


@dataclass(frozen=True, slots=True)
class SequentialResult:
    root: SequentialNode
    current: SequentialNode
    best: SequentialNode
    nodes: tuple[SequentialNode, ...]
    iterations: int
    generations: int
    profile_calls: int
    accepted_depth: int
    wall_seconds: float


_FREEFORM = Strategy("freeform", "Unconstrained sequential optimization", {"cuda_cpp": ""})


def run_sequential_loop(
    *, root_evaluation: EvaluationResult, workload: WorkloadContract,
    hardware: Mapping[str, object], generator: KernelGenerator,
    evaluator: KernelEvaluator, profiler: NodeProfiler, trace: SearchTraceStore,
    generation_budget: int, config: SequentialConfig = SequentialConfig(), seed: int = 0,
    resource_graph: Mapping[str, object] | None = None,
) -> SequentialResult:
    if root_evaluation.status != ProposalStatus.VALID:
        raise ValueError("sequential root must be valid")
    if root_evaluation.program is None or root_evaluation.benchmark is None:
        raise ValueError("sequential root requires a program and benchmark")
    if config.include_hardware_graph and resource_graph is None:
        raise ValueError("hardware graph context requires a supported resource graph")
    budget = GenerationBudget(generation_budget)
    started = perf_counter()
    root = SequentialNode(str(uuid4()), root_evaluation)
    nodes = [root]
    by_state = {root_evaluation.state_key: root}
    current = best = root
    iterations = profile_calls = accepted_depth = 0
    feedback: Mapping[str, object] | None = None

    def node_payload(node: SequentialNode, *, is_root: bool = False) -> dict[str, object]:
        evaluation = node.evaluation
        assert evaluation.program is not None
        return {
            "node_id": node.id, "state_key": evaluation.state_key,
            "program_text": evaluation.program.source,
            "backend_type": evaluation.program.backend,
            "reward": evaluation.reward,
            "evaluation": serialize_evaluation(evaluation),
            "profile": dict(node.profile) if node.profile is not None else None,
            "is_root": is_root,
            "parent_node_id": node.parent_node_id,
        }

    def profile(node: SequentialNode, iteration: int, trigger: str) -> None:
        nonlocal profile_calls
        profile_calls += 1
        trace.emit("profiling_started", {
            "iteration": iteration, "node_id": node.id,
            "profile_level": "lightweight", "metric_set": config.profile_metric_set,
            "profile_call": profile_calls, "trigger": trigger,
        })
        profile_started = perf_counter()
        node.profile = dict(profiler.lightweight_profile(node.evaluation, workload))
        trace.emit("node_profiled", {
            **node_payload(node), "iteration": iteration,
            "profile_level": "lightweight", "metric_set": config.profile_metric_set,
            "profile_call": profile_calls, "trigger": trigger,
            "profile_wall_seconds": perf_counter() - profile_started,
        })

    trace.emit("run_started", {
        "algorithm": "sequential", "seed": seed, "config": asdict(config),
        "workload": serialize_workload(workload), "hardware": dict(hardware),
        "generation_budget": generation_budget, "mutation_budget": 0,
        "root_node_id": root.id,
    })
    trace.emit("node_created", node_payload(root, is_root=True))
    try:
        profile(root, 0, "initial_context")
        while not budget.exhausted:
            iterations += 1
            evaluation = current.evaluation
            assert evaluation.program is not None
            compiler = None
            if config.compiler_detail != "none":
                compiler = prompt_compiler_summary(evaluation.metadata.get("compiler_context"))
                if config.compiler_detail == "resources":
                    compiler.pop("sass_static_instruction_sites", None)
            dataflow = None
            if config.dataflow_view != "none":
                dataflow = prompt_dataflow_view(trace.get_node_dataflow(current.id), config.dataflow_view)
                if config.dataflow_description == "none":
                    dataflow.pop("limits", None)
                    dataflow.pop("uncertainty", None)
            request = GenerationRequest(
                parent=evaluation.program, strategy=_FREEFORM, workload=workload,
                hardware=hardware,
                profile=_visible_profile(current.profile, config.profile_visibility),
                compiler_summary=compiler, dataflow_view=dataflow,
                current_benchmark=serialize_benchmark(evaluation.benchmark),
                recent_feedback=feedback,
                hardware_resource_graph=(
                    prompt_h100_resource_graph(resource_graph)
                    if config.include_hardware_graph and resource_graph is not None
                    else None
                ),
                best_so_far=(
                    {
                        "source": best.evaluation.program.source,
                        "median_us": best.evaluation.benchmark.median_us,
                    }
                    if config.include_best_so_far and best.id != current.id
                    else None
                ),
            )
            parent = current
            outcome = run_proposal(
                generator=generator, evaluator=evaluator, budget=budget, request=request,
                max_repairs=config.max_repairs,
                max_infrastructure_retries=config.max_infrastructure_retries,
            )
            result = outcome.result
            child = None
            reused = False
            if result.status == ProposalStatus.VALID:
                assert result.state_key is not None
                child = by_state.get(result.state_key)
                reused = child is not None
                if child is None:
                    child = SequentialNode(str(uuid4()), result, parent_node_id=parent.id)
                    nodes.append(child)
                    by_state[result.state_key] = child
            for attempt in outcome.attempts:
                final = attempt is outcome.attempts[-1]
                attempt_result = attempt.evaluation
                serialized = serialize_evaluation(attempt_result)
                choice = attempt.generation.metadata or {}
                trace.emit("generation", {
                    "generation_id": attempt.generation.generation_id,
                    "iteration": iterations, "b_gen": attempt.b_gen,
                    "parent_node_id": parent.id, "strategy_id": "freeform",
                    "repair_attempt": attempt.attempt_number,
                    "proposal_status": attempt_result.status.value,
                    "invalid_reason": serialized["invalid_reason"],
                    "compile_status": serialized["compile_status"],
                    "correctness_status": serialized["correctness_status"],
                    "generation": serialize_generation(attempt.generation),
                    "selected_strategy_id": choice.get("selected_strategy_id"),
                    "strategy_choice_status": choice.get("strategy_choice_status"),
                    "original_selected_strategy_id": choice.get("strategy_choice_original_id"),
                    "evaluation": serialized,
                    "prompt_hash": attempt.generation.prompt_hash,
                    "prompt_text": attempt.generation.prompt_text,
                    "api_instructions": attempt.generation.instructions_text,
                    "raw_output": attempt.generation.raw_output,
                    "input_tokens": attempt.generation.input_tokens,
                    "output_tokens": attempt.generation.output_tokens,
                    "llm_latency_seconds": attempt.generation.latency_seconds,
                    "candidate_program": attempt.generation.program.source if attempt.generation.program else None,
                    "state_key": attempt_result.state_key,
                    "reward": attempt_result.reward,
                    "benchmark": serialized["benchmark"],
                    "worker_id": attempt_result.worker_id,
                    "environment_manifest_id": attempt_result.environment_manifest_id,
                    "created_node_id": child.id if final and child else None,
                    "reused_node": reused if final else False,
                    "proposal_mechanism": "llm_generation",
                })
            if child is not None:
                if reused:
                    trace.emit("node_reused", {
                        "iteration": iterations, "parent_node_id": parent.id,
                        "node_id": child.id, "generation_id": outcome.attempts[-1].generation.generation_id,
                    })
                else:
                    trace.emit("node_created", {
                        **node_payload(child), "iteration": iterations,
                        "generation_id": outcome.attempts[-1].generation.generation_id,
                    })
                    profile(child, iterations, "accepted_candidate")
                current = child
                accepted_depth += 1
                if child.evaluation.reward is not None and best.evaluation.reward is not None and child.evaluation.reward > best.evaluation.reward:
                    best = child
                    trace.emit("new_global_best", {
                        "iteration": iterations, "node_id": child.id,
                        "reward": child.evaluation.reward,
                    })
            feedback = {
                "status": result.status.value,
                "invalid_reason": result.invalid_reason.value if result.invalid_reason else None,
                "verification": _repair_evidence(result) if result.status != ProposalStatus.VALID else None,
                "failed_candidate_source": (
                    outcome.attempts[-1].generation.program.source
                    if child is None and outcome.attempts[-1].generation.program else None
                ),
                "candidate_latency_us": result.benchmark.median_us if result.benchmark else None,
                "current_latency_us": current.evaluation.benchmark.median_us if current.evaluation.benchmark else None,
                "accepted": child is not None,
                "best_latency_us": best.evaluation.benchmark.median_us if best.evaluation.benchmark else None,
            }
            trace.emit("sequential_step", {
                "iteration": iterations, "b_gen": budget.snapshot().used,
                "parent_node_id": parent.id, "current_node_id": current.id,
                "best_node_id": best.id, "accepted_depth": accepted_depth,
                "feedback": feedback,
            })
        result = SequentialResult(root, current, best, tuple(nodes), iterations,
                                  budget.snapshot().used, profile_calls, accepted_depth,
                                  perf_counter() - started)
        trace.emit("run_completed", {
            "iterations": iterations, "b_gen": result.generations,
            "b_mut": 0, "b_prior": 0, "profile_calls": profile_calls,
            "drift_probe_calls": 0, "best_node_id": best.id,
            "best_reward": best.evaluation.reward,
            "current_node_id": current.id, "accepted_depth": accepted_depth,
            "unique_node_count": len(nodes), "wall_seconds": result.wall_seconds,
        })
        return result
    except Exception as error:
        trace.emit("run_failed", {
            "iterations": iterations, "b_gen": budget.snapshot().used,
            "b_mut": 0, "b_prior": 0, "profile_calls": profile_calls,
            "drift_probe_calls": 0, "error_type": type(error).__name__,
            "message": str(error), "wall_seconds": perf_counter() - started,
        })
        raise


def _visible_profile(
    profile: Mapping[str, object] | None, visibility: str,
) -> Mapping[str, object] | None:
    if profile is None or visibility == "none":
        return None
    if visibility == "full":
        return profile
    summary = profile.get("summary")
    metrics = profile.get("metrics")
    return {
        **profile,
        "prompt_visibility": "lightweight_v1_metrics_from_collected_profile",
        "summary": {
            key: value for key, value in summary.items()
            if key in LIGHTWEIGHT_V1_ALIASES.values()
        } if isinstance(summary, Mapping) else {},
        "metrics": {
            key: value for key, value in metrics.items()
            if key in LIGHTWEIGHT_V1_METRICS
        } if isinstance(metrics, Mapping) else {},
        "resolved_metrics": [
            key for key in profile.get("resolved_metrics", [])
            if key in LIGHTWEIGHT_V1_METRICS
        ] if isinstance(profile.get("resolved_metrics"), list) else [],
    }
