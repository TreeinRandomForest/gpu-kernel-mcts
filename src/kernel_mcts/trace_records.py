from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping

from .domain import CompileStatus, CorrectnessStatus, InvalidReason, ProposalStatus


class IterationStatus(StrEnum):
    VALID = "VALID"
    INVALID = "INVALID"
    INFRASTRUCTURE_FAILURE = "INFRASTRUCTURE_FAILURE"
    DEPTH_LIMIT = "DEPTH_LIMIT"
    CYCLE = "CYCLE"


class SelectionMode(StrEnum):
    EXPAND = "EXPAND"
    UCB = "UCB"


@dataclass(frozen=True, slots=True)
class SearchRunRecord:
    run_id: str
    benchmark_id: str
    algorithm: str
    config: Mapping[str, Any]
    started_at: str
    workload: Mapping[str, Any] = field(default_factory=dict)
    hardware: Mapping[str, Any] = field(default_factory=dict)
    toolchain: Mapping[str, Any] = field(default_factory=dict)
    seed: int | None = None
    generation_budget: int | None = None
    environment_manifest_id: str | None = None
    ended_at: str | None = None
    best_node_id: str | None = None
    final_b_gen: int | None = None
    final_b_prior: int | None = None
    final_iterations: int | None = None
    git_commit: str | None = None
    dirty_tree: bool | None = None
    model_name: str | None = None


@dataclass(frozen=True, slots=True)
class GenerationRecord:
    generation_id: str
    run_id: str
    b_gen: int
    parent_node_id: str
    strategy_id: str
    repair_attempt: int
    proposal_status: ProposalStatus
    iteration: int | None = None
    strategy_prior: float | None = None
    parent_visit_count: int | None = None
    parent_action_q_mean: float | None = None
    parent_action_q_max: float | None = None
    invalid_reason: InvalidReason | None = None
    compile_status: CompileStatus = CompileStatus.NOT_ATTEMPTED
    correctness_status: CorrectnessStatus = CorrectnessStatus.NOT_TESTED
    prompt_hash: str | None = None
    prompt_text: str | None = None
    raw_output: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    llm_latency_seconds: float | None = None
    candidate_program: str | None = None
    state_key: str | None = None
    reward: float | None = None
    benchmark: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    worker_id: str | None = None
    environment_manifest_id: str | None = None
    created_node_id: str | None = None
    reused_node: bool = False


@dataclass(frozen=True, slots=True)
class NodeRecord:
    run_id: str
    node_id: str
    state_key: str
    program_text: str
    backend_type: str
    reward: float
    workload: Mapping[str, Any] = field(default_factory=dict)
    hardware: Mapping[str, Any] = field(default_factory=dict)
    benchmark: Mapping[str, Any] | None = None
    profile: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    source_hash: str | None = None
    binary_hash: str | None = None
    launch_config: Mapping[str, Any] = field(default_factory=dict)
    worker_id: str | None = None
    environment_manifest_id: str | None = None


@dataclass(frozen=True, slots=True)
class StrategyEdgeRecord:
    run_id: str
    parent_node_id: str
    strategy_id: str
    prior: float
    visits: int
    value_sum: float
    q_mean: float
    q_max: float | None
    proposal_count: int
    generation_attempt_count: int
    repair_generation_count: int
    valid_proposal_count: int
    invalid_proposal_count: int


@dataclass(frozen=True, slots=True)
class RealizationEdgeRecord:
    run_id: str
    parent_node_id: str
    strategy_id: str
    child_node_id: str
    descents: int
    value_sum: float
    q_mean: float


@dataclass(frozen=True, slots=True)
class IterationRecord:
    run_id: str
    iteration: int
    status: IterationStatus
    expanded_parent_node_id: str | None = None
    selected_strategy_id: str | None = None
    leaf_node_id: str | None = None
    backed_up_reward: float | None = None
    b_gen: int = 0
    b_prior: int = 0


@dataclass(frozen=True, slots=True)
class IterationStepRecord:
    run_id: str
    iteration: int
    step: int
    node_id: str
    strategy_id: str
    selection_mode: SelectionMode
    child_node_id: str | None = None
