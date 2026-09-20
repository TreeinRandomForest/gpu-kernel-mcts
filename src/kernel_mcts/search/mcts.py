from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from random import Random
from typing import Mapping, Sequence
from uuid import uuid4

from ..budget import GenerationBudget, MutationBudget
from ..domain import EvaluationResult, ProposalStatus, Strategy, WorkloadContract
from ..generation import (
    GenerationRequest,
    KernelGenerator,
    ProposalBudgetKind,
    select_proposal_mechanism,
)
from ..interfaces import EventSink, KernelEvaluator, MeasurementDriftMonitor, NodeProfiler, NullEventSink, StrategyPriorProvider
from ..priors import validate_priors
from ..profiling import PROFILE_METRIC_SET_IDS
from ..proposals import GenerationAttempt, run_proposal
from ..serialization import (
    serialize_evaluation,
    serialize_generation,
    serialize_profile,
    serialize_workload,
)
from ..trace_records import IterationStatus, SelectionMode
from .model import RealizationEdge, SearchNode, StrategyEdge, TranspositionTable


@dataclass(frozen=True, slots=True)
class MCTSConfig:
    c_puct: float = 1.5
    c_ucb: float = 1.0
    c_pw: float = 1.0
    alpha_pw: float = 0.5
    k_max: int = 4
    max_depth: int = 10
    max_repairs: int = 2
    max_infrastructure_retries: int = 1
    include_incoming_profile_delta: bool = False
    profile_metric_set: str = "lightweight_v1"
    measurement_drift_interval: int = 0
    measurement_drift_threshold: float = 0.05

    def __post_init__(self) -> None:
        if self.k_max < 1 or self.max_depth < 1:
            raise ValueError("k_max and max_depth must be positive")
        if self.max_repairs < 0 or self.max_infrastructure_retries < 0:
            raise ValueError("retry limits cannot be negative")
        if self.c_pw <= 0 or not 0 <= self.alpha_pw <= 1:
            raise ValueError("invalid progressive-widening configuration")
        if self.profile_metric_set not in PROFILE_METRIC_SET_IDS:
            raise ValueError("unknown profile metric set")
        if self.measurement_drift_interval < 0 or self.measurement_drift_threshold < 0:
            raise ValueError("measurement drift settings cannot be negative")


@dataclass(frozen=True, slots=True)
class SearchResult:
    root: SearchNode
    best: SearchNode
    nodes: tuple[SearchNode, ...] #all unique nodes
    iterations: int #select -> expand/evaluate -> optional backup cycles
    generations: int #LLM calls for gen incl. repair
    mutations: int #deterministic typed mutation proposals
    prior_calls: int #B_prior: LLM calls used to obtain strategy priors
    profile_calls: int #profiler executions, separate from B_gen and B_prior
    drift_probe_calls: int = 0


@dataclass(frozen=True, slots=True)
class ExpansionOutcome:
    status: ProposalStatus
    node: SearchNode | None = None


@dataclass(frozen=True, slots=True)
class SelectionStep:
    node_id: str
    strategy_id: str
    selection_mode: SelectionMode
    child_node_id: str | None = None
    total_action_visits: int = 0
    puct_candidates: tuple[Mapping[str, object], ...] = ()
    existing_children: int = 0
    allowed_children: int = 0
    ucb_candidates: tuple[Mapping[str, object], ...] = ()


@dataclass(frozen=True, slots=True)
class IterationOutcome:
    status: IterationStatus
    steps: tuple[SelectionStep, ...]
    leaf: SearchNode | None = None
    expanded_parent_node_id: str | None = None
    selected_strategy_id: str | None = None
    backed_up_reward: float | None = None


@dataclass(frozen=True, slots=True)
class SelectedEdge:
    parent_node_id: str
    strategy: StrategyEdge
    realization: RealizationEdge


class _ProposalSpaceExhausted(RuntimeError):
    pass


class MCTS:
    def __init__(
        self,
        *,
        strategies: Sequence[Strategy],
        workload: WorkloadContract,
        generator: KernelGenerator,
        evaluator: KernelEvaluator,
        prior_provider: StrategyPriorProvider,
        budget: GenerationBudget,
        mutation_budget: MutationBudget | None = None,
        hardware: Mapping[str, object] | None = None,
        config: MCTSConfig = MCTSConfig(),
        seed: int = 0,
        profiler: NodeProfiler | None = None,
        drift_monitor: MeasurementDriftMonitor | None = None,
        events: EventSink | None = None,
    ) -> None:
        if not strategies:
            raise ValueError("at least one strategy is required")
        self.strategies = {strategy.id: strategy for strategy in strategies}
        self.workload = workload
        self.generator = generator
        self.evaluator = evaluator
        self.prior_provider = prior_provider
        self.budget = budget
        self.mutation_budget = mutation_budget or MutationBudget(0)
        self.hardware = hardware or {}
        self.config = config
        self.seed = seed
        self.rng = Random(seed)
        self.profiler = profiler
        self.drift_monitor = drift_monitor
        self.events = events or NullEventSink()
        self.nodes = TranspositionTable()
        self.prior_calls = 0
        self.profile_calls = 0
        self.drift_probe_calls = 0
        self._last_drift_node_count = 1
        self._exhausted_nodes: set[str] = set()

    def run(self, root_evaluation: EvaluationResult) -> SearchResult:
        root = SearchNode(str(uuid4()), root_evaluation)
        self.nodes.add(root)
        best = root
        iterations = 0
        self.events.emit(
            "run_started",
            {
                "algorithm": "mcts",
                "seed": self.seed,
                "config": asdict(self.config),
                "workload": serialize_workload(self.workload),
                "hardware": dict(self.hardware),
                "generation_budget": self.budget.snapshot().limit,
                "mutation_budget": self.mutation_budget.snapshot().limit,
                "root_node_id": root.id,
            },
        )
        self.events.emit("node_created", self._node_payload(root, is_root=True))
        try:
            while not self._all_proposal_budgets_exhausted():
                iterations += 1
                outcome = self._iterate(root, iterations)
                self.events.emit(
                    "iteration_completed",
                    self._iteration_payload(iterations, outcome),
                )
                leaf = outcome.leaf
                if leaf is not None and leaf.reward > best.reward:
                    best = leaf
                    self.events.emit(
                        "new_global_best",
                        {"iteration": iterations, "node_id": leaf.id, "reward": leaf.reward},
                    )
                self._maybe_probe_measurement_drift(root, best, iterations)
                if outcome.status == IterationStatus.PROPOSAL_SPACE_EXHAUSTED:
                    break
            if self.profiler is not None and best.profile is None:
                self._profile_node(best, iterations, "final_best")
        except Exception as error:
            self.events.emit(
                "run_failed",
                {
                    "iterations": iterations,
                    "b_gen": self.budget.snapshot().used,
                    "b_mut": self.mutation_budget.snapshot().used,
                    "b_prior": self.prior_calls,
                    "profile_calls": self.profile_calls,
                    "drift_probe_calls": self.drift_probe_calls,
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
            )
            raise
        result = SearchResult(
            root,
            best,
            tuple(self.nodes.values()),
            iterations,
            self.budget.snapshot().used,
            self.mutation_budget.snapshot().used,
            self.prior_calls,
            self.profile_calls,
            self.drift_probe_calls,
        )
        self._emit_final_snapshots(result)
        self.events.emit(
            "run_completed",
            {
                "iterations": result.iterations,
                "b_gen": result.generations,
                "b_mut": result.mutations,
                "b_prior": result.prior_calls,
                "profile_calls": result.profile_calls,
                "drift_probe_calls": result.drift_probe_calls,
                "best_node_id": result.best.id,
                "best_reward": result.best.reward,
                "unique_node_count": len(result.nodes),
            },
        )
        return result

    def _all_proposal_budgets_exhausted(self) -> bool:
        return self.budget.exhausted and self.mutation_budget.exhausted

    def _maybe_probe_measurement_drift(
        self, root: SearchNode, best: SearchNode, iteration: int
    ) -> None:
        interval = self.config.measurement_drift_interval
        if self.drift_monitor is None or interval == 0:
            return
        node_count = len(tuple(self.nodes.values()))
        if node_count - self._last_drift_node_count < interval:
            return
        self._last_drift_node_count = node_count
        for role, node in (("root", root), ("best", best)):
            if role == "best" and best is root:
                continue
            self.drift_probe_calls += 1
            probe = self.drift_monitor.remeasure(node.evaluation, self.workload)
            original = node.evaluation.benchmark
            measured = probe.benchmark
            ratio = None
            if original is not None and measured is not None:
                ratio = measured.median_us / original.median_us
            self.events.emit(
                "measurement_drift_probe",
                {
                    "iteration": iteration,
                    "role": role,
                    "node_id": node.id,
                    "probe_call": self.drift_probe_calls,
                    "status": probe.status.value,
                    "original_benchmark": serialize_evaluation(node.evaluation).get("benchmark"),
                    "probe_benchmark": serialize_evaluation(probe).get("benchmark"),
                    "latency_ratio": ratio,
                    "threshold": self.config.measurement_drift_threshold,
                    "suspect": ratio is not None and abs(ratio - 1.0) > self.config.measurement_drift_threshold,
                },
            )

    def _emit_final_snapshots(self, result: SearchResult) -> None:
        for node in result.nodes:
            self.events.emit(
                "node_snapshot",
                self._node_payload(node, is_root=node is result.root),
            )
            for action in node.actions.values():
                strategy_payload = self._strategy_edge_payload(node.id, action)
                self.events.emit("strategy_edge_snapshot", strategy_payload)
                for realization in action.realizations.values():
                    self.events.emit(
                        "realization_edge_snapshot",
                        {
                            "parent_node_id": node.id,
                            "strategy_id": action.strategy_id,
                            "child_node_id": realization.child_id,
                            "realization": {
                                "descents": realization.descents,
                                "value_sum": realization.value_sum,
                                "q_mean": realization.q_mean,
                            },
                        },
                    )

    def _ensure_actions(self, node: SearchNode, iteration: int | None = None) -> None:
        if node.actions:
            return
        if self.profiler is not None and node.profile is None:
            self._profile_node(node, iteration, "expansion")
        if self.prior_provider.counts_toward_b_prior:
            self.prior_calls += 1
        priors = validate_priors(
            self.prior_provider.get_priors(node.program, self.workload, tuple(self.strategies.values()), node.profile),
            tuple(self.strategies.values()),
        )
        node.actions = {key: StrategyEdge(key, prior) for key, prior in priors.items()}
        self.events.emit(
            "strategy_priors",
            {
                "iteration": iteration,
                "node_id": node.id,
                "provider": self.prior_provider.name,
                "counts_toward_b_prior": self.prior_provider.counts_toward_b_prior,
                "b_prior": self.prior_calls,
                "priors": priors,
            },
        )

    def _iterate(self, root: SearchNode, iteration: int | None = None) -> IterationOutcome:
        """PUCT selection
        """

        node = root
        traversal_depth = 0
        path: list[SelectedEdge] = []
        steps: list[SelectionStep] = []
        seen = {node.id}
        while traversal_depth < self.config.max_depth:
            self._ensure_actions(node, iteration)
            try:
                action, puct_candidates, total_action_visits = (
                    self._select_action_with_scores(node, respect_availability=True)
                )
            except _ProposalSpaceExhausted:
                self._exhausted_nodes.add(node.id)
                if node is not root:
                    self._backup(path, node.reward, iteration)
                return IterationOutcome(
                    status=(
                        IterationStatus.PROPOSAL_SPACE_EXHAUSTED
                        if node is root
                        else IterationStatus.DEAD_END
                    ),
                    steps=tuple(steps),
                    leaf=node if node is not root else None,
                    backed_up_reward=node.reward if node is not root else None,
                )
            # Use the prospective valid visit so the first selection allows one child.
            prospective_visits = action.visits + 1
            allowed_children = self._allowed_children(prospective_visits)
            existing_children = len(action.realizations)
            if (
                existing_children < allowed_children
                and self._can_generate_new_realization(node, action)
            ):  # progressive widening
                outcome = self._expand(
                    node,
                    action,
                    iteration,
                    incoming_edge=path[-1] if path else None,
                )
                steps.append(
                    SelectionStep(
                        node.id,
                        action.strategy_id,
                        SelectionMode.EXPAND,
                        outcome.node.id if outcome.node is not None else None,
                        total_action_visits,
                        puct_candidates,
                        existing_children,
                        allowed_children,
                    )
                )
                if outcome.status != ProposalStatus.VALID:
                    return IterationOutcome(
                        status=IterationStatus(outcome.status.value),
                        steps=tuple(steps),
                        expanded_parent_node_id=node.id,
                        selected_strategy_id=action.strategy_id,
                    )
                assert outcome.node is not None
                leaf = outcome.node
                realization = action.realizations[leaf.id]
                path.append(SelectedEdge(node.id, action, realization))
                self._backup(path, leaf.reward, iteration)
                return IterationOutcome(
                    status=IterationStatus.VALID,
                    steps=tuple(steps),
                    leaf=leaf,
                    expanded_parent_node_id=node.id,
                    selected_strategy_id=action.strategy_id,
                    backed_up_reward=leaf.reward,
                )
            can_generate_new = self._can_generate_new_realization(node, action)
            realization, ucb_candidates = self._select_realization_with_scores(
                action,
                excluded_children=(
                    None if can_generate_new else self._exhausted_nodes
                ),
            )
            path.append(SelectedEdge(node.id, action, realization))
            traversal_depth += 1
            child = next(item for item in self.nodes.values() if item.id == realization.child_id)
            steps.append(
                SelectionStep(
                    node.id,
                    action.strategy_id,
                    SelectionMode.UCB,
                    child.id,
                    total_action_visits,
                    puct_candidates,
                    existing_children,
                    allowed_children,
                    ucb_candidates,
                )
            )

            if child.id in seen:
                self._backup(path, child.reward, iteration)
                return IterationOutcome(
                    status=IterationStatus.CYCLE,
                    steps=tuple(steps),
                    leaf=child,
                    backed_up_reward=child.reward,
                )
            seen.add(child.id)
            node = child
        self._backup(path, node.reward, iteration)
        return IterationOutcome(
            status=IterationStatus.DEPTH_LIMIT,
            steps=tuple(steps),
            leaf=node,
            backed_up_reward=node.reward,
        )

    def _profile_node(
        self,
        node: SearchNode,
        iteration: int | None,
        trigger: str,
    ) -> None:
        assert self.profiler is not None
        self.profile_calls += 1
        profile_call = self.profile_calls
        self.events.emit(
            "profiling_started",
            {
                "iteration": iteration,
                "node_id": node.id,
                "profile_level": "lightweight",
                "metric_set": self.config.profile_metric_set,
                "profile_call": profile_call,
                "trigger": trigger,
            },
        )
        node.profile = dict(
            self.profiler.lightweight_profile(node.evaluation, self.workload)
        )
        self.events.emit(
            "node_profiled",
            {
                **self._node_payload(node, is_root=False),
                "iteration": iteration,
                "profile_level": "lightweight",
                "metric_set": self.config.profile_metric_set,
                "profile_call": profile_call,
                "trigger": trigger,
            },
        )

    def _select_action(self, node: SearchNode) -> StrategyEdge:
        selected, _, _ = self._select_action_with_scores(node)
        return selected

    def _select_action_with_scores(
        self, node: SearchNode, *, respect_availability: bool = False
    ) -> tuple[StrategyEdge, tuple[Mapping[str, object], ...], int]:
        total = sum(edge.visits for edge in node.actions.values())
        exploration_scale = math.sqrt(total)
        scored: list[tuple[float, StrategyEdge, float]] = []
        for edge in node.actions.values():
            if respect_availability and not self._action_available(node, edge):
                continue
            explore = (
                self.config.c_puct
                * edge.prior
                * exploration_scale
                / (1 + edge.visits)
            )
            scored.append((edge.q_mean + explore, edge, explore))
        if not scored:
            raise _ProposalSpaceExhausted
        best_score = max(score for score, _, _ in scored)
        tied = [edge for score, edge, _ in scored if score == best_score]
        selected = self.rng.choice(tied)
        candidates = tuple(
            {
                "strategy_id": edge.strategy_id,
                "prior": edge.prior,
                "visits": edge.visits,
                "q_mean": edge.q_mean,
                "q_max": edge.q_max if math.isfinite(edge.q_max) else None,
                "exploit_term": edge.q_mean,
                "explore_term": explore,
                "total_score": score,
                "selected": edge is selected,
            }
            for score, edge, explore in scored
        )
        return selected, candidates, total

    def _proposal_request(
        self,
        parent: SearchNode,
        action: StrategyEdge,
        incoming_edge: SelectedEdge | None = None,
    ) -> GenerationRequest:
        return GenerationRequest(
            parent=parent.program,
            strategy=self.strategies[action.strategy_id],
            workload=self.workload,
            hardware=self.hardware,
            profile=parent.profile,
            incoming_profile_delta=self._incoming_profile_delta(parent, incoming_edge),
        )

    def _can_generate_new_realization(
        self, parent: SearchNode, action: StrategyEdge
    ) -> bool:
        request = self._proposal_request(parent, action)
        mechanism, _ = select_proposal_mechanism(
            self.generator,
            request,
            generation_budget_available=not self.budget.exhausted,
            mutation_budget_available=not self.mutation_budget.exhausted,
        )
        return mechanism is not None

    def _action_available(self, parent: SearchNode, action: StrategyEdge) -> bool:
        if self._can_generate_new_realization(parent, action):
            return True
        return any(
            child_id not in self._exhausted_nodes
            for child_id in action.realizations
        )

    def _allowed_children(self, visits: int) -> int:
        #progressive widening budget
        return min(self.config.k_max, math.ceil(self.config.c_pw * visits ** self.config.alpha_pw))

    def _select_realization(self, action: StrategyEdge) -> RealizationEdge:
        selected, _ = self._select_realization_with_scores(action)
        return selected

    def _select_realization_with_scores(
        self,
        action: StrategyEdge,
        *,
        excluded_children: set[str] | None = None,
    ) -> tuple[RealizationEdge, tuple[Mapping[str, object], ...]]:
        scored: list[tuple[float, RealizationEdge, float]] = []
        for edge in action.realizations.values():
            if excluded_children is not None and edge.child_id in excluded_children:
                continue
            explore = self.config.c_ucb * math.sqrt(
                math.log1p(action.visits) / (1 + edge.descents)
            )
            scored.append((edge.q_mean + explore, edge, explore))
        selected = max(scored, key=lambda item: item[0])[1]
        candidates = tuple(
            {
                "child_node_id": edge.child_id,
                "descents": edge.descents,
                "q_mean": edge.q_mean,
                "exploit_term": edge.q_mean,
                "explore_term": explore,
                "total_score": score,
                "selected": edge is selected,
            }
            for score, edge, explore in scored
        )
        return selected, candidates

    def _expand(
        self,
        parent: SearchNode,
        action: StrategyEdge,
        iteration: int | None = None,
        incoming_edge: SelectedEdge | None = None,
    ) -> ExpansionOutcome:
        action.proposal_count += 1
        outcome = run_proposal(
            generator=self.generator,
            evaluator=self.evaluator,
            budget=self.budget,
            mutation_budget=self.mutation_budget,
            request=self._proposal_request(parent, action, incoming_edge),
            max_repairs=self.config.max_repairs,
            max_infrastructure_retries=self.config.max_infrastructure_retries,
        )
        generation_attempts = sum(
            attempt.budget_kind == ProposalBudgetKind.GENERATION
            for attempt in outcome.attempts
        )
        mutation_attempts = len(outcome.attempts) - generation_attempts
        action.generation_attempt_count += generation_attempts
        action.mutation_attempt_count += mutation_attempts
        action.repair_generation_count += max(0, generation_attempts - 1)
        result = outcome.result
        child: SearchNode | None = None
        reused_node = False
        if result.status != ProposalStatus.VALID:
            if result.status == ProposalStatus.INVALID:
                action.invalid_proposal_count += 1
        else:
            action.valid_proposal_count += 1
            assert result.program is not None and result.state_key is not None and result.reward is not None
            candidate = SearchNode(str(uuid4()), result)
            child = self.nodes.add(candidate)
            reused_node = child is not candidate
            action.realizations.setdefault(child.id, RealizationEdge(child.id))

        final_attempt = outcome.attempts[-1]
        for attempt in outcome.attempts:
            associated_child = child if attempt is final_attempt else None
            self.events.emit(
                "generation",
                self._generation_payload(
                    parent,
                    action,
                    attempt,
                    iteration,
                    associated_child,
                    reused_node if associated_child is not None else False,
                ),
            )
        if child is None:
            return ExpansionOutcome(result.status)

        relationship = {
            "iteration": iteration,
            "parent_node_id": parent.id,
            "strategy_id": action.strategy_id,
            "generation_id": final_attempt.generation.generation_id,
            "node_id": child.id,
        }
        if reused_node:
            self.events.emit("node_reused", relationship)
        else:
            self.events.emit(
                "node_created",
                {**self._node_payload(child), **relationship},
            )
        return ExpansionOutcome(ProposalStatus.VALID, child)

    def _incoming_profile_delta(
        self,
        node: SearchNode,
        incoming_edge: SelectedEdge | None,
    ) -> Mapping[str, object] | None:
        """Describe the active incoming edge for an opt-in path-dependent ablation."""
        if not self.config.include_incoming_profile_delta or incoming_edge is None:
            return None
        predecessor = next(
            item
            for item in self.nodes.values()
            if item.id == incoming_edge.parent_node_id
        )
        if predecessor.profile is None or node.profile is None:
            return None
        previous_summary = predecessor.profile.get("summary")
        current_summary = node.profile.get("summary")
        summary_delta: dict[str, float] = {}
        if isinstance(previous_summary, Mapping) and isinstance(
            current_summary, Mapping
        ):
            for key in sorted(previous_summary.keys() & current_summary.keys()):
                previous = previous_summary[key]
                current = current_summary[key]
                if (
                    isinstance(previous, (int, float))
                    and not isinstance(previous, bool)
                    and isinstance(current, (int, float))
                    and not isinstance(current, bool)
                ):
                    summary_delta[str(key)] = float(current) - float(previous)
        reward_delta = node.reward - predecessor.reward
        return {
            "basis": "selected_incoming_edge",
            "from_node_id": predecessor.id,
            "to_node_id": node.id,
            "strategy_id": incoming_edge.strategy.strategy_id,
            "reward_delta": reward_delta,
            "speedup_ratio": math.exp(reward_delta),
            "summary_delta": summary_delta,
        }

    def _backup(self, path: list[SelectedEdge], reward: float, iteration: int | None = None) -> None:
        for selected in path:
            action = selected.strategy
            realization = selected.realization
            action.visits += 1
            action.value_sum += reward
            action.q_max = max(action.q_max, reward)
            realization.descents += 1
            realization.value_sum += reward
            self.events.emit(
                "backup",
                {
                    "iteration": iteration,
                    "parent_node_id": selected.parent_node_id,
                    "strategy_id": action.strategy_id,
                    "child_node_id": realization.child_id,
                    "backed_up_reward": reward,
                    "strategy": self._strategy_edge_payload(
                        selected.parent_node_id, action
                    )["strategy"],
                    "realization": {
                        "descents": realization.descents,
                        "value_sum": realization.value_sum,
                        "q_mean": realization.q_mean,
                    },
                },
            )

    @staticmethod
    def _strategy_edge_payload(
        parent_node_id: str,
        action: StrategyEdge,
    ) -> Mapping[str, object]:
        return {
            "parent_node_id": parent_node_id,
            "strategy_id": action.strategy_id,
            "strategy": {
                "prior": action.prior,
                "visits": action.visits,
                "value_sum": action.value_sum,
                "q_mean": action.q_mean,
                "q_max": action.q_max if math.isfinite(action.q_max) else None,
                "proposal_count": action.proposal_count,
            "generation_attempt_count": action.generation_attempt_count,
            "mutation_attempt_count": action.mutation_attempt_count,
                "repair_generation_count": action.repair_generation_count,
                "valid_proposal_count": action.valid_proposal_count,
                "invalid_proposal_count": action.invalid_proposal_count,
            },
        }

    @staticmethod
    def _generation_payload(
        parent: SearchNode,
        action: StrategyEdge,
        attempt: GenerationAttempt,
        iteration: int | None,
        child: SearchNode | None,
        reused_node: bool,
    ) -> Mapping[str, object]:
        result = attempt.evaluation
        serialized_evaluation = serialize_evaluation(result)
        evaluation_metadata = result.metadata
        generation_metadata = attempt.generation.metadata or {}
        return {
            "iteration": iteration,
            "generation_id": attempt.generation.generation_id,
            "b_gen": attempt.b_gen,
            "b_mut": attempt.b_mut,
            "repair_attempt": attempt.attempt_number,
            "parent_node_id": parent.id,
            "strategy_id": action.strategy_id,
            "strategy_prior": action.prior,
            "parent_visit_count": action.visits,
            "parent_action_q_mean": action.q_mean,
            "parent_action_q_max": action.q_max if math.isfinite(action.q_max) else None,
            "proposal_status": result.status.value,
            "invalid_reason": result.invalid_reason.value if result.invalid_reason is not None else None,
            "compile_status": result.compile_status.value,
            "correctness_status": result.correctness_status.value,
            "state_key": result.state_key,
            "reward": result.reward,
            "prompt_hash": attempt.generation.prompt_hash,
            "prompt_text": attempt.generation.prompt_text,
            "api_instructions": attempt.generation.instructions_text,
            "raw_output": attempt.generation.raw_output,
            "input_tokens": attempt.generation.input_tokens,
            "output_tokens": attempt.generation.output_tokens,
            "llm_latency_seconds": attempt.generation.latency_seconds,
            "candidate_program": (
                attempt.generation.program.source
                if attempt.generation.program is not None
                else None
            ),
            "benchmark": serialized_evaluation["benchmark"],
            "worker_id": result.worker_id,
            "environment_manifest_id": result.environment_manifest_id,
            "generation": serialize_generation(attempt.generation),
            "evaluation": serialized_evaluation,
            "created_node_id": child.id if child is not None else None,
            "reused_node": reused_node,
            "proposal_mechanism": generation_metadata.get(
                "proposal_mechanism", attempt.budget_kind.value
            ),
            "proposal_mechanism_candidates": [
                kind.value for kind in attempt.mechanism_candidates
            ],
            "representation": evaluation_metadata.get(
                "representation", generation_metadata.get("representation")
            ),
            "representation_schema_version": evaluation_metadata.get(
                "representation_schema_version"
            ),
            "configuration_hash": evaluation_metadata.get(
                "configuration_hash", generation_metadata.get("configuration_hash")
            ),
            "static_validation": generation_metadata.get("static_validation"),
            "transformation": generation_metadata.get("transformation"),
        }

    def _node_payload(self, node: SearchNode, *, is_root: bool = False) -> Mapping[str, object]:
        metadata = node.evaluation.metadata
        return {
            "node_id": node.id,
            "state_key": node.state_key,
            "program_text": node.program.source,
            "backend_type": node.program.backend,
            "reward": node.reward,
            "evaluation": serialize_evaluation(node.evaluation),
            "profile": serialize_profile(node.profile),
            "is_root": is_root,
            "representation": metadata.get("representation"),
            "representation_schema_version": metadata.get(
                "representation_schema_version"
            ),
            "configuration_hash": metadata.get("configuration_hash"),
        }

    def _iteration_payload(
        self,
        iteration: int,
        outcome: IterationOutcome,
    ) -> Mapping[str, object]:
        return {
            "iteration": iteration,
            "status": outcome.status.value,
            "expanded_parent_node_id": outcome.expanded_parent_node_id,
            "selected_strategy_id": outcome.selected_strategy_id,
            "leaf_node_id": outcome.leaf.id if outcome.leaf is not None else None,
            "backed_up_reward": outcome.backed_up_reward,
            "b_gen": self.budget.snapshot().used,
            "b_mut": self.mutation_budget.snapshot().used,
            "b_prior": self.prior_calls,
            "steps": [
                {
                    "step": index,
                    "node_id": step.node_id,
                    "strategy_id": step.strategy_id,
                    "selection_mode": step.selection_mode.value,
                    "child_node_id": step.child_node_id,
                    "total_action_visits": step.total_action_visits,
                    "c_puct": self.config.c_puct,
                    "c_ucb": self.config.c_ucb,
                    "c_pw": self.config.c_pw,
                    "alpha_pw": self.config.alpha_pw,
                    "k_max": self.config.k_max,
                    "existing_children": step.existing_children,
                    "allowed_children": step.allowed_children,
                    "puct_candidates": list(step.puct_candidates),
                    "ucb_candidates": list(step.ucb_candidates),
                }
                for index, step in enumerate(outcome.steps)
            ],
        }
