from __future__ import annotations

import math
from dataclasses import dataclass
from random import Random
from typing import Mapping, Sequence
from uuid import uuid4

from ..budget import GenerationBudget
from ..domain import EvaluationResult, KernelProgram, ProposalStatus, Strategy, WorkloadContract
from ..generation import GenerationRequest, KernelGenerator
from ..interfaces import EventSink, KernelEvaluator, NodeProfiler, NullEventSink, StrategyPriorProvider
from ..priors import validate_priors
from ..proposals import GenerationAttempt, run_proposal
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

    def __post_init__(self) -> None:
        if self.k_max < 1 or self.max_depth < 1:
            raise ValueError("k_max and max_depth must be positive")
        if self.max_repairs < 0 or self.max_infrastructure_retries < 0:
            raise ValueError("retry limits cannot be negative")
        if self.c_pw <= 0 or not 0 <= self.alpha_pw <= 1:
            raise ValueError("invalid progressive-widening configuration")


@dataclass(frozen=True, slots=True)
class SearchResult:
    root: SearchNode
    best: SearchNode
    nodes: tuple[SearchNode, ...] #all unique nodes
    iterations: int #MCTS iterations i.e. #of backups
    generations: int #LLM calls for gen incl. repair


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
        hardware: Mapping[str, object] | None = None,
        config: MCTSConfig = MCTSConfig(),
        seed: int = 0,
        profiler: NodeProfiler | None = None,
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
        self.hardware = hardware or {}
        self.config = config
        self.seed = seed
        self.rng = Random(seed)
        self.profiler = profiler
        self.events = events or NullEventSink()
        self.nodes = TranspositionTable()

    def run(self, root_evaluation: EvaluationResult) -> SearchResult:
        root = SearchNode(str(uuid4()), root_evaluation, 0)
        self.nodes.add(root)
        best = root
        iterations = 0
        while not self.budget.exhausted:
            leaf = self._iterate(root)
            iterations += 1
            if leaf is not None and leaf.reward > best.reward:
                best = leaf
                self.events.emit("new_global_best", {"node_id": leaf.id, "reward": leaf.reward})
        return SearchResult(root, best, tuple(self.nodes.values()), iterations, self.budget.snapshot().used)

    def _ensure_actions(self, node: SearchNode) -> None:
        if node.actions:
            return
        if self.profiler is not None and node.profile is None:
            node.profile = dict(self.profiler.lightweight_profile(node.evaluation, self.workload))
        priors = validate_priors(
            self.prior_provider.get_priors(node.program, self.workload, tuple(self.strategies.values()), node.profile),
            tuple(self.strategies.values()),
        )
        node.actions = {key: StrategyEdge(key, prior) for key, prior in priors.items()}

    def _iterate(self, root: SearchNode) -> SearchNode | None:
        node = root
        path: list[tuple[StrategyEdge, RealizationEdge | None]] = []
        seen = {node.id}
        while node.depth < self.config.max_depth:
            self._ensure_actions(node)
            action = self._select_action(node)
            # Count selection before widening, matching K_allowed(N) with first visit N=1.
            action.visits += 1
            if len(action.realizations) < self._allowed_children(action.visits):
                leaf = self._expand(node, action)
                if leaf is None:
                    self._backup(path + [(action, None)], node.reward)
                    return None
                realization = action.realizations[leaf.id]
                path.append((action, realization))
                self._backup(path, leaf.reward)
                return leaf
            realization = self._select_realization(action)
            path.append((action, realization))
            child = next(item for item in self.nodes.values() if item.id == realization.child_id)
            if child.id in seen:
                self._backup(path, child.reward)
                return child
            seen.add(child.id)
            node = child
        self._backup(path, node.reward)
        return node

    def _select_action(self, node: SearchNode) -> StrategyEdge:
        total = sum(edge.visits for edge in node.actions.values())
        exploration_scale = math.sqrt(total)
        scored = [
            (
                edge.q_mean
                + self.config.c_puct
                * edge.prior
                * exploration_scale
                / (1 + edge.visits),
                edge,
            )
            for edge in node.actions.values()
        ]
        best_score = max(score for score, _ in scored)
        tied = [edge for score, edge in scored if score == best_score]
        return self.rng.choice(tied)

    def _allowed_children(self, visits: int) -> int:
        #progressive widening budget
        return min(self.config.k_max, math.ceil(self.config.c_pw * visits ** self.config.alpha_pw))

    def _select_realization(self, action: StrategyEdge) -> RealizationEdge:
        return max(
            action.realizations.values(),
            key=lambda edge: edge.q_mean
            + self.config.c_ucb * math.sqrt(math.log1p(action.visits) / (1 + edge.descents)),
        )

    def _expand(self, parent: SearchNode, action: StrategyEdge) -> SearchNode | None:
        action.proposal_count += 1
        outcome = run_proposal(
            generator=self.generator,
            evaluator=self.evaluator,
            budget=self.budget,
            request=GenerationRequest(
                parent=parent.program,
                strategy=self.strategies[action.strategy_id],
                workload=self.workload,
                hardware=self.hardware,
                profile=parent.profile,
            ),
            max_repairs=self.config.max_repairs,
            max_infrastructure_retries=self.config.max_infrastructure_retries,
        )
        action.generation_attempt_count += len(outcome.attempts)
        action.repair_generation_count += max(0, len(outcome.attempts) - 1)
        for attempt in outcome.attempts:
            self.events.emit("generation", self._generation_payload(parent, action, attempt))
        result = outcome.result
        if result.status != ProposalStatus.VALID:
            if result.status == ProposalStatus.INVALID:
                action.invalid_proposal_count += 1
            return None
        action.valid_proposal_count += 1
        assert result.program is not None and result.state_key is not None and result.reward is not None
        candidate = SearchNode(str(uuid4()), result, parent.depth + 1)
        child = self.nodes.add(candidate)
        action.realizations.setdefault(child.id, RealizationEdge(child.id))
        return child

    def _backup(self, path: list[tuple[StrategyEdge, RealizationEdge | None]], reward: float) -> None:
        for action, realization in path:
            # The action visit was already counted during selection.
            action.value_sum += reward
            action.q_max = max(action.q_max, reward)
            if realization is not None:
                realization.descents += 1
                realization.value_sum += reward

    @staticmethod
    def _generation_payload(
        parent: SearchNode,
        action: StrategyEdge,
        attempt: GenerationAttempt,
    ) -> Mapping[str, object]:
        result = attempt.evaluation
        return {
            "generation_id": attempt.generation.generation_id,
            "b_gen": attempt.budget_index,
            "repair_attempt": attempt.attempt_number,
            "parent_node_id": parent.id,
            "strategy_id": action.strategy_id,
            "proposal_status": result.status.value,
            "invalid_reason": result.invalid_reason.value if result.invalid_reason is not None else None,
            "state_key": result.state_key,
            "reward": result.reward,
        }
