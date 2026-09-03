from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Mapping, Sequence

from .budget import GenerationBudget
from .domain import KernelProgram, ProposalStatus, Strategy, WorkloadContract
from .generation import GenerationRequest, KernelGenerator
from .interfaces import KernelEvaluator
from .proposals import run_proposal


@dataclass(frozen=True, slots=True)
class BaselineResult:
    best_program: KernelProgram
    best_reward: float
    generations: int


def independent_best_of_n(
    root: KernelProgram,
    strategies: Sequence[Strategy],
    workload: WorkloadContract,
    generator: KernelGenerator,
    evaluator: KernelEvaluator,
    budget: GenerationBudget,
    hardware: Mapping[str, object] | None = None,
    max_repairs: int = 2,
    max_infrastructure_retries: int = 1,
    seed: int = 0,
) -> BaselineResult:
    rng = Random(seed)
    best_program, best_reward = root, 0.0
    while not budget.exhausted:
        result = _propose(
            root,
            rng.choice(strategies),
            workload,
            generator,
            evaluator,
            budget,
            hardware,
            max_repairs,
            max_infrastructure_retries,
        )
        if result.status == ProposalStatus.VALID and result.reward is not None and result.reward > best_reward:
            best_program, best_reward = result.program or root, result.reward
    return BaselineResult(best_program, best_reward, budget.snapshot().used)


def greedy_search(
    root: KernelProgram,
    strategies: Sequence[Strategy],
    workload: WorkloadContract,
    generator: KernelGenerator,
    evaluator: KernelEvaluator,
    budget: GenerationBudget,
    hardware: Mapping[str, object] | None = None,
    max_repairs: int = 2,
    max_infrastructure_retries: int = 1,
    seed: int = 0,
) -> BaselineResult:
    rng = Random(seed)
    current, reward = root, 0.0
    while not budget.exhausted:
        result = _propose(
            current,
            rng.choice(strategies),
            workload,
            generator,
            evaluator,
            budget,
            hardware,
            max_repairs,
            max_infrastructure_retries,
        )
        if result.status == ProposalStatus.VALID and result.reward is not None and result.reward > reward:
            current, reward = result.program or current, result.reward
    return BaselineResult(current, reward, budget.snapshot().used)


def iterative_best_of_k(
    root: KernelProgram,
    strategies: Sequence[Strategy],
    workload: WorkloadContract,
    generator: KernelGenerator,
    evaluator: KernelEvaluator,
    budget: GenerationBudget,
    k: int = 4,
    hardware: Mapping[str, object] | None = None,
    max_repairs: int = 2,
    max_infrastructure_retries: int = 1,
    seed: int = 0,
) -> BaselineResult:
    if k < 1:
        raise ValueError("k must be positive")
    rng = Random(seed)
    current, best_program, best_reward = root, root, 0.0
    while not budget.exhausted:
        round_results = []
        for _ in range(k):
            if budget.exhausted:
                break
            result = _propose(
                current,
                rng.choice(strategies),
                workload,
                generator,
                evaluator,
                budget,
                hardware,
                max_repairs,
                max_infrastructure_retries,
            )
            if result.status == ProposalStatus.VALID and result.reward is not None:
                round_results.append(result)
        if round_results:
            winner = max(round_results, key=lambda item: item.reward or float("-inf"))
            # Best-of-K advances to the round winner even if it is locally slower.
            current = winner.program or current
            if winner.reward is not None and winner.reward > best_reward:
                best_program, best_reward = current, winner.reward
    return BaselineResult(best_program, best_reward, budget.snapshot().used)


def _propose(
    parent: KernelProgram,
    strategy: Strategy,
    workload: WorkloadContract,
    generator: KernelGenerator,
    evaluator: KernelEvaluator,
    budget: GenerationBudget,
    hardware: Mapping[str, object] | None,
    max_repairs: int,
    max_infrastructure_retries: int,
):
    return run_proposal(
        generator=generator,
        evaluator=evaluator,
        budget=budget,
        request=GenerationRequest(
            parent=parent,
            strategy=strategy,
            workload=workload,
            hardware=hardware or {},
            profile=None,
        ),
        max_repairs=max_repairs,
        max_infrastructure_retries=max_infrastructure_retries,
    ).result
