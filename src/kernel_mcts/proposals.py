from __future__ import annotations

from dataclasses import dataclass

from .budget import GenerationBudget, MutationBudget
from .domain import EvaluationResult, InvalidReason, KernelProgram, ProposalStatus, WorkloadContract
from .generation import (
    GenerationRequest,
    GenerationResult,
    KernelGenerator,
    ProposalBudgetKind,
    proposal_budget_kind,
)
from .interfaces import KernelEvaluator


@dataclass(frozen=True, slots=True)
class GenerationAttempt:
    b_gen: int
    b_mut: int
    budget_kind: ProposalBudgetKind
    attempt_number: int
    generation: GenerationResult
    evaluation: EvaluationResult

    @property
    def budget_index(self) -> int:
        return self.b_mut if self.budget_kind == ProposalBudgetKind.MUTATION else self.b_gen


@dataclass(frozen=True, slots=True)
class ProposalOutcome:
    attempts: tuple[GenerationAttempt, ...]
    result: EvaluationResult


def run_proposal(
    *,
    generator: KernelGenerator,
    evaluator: KernelEvaluator,
    budget: GenerationBudget,
    mutation_budget: MutationBudget | None = None,
    request: GenerationRequest,
    max_repairs: int,
    max_infrastructure_retries: int,
) -> ProposalOutcome:
    """Generate and evaluate one logical proposal with explicit mechanism accounting."""
    if max_repairs < 0 or max_infrastructure_retries < 0:
        raise ValueError("retry limits cannot be negative")
    attempts: list[GenerationAttempt] = []
    current_request = request
    kind = proposal_budget_kind(generator, request)
    if kind == ProposalBudgetKind.MUTATION and mutation_budget is None:
        raise ValueError("typed mutation proposal requires a mutation budget")
    maximum_attempts = 1 if kind == ProposalBudgetKind.MUTATION else max_repairs + 1

    for attempt_number in range(maximum_attempts):
        selected_budget = mutation_budget if kind == ProposalBudgetKind.MUTATION else budget
        assert selected_budget is not None
        if selected_budget.exhausted:
            break

        selected_budget.reserve()
        b_gen = budget.snapshot().used
        b_mut = mutation_budget.snapshot().used if mutation_budget is not None else 0
        generation = generator.generate(current_request)
        if generation.program is None:
            evaluation = EvaluationResult(
                ProposalStatus.INVALID,
                invalid_reason=InvalidReason.OTHER,
                metadata={"detail": "generation did not produce a kernel program"},
            )
        else:
            evaluation = _evaluate_with_retries(
                evaluator,
                generation.program,
                current_request.workload,
                max_infrastructure_retries,
            )

        attempts.append(
            GenerationAttempt(
                b_gen,
                b_mut,
                kind,
                attempt_number,
                generation,
                evaluation,
            )
        )
        if evaluation.status != ProposalStatus.INVALID:
            return ProposalOutcome(tuple(attempts), evaluation)

        if (
            kind == ProposalBudgetKind.GENERATION
            and attempt_number < max_repairs
            and not budget.exhausted
        ):
            current_request = GenerationRequest(
                parent=request.parent,
                strategy=request.strategy,
                workload=request.workload,
                hardware=request.hardware,
                profile=request.profile,
                incoming_profile_delta=request.incoming_profile_delta,
                attempt=attempt_number + 1,
                previous_program=generation.program,
                previous_result=evaluation,
            )

    if attempts:
        return ProposalOutcome(tuple(attempts), attempts[-1].evaluation)
    raise RuntimeError(f"proposal started with an exhausted {kind.value} budget")


def _evaluate_with_retries(
    evaluator: KernelEvaluator,
    program: KernelProgram,
    workload: WorkloadContract,
    max_infrastructure_retries: int,
) -> EvaluationResult:
    result = evaluator.evaluate(program, workload)
    for _ in range(max_infrastructure_retries):
        if result.status != ProposalStatus.INFRASTRUCTURE_FAILURE:
            break
        result = evaluator.evaluate(program, workload)
    return result
