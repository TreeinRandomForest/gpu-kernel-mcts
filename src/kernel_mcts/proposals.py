from __future__ import annotations

from dataclasses import dataclass

from .budget import GenerationBudget
from .domain import EvaluationResult, InvalidReason, KernelProgram, ProposalStatus, WorkloadContract
from .generation import GenerationRequest, GenerationResult, KernelGenerator
from .interfaces import KernelEvaluator


@dataclass(frozen=True, slots=True)
class GenerationAttempt:
    budget_index: int
    attempt_number: int
    generation: GenerationResult
    evaluation: EvaluationResult


@dataclass(frozen=True, slots=True)
class ProposalOutcome:
    attempts: tuple[GenerationAttempt, ...]
    result: EvaluationResult


def run_proposal(
    *,
    generator: KernelGenerator,
    evaluator: KernelEvaluator,
    budget: GenerationBudget,
    request: GenerationRequest,
    max_repairs: int,
    max_infrastructure_retries: int,
) -> ProposalOutcome:
    """Generate and evaluate one logical proposal, charging every LLM call."""
    if max_repairs < 0 or max_infrastructure_retries < 0:
        raise ValueError("retry limits cannot be negative")
    attempts: list[GenerationAttempt] = []
    current_request = request

    for attempt_number in range(max_repairs + 1):
        if budget.exhausted:
            break

        budget_index = budget.reserve()
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

        attempts.append(GenerationAttempt(budget_index, attempt_number, generation, evaluation))
        if evaluation.status != ProposalStatus.INVALID:
            return ProposalOutcome(tuple(attempts), evaluation)

        if attempt_number < max_repairs and not budget.exhausted:
            current_request = GenerationRequest(
                parent=request.parent,
                strategy=request.strategy,
                workload=request.workload,
                hardware=request.hardware,
                profile=request.profile,
                attempt=attempt_number + 1,
                previous_program=generation.program,
                previous_result=evaluation,
            )

    if attempts:
        return ProposalOutcome(tuple(attempts), attempts[-1].evaluation)
    raise RuntimeError("proposal started with an exhausted generation budget")


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
