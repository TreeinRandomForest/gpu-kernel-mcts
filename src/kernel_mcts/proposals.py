from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

from .budget import GenerationBudget, MutationBudget
from .domain import EvaluationResult, InvalidReason, KernelProgram, ProposalStatus, Strategy, WorkloadContract
from .generation import (
    GenerationRequest,
    GenerationResult,
    KernelGenerator,
    ProposalBudgetKind,
    select_proposal_mechanism,
)
from .interfaces import KernelEvaluator
from .strategy_applicability import applicable_strategies, triton_parent_stage
from .triton_action_fidelity import (
    COALESCED_LAYOUT_ACTION, PIPELINED_WGMMA_ACTION, SERIAL_WGMMA_ACTION,
    TRAVERSAL_ACTION, TRITON_CONSTEXPR_ACTION, TRITON_POINTER_LOOP_ACTION,
    TRITON_WIDE_TILE_ACTION,
    coalesced_layout_edit_mismatch, pipelined_wgmma_api_mismatch,
    serial_wgmma_api_mismatch, traversal_edit_mismatch,
    triton_constexpr_mismatch, triton_pointer_loop_mismatch,
    triton_wide_tile_mismatch,
)


_TRITON_ACTION_GUARDS = {
    TRAVERSAL_ACTION: traversal_edit_mismatch,
    COALESCED_LAYOUT_ACTION: coalesced_layout_edit_mismatch,
    SERIAL_WGMMA_ACTION: serial_wgmma_api_mismatch,
    PIPELINED_WGMMA_ACTION: pipelined_wgmma_api_mismatch,
    TRITON_POINTER_LOOP_ACTION: triton_pointer_loop_mismatch,
    TRITON_WIDE_TILE_ACTION: triton_wide_tile_mismatch,
    TRITON_CONSTEXPR_ACTION: triton_constexpr_mismatch,
}


@dataclass(frozen=True, slots=True)
class GenerationAttempt:
    b_gen: int
    b_mut: int
    budget_kind: ProposalBudgetKind
    attempt_number: int
    generation: GenerationResult
    evaluation: EvaluationResult
    mechanism_candidates: tuple[ProposalBudgetKind, ...] = ()

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
    selected_strategies: Mapping[str, Strategy] | None = None,
    allowed_candidate_stages: frozenset[str] | None = None,
) -> ProposalOutcome:
    """Generate and evaluate one logical proposal with explicit mechanism accounting."""
    if max_repairs < 0 or max_infrastructure_retries < 0:
        raise ValueError("retry limits cannot be negative")
    attempts: list[GenerationAttempt] = []
    current_request = request
    mechanism, mechanism_candidates = select_proposal_mechanism(
        generator,
        request,
        generation_budget_available=not budget.exhausted,
        mutation_budget_available=(
            mutation_budget is not None and not mutation_budget.exhausted
        ),
    )
    if mechanism is None:
        raise RuntimeError("proposal started with no available proposal mechanism")
    kind = mechanism.kind
    selected_generator = mechanism.generator
    maximum_attempts = 1 if kind == ProposalBudgetKind.MUTATION else max_repairs + 1

    for attempt_number in range(maximum_attempts):
        selected_budget = mutation_budget if kind == ProposalBudgetKind.MUTATION else budget
        assert selected_budget is not None
        if selected_budget.exhausted:
            break

        selected_budget.reserve()
        b_gen = budget.snapshot().used
        b_mut = mutation_budget.snapshot().used if mutation_budget is not None else 0
        generation = selected_generator.generate(current_request)
        effective_strategy = current_request.strategy
        choice_error: str | None = None
        if selected_strategies is not None:
            choice = generation.metadata or {}
            selected_id = choice.get("selected_strategy_id")
            status = choice.get("strategy_choice_status")
            if status != "valid" or not isinstance(selected_id, str):
                choice_error = f"invalid sequential strategy choice: {status}"
            else:
                effective_strategy = selected_strategies.get(selected_id, current_request.strategy)
                eligible = applicable_strategies(
                    tuple(selected_strategies.values()), current_request.parent,
                    current_request.workload, current_request.hardware,
                )
                if effective_strategy not in eligible:
                    choice_error = f"strategy {selected_id} is not applicable to the current parent"
        if generation.program is None:
            evaluation = EvaluationResult(
                ProposalStatus.INVALID,
                invalid_reason=InvalidReason.OTHER,
                metadata={"detail": "generation did not produce a kernel program"},
            )
        elif choice_error is not None:
            evaluation = EvaluationResult(
                ProposalStatus.INVALID, program=generation.program,
                invalid_reason=InvalidReason.STRATEGY_MISMATCH,
                metadata={"detail": choice_error},
            )
        elif (allowed_candidate_stages is not None
              and (generation.program.backend != current_request.parent.backend
                   or triton_parent_stage(generation.program.source) not in allowed_candidate_stages)):
            evaluation = EvaluationResult(
                ProposalStatus.INVALID, program=generation.program,
                invalid_reason=InvalidReason.STRATEGY_MISMATCH,
                metadata={"detail": "candidate is outside the allowed Triton source stages"},
            )
        elif effective_strategy.candidate_stage is not None:
            expected_stage = effective_strategy.candidate_stage
            actual_stage = (
                triton_parent_stage(generation.program.source)
                if generation.program.backend == current_request.parent.backend == "triton_h100"
                else "wrong_backend"
            )
            if actual_stage != expected_stage:
                evaluation = EvaluationResult(
                    ProposalStatus.INVALID,
                    program=generation.program,
                    invalid_reason=InvalidReason.STRATEGY_MISMATCH,
                    metadata={
                        "detail": "generated source did not match the selected strategy stage",
                        "expected_candidate_stage": expected_stage,
                        "observed_candidate_stage": actual_stage,
                    },
                )
            elif (effective_strategy.id in _TRITON_ACTION_GUARDS
                  and (detail := _TRITON_ACTION_GUARDS[effective_strategy.id](
                      current_request.parent.source, generation.program.source
                  )) is not None):
                evaluation = EvaluationResult(
                    ProposalStatus.INVALID,
                    program=generation.program,
                    invalid_reason=InvalidReason.STRATEGY_MISMATCH,
                    metadata={
                        "detail": detail,
                        "expected_candidate_stage": expected_stage,
                        "observed_candidate_stage": actual_stage,
                        "strategy_edit_guard": effective_strategy.id,
                    },
                )
            else:
                evaluation = _evaluate_with_retries(
                    evaluator, generation.program, current_request.workload,
                    max_infrastructure_retries,
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
                mechanism_candidates,
            )
        )
        if evaluation.status != ProposalStatus.INVALID:
            return ProposalOutcome(tuple(attempts), evaluation)

        if (
            kind == ProposalBudgetKind.GENERATION
            and attempt_number < max_repairs
            and not budget.exhausted
        ):
            metadata = generation.metadata or {}
            declared_id = metadata.get("selected_strategy_id")
            locked_id = metadata.get("strategy_choice_lock_id")
            current_request = replace(
                request,
                attempt=attempt_number + 1,
                previous_program=generation.program,
                previous_result=evaluation,
                strategy_choice_original_id=(
                    current_request.strategy_choice_original_id
                    or (declared_id if isinstance(declared_id, str) else None)
                ),
                strategy_choice_locked_id=(
                    current_request.strategy_choice_locked_id
                    or (locked_id if isinstance(locked_id, str) else None)
                ),
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
