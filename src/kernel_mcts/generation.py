from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping, Protocol

from .domain import EvaluationResult, KernelProgram, Strategy, WorkloadContract


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    parent: KernelProgram
    strategy: Strategy
    workload: WorkloadContract
    hardware: Mapping[str, object]
    profile: Mapping[str, object] | None
    incoming_profile_delta: Mapping[str, object] | None = None
    attempt: int = 0 #repair attempts
    previous_program: KernelProgram | None = None #track for repair
    previous_result: EvaluationResult | None = None #track for repair


@dataclass(frozen=True, slots=True)
class GenerationResult:
    generation_id: str
    raw_output: str
    program: KernelProgram | None
    prompt_hash: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_seconds: float | None = None
    metadata: Mapping[str, object] | None = None
    prompt_text: str | None = None
    instructions_text: str | None = None


class KernelGenerator(Protocol):
    def generate(self, request: GenerationRequest) -> GenerationResult: ...


class ProposalBudgetKind(StrEnum):
    GENERATION = "generation"
    MUTATION = "mutation"


@dataclass(frozen=True, slots=True)
class ProposalMechanism:
    kind: ProposalBudgetKind
    generator: KernelGenerator


class MutationFirstGenerator:
    """Route widening to typed mutation before falling back to generation."""

    def __init__(
        self,
        mutation_generator: KernelGenerator,
        generation_generator: KernelGenerator,
    ) -> None:
        self.mutation_generator = mutation_generator
        self.generation_generator = generation_generator

    def proposal_mechanisms(
        self, request: GenerationRequest
    ) -> tuple[ProposalMechanism, ...]:
        return (
            ProposalMechanism(ProposalBudgetKind.MUTATION, self.mutation_generator),
            ProposalMechanism(
                ProposalBudgetKind.GENERATION, self.generation_generator
            ),
        )


def proposal_budget_kind(
    generator: KernelGenerator, request: GenerationRequest
) -> ProposalBudgetKind:
    resolver = getattr(generator, "proposal_budget_kind", None)
    if resolver is None:
        return ProposalBudgetKind.GENERATION
    return ProposalBudgetKind(resolver(request))


def can_generate(generator: KernelGenerator, request: GenerationRequest) -> bool:
    predicate = getattr(generator, "can_generate", None)
    return True if predicate is None else bool(predicate(request))


def proposal_mechanisms(
    generator: KernelGenerator, request: GenerationRequest
) -> tuple[ProposalMechanism, ...]:
    resolver = getattr(generator, "proposal_mechanisms", None)
    if resolver is None:
        return (ProposalMechanism(proposal_budget_kind(generator, request), generator),)
    mechanisms = tuple(resolver(request))
    if not mechanisms:
        return ()
    return tuple(
        ProposalMechanism(ProposalBudgetKind(item.kind), item.generator)
        for item in mechanisms
    )


def select_proposal_mechanism(
    generator: KernelGenerator,
    request: GenerationRequest,
    *,
    generation_budget_available: bool,
    mutation_budget_available: bool,
) -> tuple[ProposalMechanism | None, tuple[ProposalBudgetKind, ...]]:
    """Select the first capable, budget-eligible mechanism in router order."""
    eligible: list[ProposalMechanism] = []
    for mechanism in proposal_mechanisms(generator, request):
        budget_available = (
            mutation_budget_available
            if mechanism.kind == ProposalBudgetKind.MUTATION
            else generation_budget_available
        )
        if not budget_available or not can_generate(mechanism.generator, request):
            continue
        eligible.append(mechanism)
    return (
        eligible[0] if eligible else None,
        tuple(mechanism.kind for mechanism in eligible),
    )
