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
