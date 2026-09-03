from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

from .domain import EvaluationResult, KernelProgram, Strategy, WorkloadContract


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    parent: KernelProgram
    strategy: Strategy
    workload: WorkloadContract
    hardware: Mapping[str, object]
    profile: Mapping[str, object] | None
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


class KernelGenerator(Protocol):
    def generate(self, request: GenerationRequest) -> GenerationResult: ...
