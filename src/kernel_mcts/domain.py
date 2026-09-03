from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping


class ProposalStatus(StrEnum):
    VALID = "VALID"
    INVALID = "INVALID"
    INFRASTRUCTURE_FAILURE = "INFRASTRUCTURE_FAILURE"


class InvalidReason(StrEnum):
    COMPILE_FAILURE = "COMPILE_FAILURE"
    LAUNCH_FAILURE = "LAUNCH_FAILURE"
    CORRECTNESS_FAILURE = "CORRECTNESS_FAILURE"
    TIMEOUT = "TIMEOUT"
    BENCHMARK_FAILURE = "BENCHMARK_FAILURE"
    OTHER = "OTHER"


@dataclass(frozen=True, slots=True)
class ShapeCase:
    dimensions: Mapping[str, int]
    weight: float #total mass enforced by WorkloadContract


@dataclass(frozen=True, slots=True)
class WorkloadContract:
    benchmark_id: str
    operation: str
    dtype: str
    shapes: tuple[ShapeCase, ...] #distribution of shapes
    rtol: float #relative numerical tolerance
    atol: float #absolute numerical tolerance
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.shapes:
            raise ValueError("workload must contain at least one benchmark shape")
        total = sum(item.weight for item in self.shapes)
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"shape weights must sum to 1, got {total}")
        if any(item.weight <= 0 for item in self.shapes):
            raise ValueError("shape weights must be positive")


@dataclass(frozen=True, slots=True)
class KernelProgram:
    source: str
    backend: str = "cuda_cpp"


@dataclass(frozen=True, slots=True)
class Strategy: #prompts for each strategy (and for each backend)
    id: str
    description: str
    prompts: Mapping[str, str] #backend |-> strategy prompt

    def prompt_for(self, backend: str) -> str:
        try:
            return self.prompts[backend]
        except KeyError as error:
            raise ValueError(f"strategy {self.id!r} has no prompt for {backend!r}") from error


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    timings_us: tuple[float, ...]
    median_us: float
    per_shape_median_us: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EvaluationResult: #leaf candidate result
    status: ProposalStatus
    program: KernelProgram | None = None
    state_key: str | None = None #hash of kernel+workload/launch/hardware/toolchain to identify duplicate nodes
    reward: float | None = None #log(root_lat/leaf_lat)
    benchmark: BenchmarkResult | None = None
    invalid_reason: InvalidReason | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status == ProposalStatus.VALID:
            if self.program is None or self.state_key is None or self.reward is None:
                raise ValueError("valid evaluations require program, state_key, and reward")
        elif self.invalid_reason is None and self.status == ProposalStatus.INVALID:
            raise ValueError("invalid evaluations require an invalid_reason")

