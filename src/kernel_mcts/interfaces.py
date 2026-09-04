from __future__ import annotations

from typing import Mapping, Protocol, Sequence

from .domain import EvaluationResult, KernelProgram, Strategy, WorkloadContract


class KernelEvaluator(Protocol):
    def evaluate(self, program: KernelProgram, workload: WorkloadContract) -> EvaluationResult: ...


class StrategyPriorProvider(Protocol):
    name: str
    counts_toward_b_prior: bool

    def get_priors(
        self,
        program: KernelProgram,
        workload: WorkloadContract,
        strategies: Sequence[Strategy],
        profile: Mapping[str, object] | None,
    ) -> Mapping[str, float]: ...


class NodeProfiler(Protocol):
    def lightweight_profile(
        self, evaluation: EvaluationResult, workload: WorkloadContract
    ) -> Mapping[str, object]: ...


class EventSink(Protocol):
    def emit(self, event_type: str, payload: Mapping[str, object]) -> None: ...


class NullEventSink:
    def emit(self, event_type: str, payload: Mapping[str, object]) -> None:
        pass
