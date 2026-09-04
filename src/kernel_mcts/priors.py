from __future__ import annotations

from typing import Mapping, Sequence

from .domain import KernelProgram, Strategy, WorkloadContract


class UniformStrategyPrior:
    name = "uniform"
    counts_toward_b_prior = False

    def get_priors(
        self,
        program: KernelProgram,
        workload: WorkloadContract,
        strategies: Sequence[Strategy],
        profile: Mapping[str, object] | None,
    ) -> Mapping[str, float]:
        if not strategies:
            raise ValueError("at least one strategy is required")
        probability = 1.0 / len(strategies)
        return {strategy.id: probability for strategy in strategies}


def validate_priors(
    priors: Mapping[str, float], strategies: Sequence[Strategy]
) -> dict[str, float]:
    expected = {strategy.id for strategy in strategies}
    if set(priors) != expected:
        raise ValueError("prior keys must exactly match strategy IDs")
    if any(value < 0 for value in priors.values()):
        raise ValueError("strategy priors cannot be negative")
    total = sum(priors.values())
    if total <= 0:
        raise ValueError("strategy priors must have positive mass")
    return {key: value / total for key, value in priors.items()}
