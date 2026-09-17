from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from random import Random
from typing import Mapping, Sequence

from .domain import EvaluationResult, KernelProgram, ProposalStatus, WorkloadContract
from .interfaces import EventSink, KernelEvaluator
from .serialization import serialize_evaluation


_ANNOTATION = re.compile(
    r"^\s*//\s*KERNEL_MCTS_TUNE\s+([A-Z][A-Z0-9_]*)\s*=\s*([0-9]+(?:\s*,\s*[0-9]+)+)\s*$",
    re.MULTILINE,
)


@dataclass(frozen=True, slots=True)
class TuningParameter:
    name: str
    choices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TuningConfig:
    budget: int
    method: str = "random"
    seed: int = 0

    def __post_init__(self) -> None:
        if self.budget < 1:
            raise ValueError("tuning budget must be positive")
        if self.method not in {"random", "grid"}:
            raise ValueError("tuning method must be 'random' or 'grid'")


@dataclass(frozen=True, slots=True)
class TuningTrial:
    number: int
    parameters: Mapping[str, int]
    evaluation: EvaluationResult


@dataclass(frozen=True, slots=True)
class TuningResult:
    parameters: tuple[TuningParameter, ...]
    trials: tuple[TuningTrial, ...]
    best: EvaluationResult | None
    best_parameters: Mapping[str, int] | None
    baseline_reward: float | None = None
    skipped_reason: str | None = None

    @property
    def used(self) -> int:
        return len(self.trials)

    @property
    def improved(self) -> bool:
        return (
            self.best is not None
            and self.best.reward is not None
            and self.baseline_reward is not None
            and self.best.reward > self.baseline_reward
        )


def parse_tuning_parameters(program: KernelProgram) -> tuple[TuningParameter, ...]:
    parameters: list[TuningParameter] = []
    seen: set[str] = set()
    for match in _ANNOTATION.finditer(program.source):
        name = match.group(1)
        if name in seen:
            raise ValueError(f"duplicate tuning parameter {name}")
        choices = tuple(dict.fromkeys(int(value.strip()) for value in match.group(2).split(",")))
        if len(choices) < 2:
            raise ValueError(f"invalid choices for tuning parameter {name}")
        define = re.compile(rf"^\s*#define\s+{re.escape(name)}\s+[0-9]+\s*$", re.MULTILINE)
        if define.search(program.source) is None:
            raise ValueError(f"tuning parameter {name} has no integer #define")
        seen.add(name)
        parameters.append(TuningParameter(name, choices))
    return tuple(parameters)


def render_configuration(
    program: KernelProgram, parameters: Mapping[str, int]
) -> KernelProgram:
    source = program.source
    for name, value in parameters.items():
        define = re.compile(
            rf"^(\s*#define\s+{re.escape(name)}\s+)[0-9]+(\s*)$", re.MULTILINE
        )
        source, count = define.subn(rf"\g<1>{value}\g<2>", source, count=1)
        if count != 1:
            raise ValueError(f"tuning parameter {name} has no unique integer #define")
    return KernelProgram(source, program.backend)


class PostSearchAutotuner:
    def __init__(
        self,
        evaluator: KernelEvaluator,
        workload: WorkloadContract,
        config: TuningConfig,
        events: EventSink,
    ) -> None:
        self._evaluator = evaluator
        self._workload = workload
        self._config = config
        self._events = events

    def run(self, baseline: EvaluationResult) -> TuningResult:
        if baseline.program is None:
            raise ValueError("autotuning requires a program")
        parameters = parse_tuning_parameters(baseline.program)
        if not parameters:
            reason = "final best kernel has no KERNEL_MCTS_TUNE annotations"
            self._events.emit("tuning_skipped", {"reason": reason})
            return TuningResult((), (), None, None, baseline.reward, reason)

        configurations = list(_configurations(parameters))
        if self._config.method == "random":
            Random(self._config.seed).shuffle(configurations)
        configurations = configurations[: self._config.budget]
        self._events.emit(
            "tuning_started",
            {
                "budget": self._config.budget,
                "method": self._config.method,
                "seed": self._config.seed,
                "parameters": [
                    {"name": item.name, "choices": list(item.choices)}
                    for item in parameters
                ],
                "configuration_count": len(configurations),
            },
        )
        trials: list[TuningTrial] = []
        best: EvaluationResult | None = None
        best_parameters: Mapping[str, int] | None = None
        for number, configuration in enumerate(configurations, 1):
            program = render_configuration(baseline.program, configuration)
            evaluation = self._evaluator.evaluate(program, self._workload)
            trial = TuningTrial(number, configuration, evaluation)
            trials.append(trial)
            if (
                evaluation.status == ProposalStatus.VALID
                and evaluation.reward is not None
                and (best is None or best.reward is None or evaluation.reward > best.reward)
            ):
                best = evaluation
                best_parameters = configuration
            self._events.emit(
                "tuning_trial",
                {
                    "trial": number,
                    "b_tune": number,
                    "parameters": dict(configuration),
                    "evaluation": serialize_evaluation(evaluation),
                },
            )
        self._events.emit(
            "tuning_completed",
            {
                "b_tune": len(trials),
                "best_parameters": dict(best_parameters or {}),
                "baseline_reward": baseline.reward,
                "best_evaluation": serialize_evaluation(best) if best else None,
            },
        )
        return TuningResult(
            parameters,
            tuple(trials),
            best,
            best_parameters,
            baseline.reward,
        )


def _configurations(
    parameters: Sequence[TuningParameter],
) -> Sequence[dict[str, int]]:
    return tuple(
        dict(zip((parameter.name for parameter in parameters), values))
        for values in itertools.product(*(parameter.choices for parameter in parameters))
    )
