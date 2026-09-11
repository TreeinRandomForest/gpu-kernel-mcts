from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import Mapping, Protocol, Sequence
from uuid import uuid4

from .budget import GenerationBudget
from .domain import (
    BenchmarkResult,
    EvaluationResult,
    KernelProgram,
    ProposalStatus,
    Strategy,
    WorkloadContract,
)
from .evaluation import root_normalized_reward
from .generation import KernelGenerator
from .interfaces import EventSink, KernelEvaluator, StrategyPriorProvider
from .providers import GPUProvider, GPUWorker, HardwareSpec
from .search import MCTS, MCTSConfig, SearchResult
from .serialization import serialize_program, serialize_workload


class SearchTraceStore(EventSink, Protocol):
    def start_run(
        self,
        run_id: str,
        benchmark_id: str,
        algorithm: str,
        config: Mapping[str, object],
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class SearchExecution:
    run_id: str
    result: SearchResult


class WorkerKernelEvaluator:
    """Adapt one run-scoped GPU worker to the search evaluator interface."""

    def __init__(self, worker: GPUWorker, run_id: str) -> None:
        self._worker = worker
        self._run_id = run_id

    def evaluate(
        self,
        program: KernelProgram,
        workload: WorkloadContract,
    ) -> EvaluationResult:
        payload = {
            "program": serialize_program(program),
            "workload": serialize_workload(workload),
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return self._worker.evaluate(
            f"{self._run_id}:evaluation:{digest}",
            program,
            workload,
            "tier0",
        )


class RootNormalizedEvaluator:
    """Normalize valid worker results against this run's measured root."""

    def __init__(
        self,
        delegate: KernelEvaluator,
        root_benchmark: BenchmarkResult,
    ) -> None:
        self._delegate = delegate
        self._root_benchmark = root_benchmark

    def evaluate(
        self,
        program: KernelProgram,
        workload: WorkloadContract,
    ) -> EvaluationResult:
        evaluation = self._delegate.evaluate(program, workload)
        if evaluation.status != ProposalStatus.VALID:
            return evaluation
        if evaluation.benchmark is None:
            raise RuntimeError("valid candidate evaluation must include a benchmark")
        return replace(
            evaluation,
            reward=root_normalized_reward(
                self._root_benchmark,
                evaluation.benchmark,
                workload,
            ),
        )


def run_mcts_search(
    *,
    provider: GPUProvider,
    hardware: HardwareSpec,
    workload: WorkloadContract,
    root_program: KernelProgram,
    strategies: Sequence[Strategy],
    generator: KernelGenerator,
    prior_provider: StrategyPriorProvider,
    generation_budget: int,
    trace: SearchTraceStore,
    mcts_config: MCTSConfig = MCTSConfig(),
    seed: int = 0,
    run_id: str | None = None,
) -> SearchExecution:
    """Run one global MCTS search on one acquired worker with durable traces."""
    budget = GenerationBudget(generation_budget)
    resolved_run_id = run_id or str(uuid4())
    hardware_payload = asdict(hardware)
    trace.start_run(
        resolved_run_id,
        workload.benchmark_id,
        "mcts",
        {
            "generation_budget": generation_budget,
            "mcts": asdict(mcts_config),
            "seed": seed,
        },
    )
    worker: GPUWorker | None = None
    mcts_started = False
    try:
        worker = provider.acquire_worker(hardware)
        trace.emit("environment_manifest", worker.get_environment_manifest().as_dict())
        worker_evaluator = WorkerKernelEvaluator(worker, resolved_run_id)
        root_evaluation = worker_evaluator.evaluate(root_program, workload)
        if root_evaluation.status != ProposalStatus.VALID:
            raise RuntimeError(
                f"root evaluation must be valid, got {root_evaluation.status.value}"
            )
        if root_evaluation.benchmark is None:
            raise RuntimeError("valid root evaluation must include a benchmark")
        root_evaluation = replace(root_evaluation, reward=0.0)
        evaluator = RootNormalizedEvaluator(
            worker_evaluator,
            root_evaluation.benchmark,
        )
        search = MCTS(
            strategies=strategies,
            workload=workload,
            generator=generator,
            evaluator=evaluator,
            prior_provider=prior_provider,
            budget=budget,
            hardware=hardware_payload,
            config=mcts_config,
            seed=seed,
            events=trace,
        )
        mcts_started = True
        return SearchExecution(resolved_run_id, search.run(root_evaluation))
    except Exception as error:
        if not mcts_started:
            trace.emit(
                "run_failed",
                {
                    "iterations": 0,
                    "b_gen": 0,
                    "b_prior": 0,
                    "error_type": type(error).__name__,
                },
            )
        raise
    finally:
        if worker is not None:
            provider.release_worker(worker)
