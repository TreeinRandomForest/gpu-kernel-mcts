from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import Callable, Mapping, Protocol, Sequence
from uuid import uuid4

from .autotuning import PostSearchAutotuner, TuningConfig, TuningResult
from .budget import GenerationBudget, MutationBudget
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
from .serialization import serialize_evaluation, serialize_program, serialize_workload


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
    tuning: TuningResult | None = None


@dataclass(frozen=True, slots=True)
class CandidateEvaluationExecution:
    run_id: str
    root: EvaluationResult
    candidate: EvaluationResult
    environment_manifest: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class StandaloneTuningExecution:
    run_id: str
    baseline: EvaluationResult
    tuning: TuningResult


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
        evaluation_id = _worker_evaluation_id(self._run_id, program, workload)
        return self._worker.evaluate(
            evaluation_id,
            program,
            workload,
            "tier0",
        )


class WorkerNodeProfiler:
    """Request a lazy profile of the worker-cached compiled artifact."""

    def __init__(
        self,
        worker: GPUWorker,
        run_id: str,
        metric_set: str = "lightweight_v1",
    ) -> None:
        self._worker = worker
        self._run_id = run_id
        self._metric_set = metric_set

    def lightweight_profile(
        self,
        evaluation: EvaluationResult,
        workload: WorkloadContract,
    ) -> Mapping[str, object]:
        if evaluation.program is None:
            raise ValueError("profiling requires an evaluated program")
        evaluation_id = _worker_evaluation_id(
            self._run_id,
            evaluation.program,
            workload,
        )
        return self._worker.profile(
            evaluation_id,
            "lightweight",
            self._metric_set,
        )


class WorkerMeasurementDriftMonitor:
    def __init__(self, worker: GPUWorker, run_id: str) -> None:
        self._worker = worker
        self._run_id = run_id
        self._calls = 0

    def remeasure(
        self, evaluation: EvaluationResult, workload: WorkloadContract
    ) -> EvaluationResult:
        if evaluation.program is None:
            raise ValueError("drift remeasurement requires an evaluated program")
        self._calls += 1
        evaluation_id = f"{_worker_evaluation_id(self._run_id, evaluation.program, workload)}-drift-{self._calls}"
        return self._worker.evaluate(evaluation_id, evaluation.program, workload, "tier0")


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


def run_candidate_evaluation(
    *,
    provider: GPUProvider,
    hardware: HardwareSpec,
    workload: WorkloadContract,
    root_program: KernelProgram,
    candidate_program: KernelProgram,
    run_id: str | None = None,
    max_infrastructure_retries: int = 1,
) -> CandidateEvaluationExecution:
    """Evaluate one candidate against one freshly measured root on one worker."""
    if max_infrastructure_retries < 0:
        raise ValueError("max_infrastructure_retries cannot be negative")
    resolved_run_id = run_id or str(uuid4())
    worker: GPUWorker | None = None
    try:
        worker = provider.acquire_worker(hardware)
        manifest = worker.get_environment_manifest().as_dict()
        worker_evaluator = WorkerKernelEvaluator(worker, resolved_run_id)
        root_evaluation = _evaluate_with_infrastructure_retries(
            worker_evaluator,
            root_program,
            workload,
            max_infrastructure_retries,
        )
        if root_evaluation.status != ProposalStatus.VALID:
            raise RuntimeError(
                f"root evaluation must be valid, got {root_evaluation.status.value}"
            )
        if root_evaluation.benchmark is None:
            raise RuntimeError("valid root evaluation must include a benchmark")
        root_evaluation = replace(root_evaluation, reward=0.0)
        candidate_evaluation = _evaluate_with_infrastructure_retries(
            RootNormalizedEvaluator(worker_evaluator, root_evaluation.benchmark),
            candidate_program,
            workload,
            max_infrastructure_retries,
        )
        return CandidateEvaluationExecution(
            resolved_run_id,
            root_evaluation,
            candidate_evaluation,
            manifest,
        )
    finally:
        if worker is not None:
            provider.release_worker(worker)


def _evaluate_with_infrastructure_retries(
    evaluator: KernelEvaluator,
    program: KernelProgram,
    workload: WorkloadContract,
    retries: int,
    on_attempt: Callable[[int, EvaluationResult], None] | None = None,
) -> EvaluationResult:
    result = evaluator.evaluate(program, workload)
    if on_attempt is not None:
        on_attempt(1, result)
    for retry in range(retries):
        if result.status != ProposalStatus.INFRASTRUCTURE_FAILURE:
            break
        result = evaluator.evaluate(program, workload)
        if on_attempt is not None:
            on_attempt(retry + 2, result)
    return result


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
    mutation_budget: int = 0,
    trace: SearchTraceStore,
    mcts_config: MCTSConfig = MCTSConfig(),
    seed: int = 0,
    run_id: str | None = None,
    model_name: str | None = None,
    run_metadata: Mapping[str, object] | None = None,
    tuning_config: TuningConfig | None = None,
) -> SearchExecution:
    """Run one global MCTS search on one acquired worker with durable traces."""
    budget = GenerationBudget(generation_budget)
    mutations = MutationBudget(mutation_budget)
    resolved_run_id = run_id or str(uuid4())
    hardware_payload = asdict(hardware)
    run_config: dict[str, object] = {
        "generation_budget": generation_budget,
        "mutation_budget": mutation_budget,
        "mcts": asdict(mcts_config),
        "seed": seed,
    }
    if model_name is not None:
        run_config["model_name"] = model_name
    if run_metadata is not None:
        reserved = set(run_config) & set(run_metadata)
        if reserved:
            names = ", ".join(sorted(reserved))
            raise ValueError(f"run metadata overrides reserved keys: {names}")
        run_config.update(run_metadata)
    trace.start_run(
        resolved_run_id,
        workload.benchmark_id,
        "mcts",
        run_config,
    )
    worker: GPUWorker | None = None
    mcts_started = False
    root_evaluation: EvaluationResult | None = None
    root_attempts = 0
    try:
        worker = provider.acquire_worker(hardware)
        trace.emit("environment_manifest", worker.get_environment_manifest().as_dict())
        worker_evaluator = WorkerKernelEvaluator(worker, resolved_run_id)
        def record_root_attempt(attempt: int, result: EvaluationResult) -> None:
            nonlocal root_attempts
            root_attempts = attempt
            trace.emit(
                "root_evaluation_attempt",
                {
                    "attempt": attempt,
                    "max_infrastructure_retries": (
                        mcts_config.max_infrastructure_retries
                    ),
                    "evaluation": serialize_evaluation(result),
                },
            )

        root_evaluation = _evaluate_with_infrastructure_retries(
            worker_evaluator,
            root_program,
            workload,
            mcts_config.max_infrastructure_retries,
            record_root_attempt,
        )
        if root_evaluation.status != ProposalStatus.VALID:
            error_type = root_evaluation.metadata.get("error_type")
            detail = f", error_type={error_type}" if isinstance(error_type, str) else ""
            raise RuntimeError(
                f"root evaluation must be valid after {root_attempts} attempt(s), "
                f"got {root_evaluation.status.value}{detail}"
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
            mutation_budget=mutations,
            hardware=hardware_payload,
            config=mcts_config,
            seed=seed,
            events=trace,
            profiler=WorkerNodeProfiler(
                worker,
                resolved_run_id,
                mcts_config.profile_metric_set,
            ),
            drift_monitor=WorkerMeasurementDriftMonitor(worker, resolved_run_id),
        )
        mcts_started = True
        result = search.run(root_evaluation)
        tuning = (
            PostSearchAutotuner(evaluator, workload, tuning_config, trace).run(
                result.best.evaluation
            )
            if tuning_config is not None
            else None
        )
        return SearchExecution(resolved_run_id, result, tuning)
    except Exception as error:
        if not mcts_started:
            payload: dict[str, object] = {
                "iterations": 0,
                "b_gen": 0,
                "b_mut": 0,
                "b_prior": 0,
                "error_type": type(error).__name__,
                "root_evaluation_attempts": root_attempts,
            }
            if root_evaluation is not None:
                payload["root_evaluation_status"] = root_evaluation.status.value
                underlying = root_evaluation.metadata.get("error_type")
                if isinstance(underlying, str):
                    payload["root_error_type"] = underlying
            trace.emit("run_failed", payload)
        raise
    finally:
        if worker is not None:
            provider.release_worker(worker)


def run_standalone_autotuning(
    *,
    provider: GPUProvider,
    hardware: HardwareSpec,
    workload: WorkloadContract,
    program: KernelProgram,
    tuning_config: TuningConfig,
    trace: SearchTraceStore,
    run_id: str | None = None,
    run_metadata: Mapping[str, object] | None = None,
) -> StandaloneTuningExecution:
    """Tune one annotated program on one run-scoped worker."""
    resolved_run_id = run_id or str(uuid4())
    config: dict[str, object] = {
        "tuning": asdict(tuning_config),
        "seed": tuning_config.seed,
    }
    if run_metadata:
        config.update(run_metadata)
    trace.start_run(
        resolved_run_id,
        workload.benchmark_id,
        "standalone_autotuning",
        config,
    )
    worker: GPUWorker | None = None
    try:
        worker = provider.acquire_worker(hardware)
        trace.emit("environment_manifest", worker.get_environment_manifest().as_dict())
        evaluator = WorkerKernelEvaluator(worker, resolved_run_id)
        baseline = evaluator.evaluate(program, workload)
        trace.emit(
            "tuning_baseline_evaluated",
            {"evaluation": serialize_evaluation(baseline)},
        )
        if baseline.status != ProposalStatus.VALID or baseline.benchmark is None:
            raise RuntimeError(
                f"tuning baseline must be valid, got {baseline.status.value}"
            )
        tuning = PostSearchAutotuner(
            evaluator,
            workload,
            tuning_config,
            trace,
        ).run(baseline)
        return StandaloneTuningExecution(resolved_run_id, baseline, tuning)
    finally:
        if worker is not None:
            provider.release_worker(worker)


def _worker_evaluation_id(
    run_id: str,
    program: KernelProgram,
    workload: WorkloadContract,
) -> str:
    payload = {
        "program": serialize_program(program),
        "workload": serialize_workload(workload),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return f"{run_id}:evaluation:{digest}"
