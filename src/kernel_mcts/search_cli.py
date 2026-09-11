from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Mapping, TextIO

from .benchmarks import BF16_GEMM_WORKLOAD, load_bf16_gemm_root
from .config import load_data, parse_strategies
from .domain import Strategy
from .generation import GenerationRequest, GenerationResult, KernelGenerator
from .llm_generation import LLMKernelGenerator
from .openai_client import OpenAIResponsesClient, OpenAIResponsesConfig
from .orchestration import run_mcts_search
from .persistence import SQLiteTraceStore
from .priors import UniformStrategyPrior
from .providers import HardwareSpec, RunPodConfig
from .runpod import create_runpod_provider
from .runpod_cli import ReadinessProgress
from .runpod_volume import resolve_reusable_volume
from .search import MCTSConfig
from .smoke import SmokeKernelGenerator


class SearchCLIProgress:
    """Report durable search events without changing persistence semantics."""

    def __init__(
        self,
        trace: SQLiteTraceStore,
        readiness: ReadinessProgress,
        stream: TextIO = sys.stdout,
    ) -> None:
        self._trace = trace
        self._readiness = readiness
        self._stream = stream

    def start_run(
        self,
        run_id: str,
        benchmark_id: str,
        algorithm: str,
        config: Mapping[str, object],
    ) -> None:
        self._trace.start_run(run_id, benchmark_id, algorithm, config)

    def emit(self, event_type: str, payload: Mapping[str, object]) -> None:
        self._trace.emit(event_type, payload)
        if event_type == "environment_manifest":
            self._readiness.finish()
            self._write("Worker ready; starting root evaluation")
        elif event_type == "root_evaluation_attempt":
            evaluation = payload.get("evaluation")
            status = evaluation.get("status") if isinstance(evaluation, Mapping) else None
            self._write(
                f"Root evaluation attempt {payload.get('attempt')}: status={status}"
            )
        elif event_type == "profiling_started":
            self._write(
                f"Starting lightweight profile {payload.get('profile_call')} "
                f"for node {payload.get('node_id')}"
            )
        elif event_type == "node_profiled":
            self._write(
                f"Lightweight profile completed: profile_call={payload.get('profile_call')}"
            )
        elif event_type == "generation":
            self._write(
                "Generation evaluated: "
                f"B_gen={payload.get('b_gen')}, status={payload.get('proposal_status')}, "
                f"strategy={payload.get('strategy_id')}"
            )
        elif event_type == "iteration_completed":
            self._write(
                "MCTS iteration completed: "
                f"iteration={payload.get('iteration')}, B_gen={payload.get('b_gen')}, "
                f"status={payload.get('status')}, "
                f"backed_up_reward={payload.get('backed_up_reward')}"
            )
        elif event_type == "new_global_best":
            self._write(
                f"New best: iteration={payload.get('iteration')}, "
                f"reward={payload.get('reward')}"
            )
        elif event_type == "run_completed":
            self._write(
                "Search completed; terminating worker: "
                f"iterations={payload.get('iterations')}, B_gen={payload.get('b_gen')}, "
                f"profiles={payload.get('profile_calls')}, "
                f"best_reward={payload.get('best_reward')}"
            )
        elif event_type == "run_failed":
            self._write(
                f"Search failed: error_type={payload.get('error_type')}, "
                f"B_gen={payload.get('b_gen')}"
            )

    def _write(self, message: str) -> None:
        self._stream.write(f"{message}\n")
        self._stream.flush()


class ProgressKernelGenerator:
    """Make otherwise quiet remote LLM calls visible to CLI users."""

    def __init__(
        self,
        generator: KernelGenerator,
        stream: TextIO = sys.stdout,
    ) -> None:
        self._generator = generator
        self._stream = stream
        self._calls = 0

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self._calls += 1
        self._write(
            f"Starting generation call {self._calls}: strategy={request.strategy.id}"
        )
        result = self._generator.generate(request)
        self._write(
            f"Generation call {self._calls} returned; evaluating proposal"
        )
        return result

    def _write(self, message: str) -> None:
        self._stream.write(f"{message}\n")
        self._stream.flush()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run deterministic-smoke or OpenAI-backed MCTS on an H100 worker."
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--api-key-env", default="RUNPOD_API_KEY")
    parser.add_argument("--gpu-type", default="NVIDIA H100 80GB HBM3")
    parser.add_argument("--network-volume-id")
    parser.add_argument("--data-center-id")
    parser.add_argument("--auto-volume", action="store_true")
    parser.add_argument(
        "--ephemeral-storage",
        action="store_true",
        help="use only the terminated pod's container disk with no data-center affinity",
    )
    parser.add_argument("--preferred-data-center-id")
    parser.add_argument("--volume-name", default="gpu-kernel-mcts")
    parser.add_argument("--generation-budget", type=int, default=1)
    parser.add_argument("--generator", choices=("smoke", "openai"), default="smoke")
    parser.add_argument("--model")
    parser.add_argument("--strategies", type=Path)
    parser.add_argument("--openai-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--max-output-tokens", type=int, default=16_384)
    parser.add_argument("--llm-timeout", type=float, default=180.0)
    parser.add_argument("--max-repairs", type=int, default=1)
    parser.add_argument("--max-infrastructure-retries", type=int, default=1)
    parser.add_argument("--k-max", type=int, default=4)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--best-output", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--run-id")
    parser.add_argument(
        "--confirm-create-and-terminate",
        action="store_true",
        help="required because this command creates a billable Pod",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if not arguments.confirm_create_and_terminate:
        parser.error("--confirm-create-and-terminate is required")
    if arguments.generation_budget < 1:
        parser.error("--generation-budget must be positive")
    if arguments.best_output is not None and arguments.best_output.exists():
        parser.error(f"refusing to overwrite existing best output: {arguments.best_output}")
    strategies, generator, model_name, mcts_config = _search_components(
        parser, arguments
    )
    api_key = os.environ.get(arguments.api_key_env)
    if not api_key:
        parser.error(f"environment variable {arguments.api_key_env!r} is not set")
    network_volume_id, data_center_id = resolve_reusable_volume(
        parser, arguments, api_key
    )

    progress = ReadinessProgress("search worker")
    provider = create_runpod_provider(
        RunPodConfig(
            image=arguments.image,
            api_key_env=arguments.api_key_env,
            gpu_type=arguments.gpu_type,
            startup_timeout_seconds=arguments.timeout,
            network_volume_id=network_volume_id,
            data_center_ids=(data_center_id,) if data_center_id is not None else (),
        ),
        readiness_progress=progress,
    )
    try:
        with SQLiteTraceStore(arguments.trace) as trace:
            reporting_trace = SearchCLIProgress(trace, progress)
            execution = run_mcts_search(
                provider=provider,
                hardware=HardwareSpec(
                    "H100",
                    form_factor="SXM",
                    minimum_compute_capability="9.0",
                    required_profilers=("ncu",),
                ),
                workload=BF16_GEMM_WORKLOAD,
                root_program=load_bf16_gemm_root(),
                strategies=strategies,
                generator=ProgressKernelGenerator(generator),
                prior_provider=UniformStrategyPrior(),
                generation_budget=arguments.generation_budget,
                trace=reporting_trace,
                mcts_config=mcts_config,
                seed=arguments.seed,
                run_id=arguments.run_id,
                model_name=model_name,
            )
    finally:
        progress.finish()

    result = execution.result
    if arguments.best_output is not None:
        arguments.best_output.write_text(result.best.program.source, encoding="utf-8")
    generator_label = "OpenAI" if arguments.generator == "openai" else "Smoke"
    print(
        f"{generator_label} search completed: run_id={execution.run_id}, "
        f"iterations={result.iterations}, B_gen={result.generations}, "
        f"profiles={result.profile_calls}, nodes={len(result.nodes)}, "
        f"best_reward={result.best.reward:.6g}"
    )
    print(f"SQLite trace: {arguments.trace.resolve()}")
    if arguments.best_output is not None:
        print(f"Best kernel: {arguments.best_output.resolve()}")
    return 0


def _search_components(
    parser: argparse.ArgumentParser,
    arguments: argparse.Namespace,
) -> tuple[tuple[Strategy, ...], KernelGenerator, str | None, MCTSConfig]:
    if arguments.generator == "smoke":
        if arguments.generation_budget != 1:
            parser.error("the deterministic smoke run requires --generation-budget=1")
        return (
            (
                Strategy(
                    "deterministic_smoke",
                    "Generate the fixed orchestration smoke candidate",
                    {"cuda_cpp": "fixed smoke candidate"},
                ),
            ),
            SmokeKernelGenerator(),
            None,
            MCTSConfig(
                k_max=1,
                max_repairs=0,
                max_depth=arguments.max_depth,
                max_infrastructure_retries=arguments.max_infrastructure_retries,
            ),
        )

    if not arguments.model:
        parser.error("--model is required with --generator=openai")
    if arguments.strategies is None:
        parser.error("--strategies is required with --generator=openai")
    if not os.environ.get(arguments.openai_api_key_env):
        parser.error(
            f"environment variable {arguments.openai_api_key_env!r} is not set"
        )
    strategies = parse_strategies(load_data(arguments.strategies))
    if not strategies:
        parser.error("strategy configuration must contain at least one strategy")
    generator = LLMKernelGenerator(
        OpenAIResponsesClient(
            OpenAIResponsesConfig(
                model=arguments.model,
                reasoning_effort=arguments.reasoning_effort,
                max_output_tokens=arguments.max_output_tokens,
                timeout_seconds=arguments.llm_timeout,
                api_key_env=arguments.openai_api_key_env,
                store=False,
            )
        )
    )
    return (
        strategies,
        generator,
        arguments.model,
        MCTSConfig(
            k_max=arguments.k_max,
            max_depth=arguments.max_depth,
            max_repairs=arguments.max_repairs,
            max_infrastructure_retries=arguments.max_infrastructure_retries,
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
