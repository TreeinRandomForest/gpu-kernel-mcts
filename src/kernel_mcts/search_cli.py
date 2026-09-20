from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Mapping, TextIO

from .autotuning import TuningConfig
from .benchmarks import BF16_GEMM_WORKLOAD, load_bf16_gemm_root
from .config import load_data, parse_strategies
from .cute_mutations import CUTE_MUTATION_STRATEGIES, CuteMutationGenerator
from .cute_program import PinnedCuteGemmRenderer, REFERENCE_CUTE_GEMM
from .domain import Strategy
from .generation import (
    GenerationRequest,
    GenerationResult,
    KernelGenerator,
    MutationFirstGenerator,
    ProposalMechanism,
    proposal_budget_kind,
)
from .llm_generation import LLMKernelGenerator
from .nebius import NebiusConfig, create_nebius_provider
from .openai_client import OpenAIResponsesClient, OpenAIResponsesConfig
from .orchestration import run_mcts_search
from .persistence import SQLiteTraceStore
from .priors import UniformStrategyPrior
from .profiling import PROFILE_METRIC_SET_IDS
from .provenance import capture_repository_state, image_digest_from_reference
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
            status = (
                evaluation.get("status") if isinstance(evaluation, Mapping) else None
            )
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
                f"strategy={payload.get('strategy_id')}, "
                f"B_mut={payload.get('b_mut', 0)}"
            )
        elif event_type == "iteration_completed":
            self._write(
                "MCTS iteration completed: "
                f"iteration={payload.get('iteration')}, B_gen={payload.get('b_gen')}, "
                f"status={payload.get('status')}, "
                f"backed_up_reward={payload.get('backed_up_reward')}, "
                f"B_mut={payload.get('b_mut', 0)}"
            )
        elif event_type == "new_global_best":
            self._write(
                f"New best: iteration={payload.get('iteration')}, "
                f"reward={payload.get('reward')}"
            )
        elif event_type == "run_completed":
            self._write(
                "MCTS completed: "
                f"iterations={payload.get('iterations')}, B_gen={payload.get('b_gen')}, "
                f"B_mut={payload.get('b_mut', 0)}, "
                f"profiles={payload.get('profile_calls')}, "
                f"best_reward={payload.get('best_reward')}"
            )
        elif event_type == "tuning_started":
            self._write(
                f"Starting post-search autotuning: method={payload.get('method')}, "
                f"budget={payload.get('budget')}"
            )
        elif event_type == "tuning_trial":
            evaluation = payload.get("evaluation")
            status = evaluation.get("status") if isinstance(evaluation, Mapping) else None
            reward = evaluation.get("reward") if isinstance(evaluation, Mapping) else None
            self._write(
                f"Tuning trial completed: B_tune={payload.get('b_tune')}, "
                f"status={status}, reward={reward}"
            )
        elif event_type == "tuning_completed":
            self._write(f"Post-search autotuning completed: B_tune={payload.get('b_tune')}")
        elif event_type == "tuning_skipped":
            self._write(f"Post-search autotuning skipped: {payload.get('reason')}")
        elif event_type == "run_failed":
            self._write(
                f"Search failed: error_type={payload.get('error_type')}, "
                f"B_gen={payload.get('b_gen')}, B_mut={payload.get('b_mut', 0)}"
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
        self._write(f"Generation call {self._calls} returned; evaluating proposal")
        return result

    def can_generate(self, request: GenerationRequest) -> bool:
        predicate = getattr(self._generator, "can_generate", None)
        return True if predicate is None else bool(predicate(request))

    def proposal_mechanisms(
        self, request: GenerationRequest
    ) -> tuple[ProposalMechanism, ...]:
        resolver = getattr(self._generator, "proposal_mechanisms", None)
        if resolver is not None:
            return tuple(resolver(request))
        return (ProposalMechanism(proposal_budget_kind(self._generator, request), self),)

    def _write(self, message: str) -> None:
        self._stream.write(f"{message}\n")
        self._stream.flush()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run deterministic-smoke or OpenAI-backed MCTS on an H100 worker."
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument(
        "--backend", choices=("cuda_cpp", "cute_dsl"), default="cuda_cpp"
    )
    parser.add_argument("--provider", choices=("runpod", "nebius"), default="runpod")
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
    parser.add_argument("--mutation-budget", type=int, default=0)
    parser.add_argument("--autotune", action="store_true")
    parser.add_argument("--tuning-budget", type=int, default=20)
    parser.add_argument("--tuning-method", choices=("random", "grid"), default="random")
    parser.add_argument("--tuned-best-output", type=Path)
    parser.add_argument(
        "--generator",
        choices=("smoke", "openai", "cute-mutation", "cute-mixed"),
        default="smoke",
    )
    parser.add_argument("--model")
    parser.add_argument("--strategies", type=Path)
    parser.add_argument("--openai-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--max-output-tokens", type=int, default=16_384)
    parser.add_argument("--llm-timeout", type=float, default=180.0)
    parser.add_argument("--max-repairs", type=int, default=1)
    parser.add_argument("--max-infrastructure-retries", type=int, default=1)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--k-max", type=int, default=4)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--measurement-drift-interval", type=int, default=0)
    parser.add_argument("--measurement-drift-threshold", type=float, default=0.05)
    parser.add_argument(
        "--profile-metric-set",
        choices=PROFILE_METRIC_SET_IDS,
        default="lightweight_v1",
        help="versioned, worker-allowlisted NCU metric set",
    )
    parser.add_argument(
        "--include-incoming-profile-delta",
        action="store_true",
        help=(
            "include the selected incoming edge's profile delta in generation "
            "prompts (path-dependent ablation; disabled by default)"
        ),
    )
    parser.add_argument("--best-output", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--nebius-subnet-id")
    parser.add_argument("--nebius-project-id")
    parser.add_argument("--nebius-username", default=os.environ.get("USER", "user"))
    parser.add_argument(
        "--nebius-ssh-public-key",
        type=Path,
        default=Path.home() / ".ssh" / "id_ed25519.pub",
    )
    parser.add_argument(
        "--nebius-ssh-private-key",
        type=Path,
        default=Path.home() / ".ssh" / "id_ed25519",
    )
    parser.add_argument("--nebius-platform", default="gpu-h100-sxm")
    parser.add_argument("--nebius-preset", default="1gpu-16vcpu-200gb")
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
    if arguments.generation_budget < 0 or arguments.mutation_budget < 0:
        parser.error("proposal budgets cannot be negative")
    if arguments.generation_budget + arguments.mutation_budget < 1:
        parser.error("at least one proposal budget must be positive")
    if arguments.c_puct <= 0:
        parser.error("--c-puct must be positive")
    if arguments.best_output is not None and arguments.best_output.exists():
        parser.error(
            f"refusing to overwrite existing best output: {arguments.best_output}"
        )
    if arguments.tuned_best_output is not None and not arguments.autotune:
        parser.error("--tuned-best-output requires --autotune")
    if arguments.autotune and arguments.tuning_budget < 1:
        parser.error("--tuning-budget must be positive")
    if arguments.backend == "cute_dsl" and arguments.autotune:
        parser.error("post-search autotuning is not implemented for --backend=cute_dsl")
    if arguments.tuned_best_output is not None and arguments.tuned_best_output.exists():
        parser.error(
            f"refusing to overwrite existing tuned output: {arguments.tuned_best_output}"
        )
    strategies, generator, model_name, mcts_config = _search_components(
        parser, arguments
    )
    repository = capture_repository_state(Path(__file__).resolve().parents[2])
    image_digest = image_digest_from_reference(arguments.image)
    provenance: dict[str, object] = {
        "worker_image": arguments.image,
        "reasoning_effort": (
            arguments.reasoning_effort
            if arguments.generator in {"openai", "cute-mixed"}
            else "not_applicable"
        ),
        "backend": arguments.backend,
    }
    if repository.commit is not None:
        provenance["git_commit"] = repository.commit
    if repository.dirty is not None:
        provenance["dirty_tree"] = repository.dirty
    if image_digest is not None:
        provenance["worker_image_digest"] = image_digest
    progress = ReadinessProgress("search worker")
    if arguments.provider == "runpod":
        api_key = os.environ.get(arguments.api_key_env)
        if not api_key:
            parser.error(f"environment variable {arguments.api_key_env!r} is not set")
        network_volume_id, data_center_id = resolve_reusable_volume(
            parser, arguments, api_key
        )
        provider = create_runpod_provider(
            RunPodConfig(
                image=arguments.image,
                api_key_env=arguments.api_key_env,
                gpu_type=arguments.gpu_type,
                startup_timeout_seconds=arguments.timeout,
                network_volume_id=network_volume_id,
                data_center_ids=(data_center_id,) if data_center_id is not None else (),
                project_git_commit=repository.commit,
                project_dirty_tree=repository.dirty,
                container_digest=image_digest,
                backend=arguments.backend,
            ),
            readiness_progress=progress,
        )
    else:
        if not arguments.nebius_project_id or not arguments.nebius_subnet_id:
            parser.error(
                "--nebius-project-id and --nebius-subnet-id are required "
                "with --provider=nebius"
            )
        if any(
            (
                arguments.network_volume_id,
                arguments.data_center_id,
                arguments.auto_volume,
                arguments.ephemeral_storage,
                arguments.preferred_data_center_id,
            )
        ):
            parser.error("RunPod volume options cannot be used with --provider=nebius")
        provider = create_nebius_provider(
            NebiusConfig(
                image=arguments.image,
                project_id=arguments.nebius_project_id,
                subnet_id=arguments.nebius_subnet_id,
                ssh_public_key=arguments.nebius_ssh_public_key,
                ssh_private_key=arguments.nebius_ssh_private_key,
                username=arguments.nebius_username,
                platform=arguments.nebius_platform,
                preset=arguments.nebius_preset,
                startup_timeout_seconds=arguments.timeout,
                project_git_commit=repository.commit,
                project_dirty_tree=repository.dirty,
                container_digest=image_digest,
                backend=arguments.backend,
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
                root_program=(
                    load_bf16_gemm_root()
                    if arguments.backend == "cuda_cpp"
                    else PinnedCuteGemmRenderer().render(REFERENCE_CUTE_GEMM)
                ),
                strategies=strategies,
                generator=ProgressKernelGenerator(generator),
                prior_provider=UniformStrategyPrior(),
                generation_budget=arguments.generation_budget,
                mutation_budget=arguments.mutation_budget,
                trace=reporting_trace,
                mcts_config=mcts_config,
                seed=arguments.seed,
                run_id=arguments.run_id,
                model_name=model_name,
                run_metadata=provenance,
                tuning_config=(
                    TuningConfig(
                        arguments.tuning_budget,
                        arguments.tuning_method,
                        arguments.seed,
                    )
                    if arguments.autotune
                    else None
                ),
            )
    finally:
        progress.finish()

    result = execution.result
    if arguments.best_output is not None:
        arguments.best_output.write_text(result.best.program.source, encoding="utf-8")
    if (
        arguments.tuned_best_output is not None
        and execution.tuning is not None
        and execution.tuning.best is not None
        and execution.tuning.best.program is not None
    ):
        arguments.tuned_best_output.write_text(
            execution.tuning.best.program.source, encoding="utf-8"
        )
    generator_label = {
        "openai": "OpenAI",
        "smoke": "Smoke",
        "cute-mutation": "CuTe mutation",
        "cute-mixed": "CuTe mixed",
    }[arguments.generator]
    print(
        f"{generator_label} search completed: run_id={execution.run_id}, "
        f"iterations={result.iterations}, B_gen={result.generations}, "
        f"B_mut={getattr(result, 'mutations', 0)}, "
        f"profiles={result.profile_calls}, drift_probes={result.drift_probe_calls}, "
        f"nodes={len(result.nodes)}, "
        f"best_reward={result.best.reward:.6g}"
    )
    print(f"SQLite trace: {arguments.trace.resolve()}")
    if execution.tuning is not None:
        if execution.tuning.skipped_reason is not None:
            print(f"Autotuning skipped: {execution.tuning.skipped_reason}")
        else:
            tuned_reward = (
                execution.tuning.best.reward
                if execution.tuning.best is not None
                else result.best.reward
            )
            print(
                f"Autotuning completed: B_tune={execution.tuning.used}, "
                f"best_reward={tuned_reward:.6g}, "
                f"improved={execution.tuning.improved}"
            )
    if arguments.best_output is not None:
        print(f"Best kernel: {arguments.best_output.resolve()}")
    if arguments.tuned_best_output is not None and arguments.tuned_best_output.exists():
        print(f"Tuned best kernel: {arguments.tuned_best_output.resolve()}")
    return 0


def _search_components(
    parser: argparse.ArgumentParser,
    arguments: argparse.Namespace,
) -> tuple[tuple[Strategy, ...], KernelGenerator, str | None, MCTSConfig]:
    if arguments.generator == "smoke":
        if arguments.backend != "cuda_cpp":
            parser.error("--generator=smoke requires --backend=cuda_cpp")
        if arguments.generation_budget != 1:
            parser.error("the deterministic smoke run requires --generation-budget=1")
        if arguments.mutation_budget != 0:
            parser.error("the deterministic smoke run requires --mutation-budget=0")
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
                c_puct=arguments.c_puct,
                k_max=1,
                max_repairs=0,
                max_depth=arguments.max_depth,
                max_infrastructure_retries=arguments.max_infrastructure_retries,
                include_incoming_profile_delta=(
                    arguments.include_incoming_profile_delta
                ),
                profile_metric_set=arguments.profile_metric_set,
                measurement_drift_interval=arguments.measurement_drift_interval,
                measurement_drift_threshold=arguments.measurement_drift_threshold,
            ),
        )

    if arguments.generator == "cute-mutation":
        if arguments.backend != "cute_dsl":
            parser.error("--generator=cute-mutation requires --backend=cute_dsl")
        if arguments.generation_budget != 0 or arguments.mutation_budget < 1:
            parser.error(
                "the CuTe mutation run requires --generation-budget=0 and "
                "a positive --mutation-budget"
            )
        return (
            CUTE_MUTATION_STRATEGIES,
            CuteMutationGenerator(),
            None,
            _mcts_config(arguments, max_repairs=0),
        )

    if arguments.generator == "openai" and arguments.backend != "cuda_cpp":
        parser.error("--generator=openai requires --backend=cuda_cpp")
    if arguments.generator == "cute-mixed" and arguments.backend != "cute_dsl":
        parser.error("--generator=cute-mixed requires --backend=cute_dsl")
    if arguments.generation_budget < 1:
        parser.error("LLM-backed search requires a positive --generation-budget")
    if arguments.generator == "cute-mixed" and arguments.mutation_budget < 1:
        parser.error("the mixed CuTe run requires a positive --mutation-budget")
    if not arguments.model:
        parser.error(f"--model is required with --generator={arguments.generator}")
    if arguments.generator == "openai" and arguments.strategies is None:
        parser.error("--strategies is required with --generator=openai")
    if not os.environ.get(arguments.openai_api_key_env):
        parser.error(
            f"environment variable {arguments.openai_api_key_env!r} is not set"
        )
    strategies = (
        CUTE_MUTATION_STRATEGIES
        if arguments.generator == "cute-mixed"
        else parse_strategies(load_data(arguments.strategies))
    )
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
    routed_generator = (
        MutationFirstGenerator(
            CuteMutationGenerator(), ProgressKernelGenerator(generator)
        )
        if arguments.generator == "cute-mixed"
        else generator
    )
    return (
        strategies,
        routed_generator,
        arguments.model,
        _mcts_config(arguments, max_repairs=arguments.max_repairs),
    )


def _mcts_config(
    arguments: argparse.Namespace, *, max_repairs: int
) -> MCTSConfig:
    return MCTSConfig(
        c_puct=arguments.c_puct,
        k_max=arguments.k_max,
        max_depth=arguments.max_depth,
        max_repairs=max_repairs,
        max_infrastructure_retries=arguments.max_infrastructure_retries,
        include_incoming_profile_delta=arguments.include_incoming_profile_delta,
        profile_metric_set=arguments.profile_metric_set,
        measurement_drift_interval=arguments.measurement_drift_interval,
        measurement_drift_threshold=arguments.measurement_drift_threshold,
    )


if __name__ == "__main__":
    raise SystemExit(main())
