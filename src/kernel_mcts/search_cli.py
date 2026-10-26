from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import os
import sys
from pathlib import Path
from typing import Mapping, TextIO

from .autotuning import TuningConfig
from .benchmarks import BF16_GEMM_WORKLOAD, load_bf16_gemm_root, load_triton_bf16_gemm_root
from .config import load_data, parse_strategies
from .cute_mutations import (
    CUTE_MUTATION_STRATEGIES,
    CUTE_MUTATION_STRATEGY_IDS,
    CuteMutationGenerator,
)
from .cute_generation import CuteTypedLLMGenerator
from .cute_program import (
    PinnedCuteGemmRenderer,
    REFERENCE_CUTE_GEMM,
    cute_gemm_program_from_source,
)
from .cute_independent import make_independent_cute_gemm
from .cute_independent_program import (
    IndependentCuteGemmRenderer,
    independent_cute_gemm_from_source,
)
from .dataflow_prompt import VIEWS as DATAFLOW_PROMPT_VIEWS
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
from .llm import LLMClient
from .messages_client import MessagesClient, MessagesConfig
from .nebius import NebiusConfig, create_nebius_provider
from .openai_client import OpenAIResponsesClient, OpenAIResponsesConfig
from .orchestration import run_mcts_search, run_sequential_search
from .persistence import SQLiteTraceStore
from .priors import UniformStrategyPrior
from .profiling import PROFILE_METRIC_SET_IDS
from .provenance import capture_repository_state, image_digest_from_reference
from .providers import HardwareSpec, RunPodConfig
from .runpod import create_runpod_provider
from .runpod_cli import ReadinessProgress
from .runpod_volume import resolve_reusable_volume
from .search import MCTSConfig
from .smoke import SmokeKernelGenerator, TritonSmokeKernelGenerator
from .sequential import SequentialConfig
from .sequential_generation import SEQUENTIAL_INSTRUCTIONS, SequentialKernelGenerator


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
        self._algorithm = "mcts"

    def start_run(
        self,
        run_id: str,
        benchmark_id: str,
        algorithm: str,
        config: Mapping[str, object],
    ) -> None:
        self._algorithm = algorithm
        self._trace.start_run(run_id, benchmark_id, algorithm, config)

    def store_compiler_artifact(self, sha256: str, content: bytes) -> None:
        self._trace.store_compiler_artifact(sha256, content)

    def get_node_dataflow(self, node_id: str) -> Mapping[str, object] | None:
        return self._trace.get_node_dataflow(node_id)

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
                f"{'MCTS' if self._algorithm == 'mcts' else 'Sequential'} completed: "
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
            f"Starting proposal call {self._calls}: strategy={request.strategy.id}"
        )
        result = self._generator.generate(request)
        self._write(f"Proposal call {self._calls} returned; evaluating proposal")
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
        description="Run MCTS or sequential CUDA refinement on an H100 worker."
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--algorithm", choices=("mcts", "sequential"), default="mcts")
    parser.add_argument(
        "--backend", choices=("cuda_cpp", "cute_dsl", "triton_h100"), default="cuda_cpp"
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
    parser.add_argument(
        "--cute-root-kind",
        choices=("pinned", "independent"),
        default="pinned",
    )
    parser.add_argument("--cute-root-tile-m", type=int)
    parser.add_argument("--cute-root-tile-n", type=int)
    parser.add_argument("--cute-root-cluster-m", type=int)
    parser.add_argument("--cute-root-cluster-n", type=int)
    parser.add_argument(
        "--cute-independent-root-mainloop-schedule",
        choices=("serial", "prefetch"),
        default="serial",
        help="choose the typed independent root schedule",
    )
    parser.add_argument(
        "--cute-strategy",
        action="append",
        choices=CUTE_MUTATION_STRATEGY_IDS,
        help=(
            "restrict a CuTe mutation or mixed run to this semantic strategy; "
            "repeat to select multiple strategies"
        ),
    )
    parser.add_argument("--autotune", action="store_true")
    parser.add_argument("--tuning-budget", type=int, default=20)
    parser.add_argument("--tuning-method", choices=("random", "grid"), default="random")
    parser.add_argument("--tuned-best-output", type=Path)
    parser.add_argument(
        "--generator",
        choices=("smoke", "openai", "messages", "cute-mutation", "cute-mixed"),
        default="smoke",
    )
    parser.add_argument("--model")
    parser.add_argument("--strategies", type=Path)
    parser.add_argument("--openai-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--messages-url", default=os.environ.get("ENMAAS_URL"))
    parser.add_argument("--messages-api-key-env", default="ENMAAS_KEY")
    parser.add_argument("--messages-api-version", default="2023-06-01")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--max-output-tokens", type=int, default=16_384)
    parser.add_argument("--llm-timeout", type=float, default=180.0)
    parser.add_argument("--max-repairs", type=int, default=1)
    parser.add_argument("--session-history-turns", type=int, default=2,
                        help="bounded prior text turns for the sequential session")
    parser.add_argument("--sequential-compiler-detail",
                        choices=("none", "resources", "full"), default="full")
    parser.add_argument("--sequential-profile-visibility",
                        choices=("none", "lightweight", "full"), default="full")
    parser.add_argument("--sequential-dataflow-description",
                        choices=("none", "limits"), default="limits")
    parser.add_argument("--sequential-dataflow-view",
                        choices=DATAFLOW_PROMPT_VIEWS, default="paths")
    parser.add_argument("--sequential-strategy-menu", action="store_true",
                        help="show the fixed --strategies catalog as an optional ablation")
    parser.add_argument("--sequential-best-so-far", action="store_true",
                        help="include the best verified source when the current source differs")
    parser.add_argument("--sequential-hardware-graph", action="store_true",
                        help="include a bounded H100 resource graph in sequential prompts")
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
    parser.add_argument(
        "--omit-parent-profile",
        action="store_true",
        help="hide the parent NCU profile from generation prompts while still collecting it",
    )
    parser.add_argument(
        "--include-compiler-summary",
        action="store_true",
        help="include bounded parent ptxas resources and static SASS site counts in CUDA prompts",
    )
    parser.add_argument(
        "--dataflow-prompt-view",
        choices=DATAFLOW_PROMPT_VIEWS,
        default="none",
        help="include an opt-in PTX dataflow view in CUDA prompts",
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
    if arguments.omit_parent_profile and arguments.include_incoming_profile_delta:
        parser.error("--omit-parent-profile conflicts with --include-incoming-profile-delta")
    if arguments.include_compiler_summary and (arguments.generator not in {"openai", "messages"} or arguments.backend != "cuda_cpp"):
        parser.error("--include-compiler-summary requires a CUDA LLM generator")
    if arguments.dataflow_prompt_view != "none" and (arguments.generator not in {"openai", "messages"} or arguments.backend != "cuda_cpp"):
        parser.error("--dataflow-prompt-view requires a CUDA LLM generator")
    if arguments.backend == "triton_h100":
        if arguments.algorithm != "mcts" or arguments.generator not in {"openai", "messages", "smoke"}:
            parser.error("Triton H100 currently requires MCTS with an LLM or smoke generator")
        if arguments.include_compiler_summary or arguments.dataflow_prompt_view != "none":
            parser.error("CUDA compiler/dataflow prompt views are unsupported for Triton H100")
        if arguments.autotune:
            parser.error("CUDA autotuning is unsupported for Triton H100")
        if arguments.profile_metric_set != "lightweight_v1":
            parser.error("Triton H100 currently supports --profile-metric-set=lightweight_v1")
    if arguments.best_output is not None and arguments.best_output.exists():
        parser.error(
            f"refusing to overwrite existing best output: {arguments.best_output}"
        )
    if arguments.tuned_best_output is not None and not arguments.autotune:
        parser.error("--tuned-best-output requires --autotune")
    if arguments.autotune and arguments.tuning_budget < 1:
        parser.error("--tuning-budget must be positive")
    if (
        arguments.backend == "cute_dsl"
        and arguments.autotune
        and arguments.cute_root_kind != "independent"
    ):
        parser.error(
            "post-search CuTe autotuning requires --cute-root-kind=independent"
        )
    if arguments.backend != "cute_dsl" and arguments.cute_root_kind != "pinned":
        parser.error("--cute-root-kind requires --backend=cute_dsl")
    if arguments.tuned_best_output is not None and arguments.tuned_best_output.exists():
        parser.error(
            f"refusing to overwrite existing tuned output: {arguments.tuned_best_output}"
        )
    if arguments.algorithm == "sequential":
        generator, model_name, sequential_config = _sequential_components(parser, arguments)
    else:
        strategies, generator, model_name, mcts_config = _search_components(
            parser, arguments
        )
    root_program = _root_program(parser, arguments)
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
    if arguments.generator == "messages":
        provenance["model_provider"] = "anthropic_messages_compatible"
        provenance["messages_endpoint_sha256"] = hashlib.sha256(
            arguments.messages_url.encode("utf-8")
        ).hexdigest()
        provenance["messages_api_version"] = arguments.messages_api_version
    if arguments.algorithm == "sequential":
        assert isinstance(generator, SequentialKernelGenerator)
        provenance["sequential_strategy_choice"] = generator.choice_provenance
    if arguments.backend == "cute_dsl":
        representation = (
            independent_cute_gemm_from_source(root_program.source)
            if arguments.cute_root_kind == "independent"
            else cute_gemm_program_from_source(root_program.source)
        )
        provenance["cute_root_kind"] = arguments.cute_root_kind
        provenance["cute_root_representation"] = representation.as_dict()
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
                cute_root_kind=arguments.cute_root_kind,
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
                cute_root_kind=arguments.cute_root_kind,
            ),
            readiness_progress=progress,
        )
    try:
        with SQLiteTraceStore(arguments.trace) as trace:
            reporting_trace = SearchCLIProgress(trace, progress)
            shared = dict(
                provider=provider,
                hardware=HardwareSpec(
                    "H100", form_factor="SXM",
                    minimum_compute_capability="9.0", required_profilers=("ncu",),
                ),
                workload=BF16_GEMM_WORKLOAD, root_program=root_program,
                generator=ProgressKernelGenerator(generator),
                generation_budget=arguments.generation_budget,
                trace=reporting_trace, seed=arguments.seed,
                run_id=arguments.run_id, model_name=model_name,
                run_metadata=provenance,
            )
            if arguments.algorithm == "sequential":
                execution = run_sequential_search(**shared, config=sequential_config)
            else:
                execution = run_mcts_search(
                    **shared, strategies=strategies, prior_provider=UniformStrategyPrior(),
                    mutation_budget=arguments.mutation_budget, mcts_config=mcts_config,
                    tuning_config=(
                        TuningConfig(arguments.tuning_budget, arguments.tuning_method, arguments.seed)
                        if arguments.autotune else None
                    ),
                )
    finally:
        progress.finish()

    result = execution.result
    if arguments.best_output is not None:
        best_program = (
            result.best.evaluation.program
            if arguments.algorithm == "sequential" else result.best.program
        )
        assert best_program is not None
        arguments.best_output.write_text(best_program.source, encoding="utf-8")
    if (
        arguments.algorithm == "mcts"
        and
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
        "messages": "Messages",
        "smoke": "Smoke",
        "cute-mutation": "CuTe mutation",
        "cute-mixed": "CuTe mixed",
    }[arguments.generator]
    if arguments.algorithm == "sequential":
        print(
            f"Sequential search completed: run_id={execution.run_id}, "
            f"iterations={result.iterations}, B_gen={result.generations}, "
            f"profiles={result.profile_calls}, accepted_depth={result.accepted_depth}, "
            f"nodes={len(result.nodes)}, best_reward={result.best.evaluation.reward:.6g}"
        )
        print(f"SQLite trace: {arguments.trace.resolve()}")
        if arguments.best_output is not None:
            print(f"Best kernel: {arguments.best_output.resolve()}")
        return 0
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


def _root_program(
    parser: argparse.ArgumentParser,
    arguments: argparse.Namespace,
):
    values = (
        arguments.cute_root_tile_m,
        arguments.cute_root_tile_n,
        arguments.cute_root_cluster_m,
        arguments.cute_root_cluster_n,
    )
    if arguments.cute_root_kind == "independent":
        if any(value is not None for value in values):
            parser.error("--cute-root-* schedule arguments require the pinned root")
        return IndependentCuteGemmRenderer().render(
            make_independent_cute_gemm(
                mainloop_schedule=(
                    arguments.cute_independent_root_mainloop_schedule
                )
            )
        )
    if arguments.cute_independent_root_mainloop_schedule != "serial":
        parser.error(
            "--cute-independent-root-mainloop-schedule requires the independent root"
        )
    if any(value is not None for value in values):
        if arguments.backend != "cute_dsl":
            parser.error("CuTe root schedule arguments require --backend=cute_dsl")
        if any(value is None for value in values):
            parser.error(
                "all four --cute-root-* schedule arguments must be provided together"
            )
        representation = replace(
            REFERENCE_CUTE_GEMM,
            tile_m=arguments.cute_root_tile_m,
            tile_n=arguments.cute_root_tile_n,
            cluster_m=arguments.cute_root_cluster_m,
            cluster_n=arguments.cute_root_cluster_n,
        )
        try:
            return PinnedCuteGemmRenderer().render(representation)
        except ValueError as error:
            parser.error(str(error))
    if arguments.backend == "cute_dsl":
        return PinnedCuteGemmRenderer().render(REFERENCE_CUTE_GEMM)
    if arguments.backend == "triton_h100":
        return load_triton_bf16_gemm_root()
    return load_bf16_gemm_root()


def _validate_llm_connection(
    parser: argparse.ArgumentParser, arguments: argparse.Namespace,
) -> None:
    if arguments.generator == "messages":
        if not arguments.messages_url:
            parser.error("--messages-url or ENMAAS_URL is required with --generator=messages")
        key_env = arguments.messages_api_key_env
    else:
        key_env = arguments.openai_api_key_env
    if not os.environ.get(key_env):
        parser.error(f"environment variable {key_env!r} is not set")


def _cuda_llm_client(
    arguments: argparse.Namespace, *, instructions: str | None = None,
    history_turns: int = 0,
) -> LLMClient:
    if arguments.generator == "messages":
        values = {
            "model": arguments.model,
            "url": arguments.messages_url,
            "max_output_tokens": arguments.max_output_tokens,
            "timeout_seconds": arguments.llm_timeout,
            "api_key_env": arguments.messages_api_key_env,
            "api_version": arguments.messages_api_version,
            "history_turns": history_turns,
        }
        if instructions is not None:
            values["instructions"] = instructions
        return MessagesClient(MessagesConfig(**values))
    values = {
        "model": arguments.model,
        "reasoning_effort": arguments.reasoning_effort,
        "max_output_tokens": arguments.max_output_tokens,
        "timeout_seconds": arguments.llm_timeout,
        "api_key_env": arguments.openai_api_key_env,
        "store": False,
        "history_turns": history_turns,
    }
    if instructions is not None:
        values["instructions"] = instructions
    return OpenAIResponsesClient(OpenAIResponsesConfig(**values))


def _sequential_components(
    parser: argparse.ArgumentParser, arguments: argparse.Namespace,
) -> tuple[KernelGenerator, str, SequentialConfig]:
    if arguments.backend != "cuda_cpp" or arguments.generator not in {"openai", "messages"}:
        parser.error("sequential search requires --backend=cuda_cpp and a CUDA LLM generator")
    if arguments.generation_budget < 1 or arguments.mutation_budget:
        parser.error("sequential search requires a positive generation budget and no mutation budget")
    if not arguments.model:
        parser.error("--model is required for sequential search")
    if arguments.session_history_turns < 0:
        parser.error("--session-history-turns cannot be negative")
    if arguments.autotune or arguments.tuned_best_output is not None:
        parser.error("post-search autotuning is not part of the sequential experiment")
    if arguments.sequential_strategy_menu and arguments.strategies is None:
        parser.error("--sequential-strategy-menu requires --strategies")
    if (
        arguments.omit_parent_profile or arguments.include_incoming_profile_delta
        or arguments.include_compiler_summary or arguments.dataflow_prompt_view != "none"
    ):
        parser.error("use --sequential-* context flags with sequential search")
    _validate_llm_connection(parser, arguments)
    menu: tuple[str, ...] = ()
    strategy_ids: tuple[str, ...] = ()
    catalog_sha256: str | None = None
    if arguments.sequential_strategy_menu:
        catalog_sha256 = hashlib.sha256(arguments.strategies.read_bytes()).hexdigest()
        catalog = parse_strategies(load_data(arguments.strategies))
        if not catalog:
            parser.error("strategy menu must contain at least one strategy")
        strategy_ids = tuple(item.id for item in catalog)
        if len(set(strategy_ids)) != len(strategy_ids) or "freeform" in strategy_ids:
            parser.error("strategy menu IDs must be unique and cannot use freeform")
        menu = tuple(
            f"{item.id}: {item.description}; {item.prompt_for('cuda_cpp')}"
            for item in catalog
        )
    client = _cuda_llm_client(
        arguments, instructions=SEQUENTIAL_INSTRUCTIONS,
        history_turns=arguments.session_history_turns,
    )
    generator = SequentialKernelGenerator(
        client, strategy_menu=menu, strategy_ids=strategy_ids,
        catalog_sha256=catalog_sha256,
    )
    config = SequentialConfig(
        max_repairs=arguments.max_repairs,
        max_infrastructure_retries=arguments.max_infrastructure_retries,
        profile_metric_set=arguments.profile_metric_set,
        profile_visibility=arguments.sequential_profile_visibility,
        compiler_detail=arguments.sequential_compiler_detail,
        dataflow_view=arguments.sequential_dataflow_view,
        dataflow_description=arguments.sequential_dataflow_description,
        include_best_so_far=arguments.sequential_best_so_far,
        include_hardware_graph=arguments.sequential_hardware_graph,
    )
    return generator, arguments.model, config


def _search_components(
    parser: argparse.ArgumentParser,
    arguments: argparse.Namespace,
) -> tuple[tuple[Strategy, ...], KernelGenerator, str | None, MCTSConfig]:
    if arguments.generator == "smoke":
        if arguments.cute_strategy:
            parser.error("--cute-strategy requires a CuTe generator")
        if arguments.backend not in {"cuda_cpp", "triton_h100"}:
            parser.error("--generator=smoke requires --backend=cuda_cpp or triton_h100")
        if arguments.generation_budget != 1:
            parser.error("the deterministic smoke run requires --generation-budget=1")
        if arguments.mutation_budget != 0:
            parser.error("the deterministic smoke run requires --mutation-budget=0")
        return (
            (
                Strategy(
                    "deterministic_smoke",
                    "Generate the fixed orchestration smoke candidate",
                    {arguments.backend: "fixed smoke candidate"},
                ),
            ),
            TritonSmokeKernelGenerator() if arguments.backend == "triton_h100" else SmokeKernelGenerator(),
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
                include_parent_profile=not arguments.omit_parent_profile,
                include_compiler_summary=arguments.include_compiler_summary,
                dataflow_prompt_view=arguments.dataflow_prompt_view,
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
            _selected_cute_strategies(parser, arguments),
            CuteMutationGenerator(),
            None,
            _mcts_config(arguments, max_repairs=0),
        )

    if arguments.generator in {"openai", "messages"} and arguments.backend not in {"cuda_cpp", "triton_h100"}:
        parser.error("LLM generators require --backend=cuda_cpp or triton_h100")
    if arguments.generator in {"openai", "messages"} and arguments.cute_strategy:
        parser.error("--cute-strategy requires a CuTe generator")
    if arguments.generator == "cute-mixed" and arguments.backend != "cute_dsl":
        parser.error("--generator=cute-mixed requires --backend=cute_dsl")
    if arguments.generation_budget < 1:
        parser.error("LLM-backed search requires a positive --generation-budget")
    if arguments.generator == "cute-mixed" and arguments.mutation_budget < 1:
        parser.error("the mixed CuTe run requires a positive --mutation-budget")
    if not arguments.model:
        parser.error(f"--model is required with --generator={arguments.generator}")
    if arguments.generator in {"openai", "messages"} and arguments.strategies is None:
        parser.error("--strategies is required with a CUDA LLM generator")
    _validate_llm_connection(parser, arguments)
    strategies = (
        _selected_cute_strategies(parser, arguments)
        if arguments.generator == "cute-mixed"
        else parse_strategies(load_data(arguments.strategies))
    )
    if not strategies:
        parser.error("strategy configuration must contain at least one strategy")
    generator = LLMKernelGenerator(_cuda_llm_client(arguments))
    routed_generator = (
        MutationFirstGenerator(
            CuteMutationGenerator(),
            ProgressKernelGenerator(CuteTypedLLMGenerator(generator)),
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


def _selected_cute_strategies(
    parser: argparse.ArgumentParser,
    arguments: argparse.Namespace,
) -> tuple[Strategy, ...]:
    selected = arguments.cute_strategy
    if arguments.cute_root_kind == "independent":
        supported = {
            "change_cta_tile",
            "change_cluster_shape",
            "change_pipeline_stages",
            "change_mainloop_schedule",
            "change_producer_consumer_specialization",
            "change_shared_memory_swizzle",
        }
        if selected and not set(selected) <= supported:
            parser.error(
                "the independent CuTe root currently supports only CTA-tile, "
                "cluster-shape, pipeline-stage, mainloop-schedule, producer/consumer "
                "specialization, and shared-memory-swizzle changes"
            )
        selected = selected or tuple(supported)
    if not selected:
        return CUTE_MUTATION_STRATEGIES
    selected_ids = set(selected)
    return tuple(
        strategy
        for strategy in CUTE_MUTATION_STRATEGIES
        if strategy.id in selected_ids
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
        include_parent_profile=not arguments.omit_parent_profile,
        include_compiler_summary=arguments.include_compiler_summary,
        dataflow_prompt_view=arguments.dataflow_prompt_view,
        profile_metric_set=arguments.profile_metric_set,
        measurement_drift_interval=arguments.measurement_drift_interval,
        measurement_drift_threshold=arguments.measurement_drift_threshold,
    )


if __name__ == "__main__":
    raise SystemExit(main())
