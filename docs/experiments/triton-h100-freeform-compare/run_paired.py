"""Run Triton H100 search arms on one Nebius worker per seed."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
from pathlib import Path

from kernel_mcts.benchmarks import BF16_GEMM_WORKLOAD, load_triton_bf16_gemm_root
from kernel_mcts.config import load_data, parse_strategies
from kernel_mcts.llm import DEFAULT_GENERATION_INSTRUCTIONS
from kernel_mcts.llm_generation import LLMKernelGenerator
from kernel_mcts.messages_client import MessagesClient, MessagesConfig
from kernel_mcts.nebius import NebiusConfig, create_nebius_provider
from kernel_mcts.nebius_overlay import overlay_nebius_worker, package_bundle
from kernel_mcts.orchestration import run_mcts_search, run_sequential_search
from kernel_mcts.persistence import SQLiteTraceStore
from kernel_mcts.priors import UniformStrategyPrior
from kernel_mcts.provenance import capture_repository_state, image_digest_from_reference
from kernel_mcts.providers import HardwareSpec
from kernel_mcts.search import MCTSConfig
from kernel_mcts.sequential import SequentialConfig
from kernel_mcts.sequential_generation import (
    SEQUENTIAL_TRITON_INSTRUCTIONS, SEQUENTIAL_TRITON_PROMPT_SCHEMA_VERSION,
    SequentialKernelGenerator,
)
from kernel_mcts.triton_freeform import pure_triton_freeform_strategy


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUTPUT = HERE / "output"
IMAGE = (
    "docker.io/saarora/gpu-kernel-mcts@sha256:"
    "22eb914d5041a988653d9d19a298d6bdb64f178f175563743d0af4591cd75ee5"
)
MODEL = "rits/zai-org/glm-5-3"
PROJECT_ID = "project-e00aymjmpr00grzcky46ba"
SUBNET_ID = "vpcsubnet-e00a30bhyztjvq7yaz"
COMBINED_CATALOG = ROOT / "configs/triton_h100_combined_v20.json"


class ReusedWorkerProvider:
    def __init__(self, worker) -> None:
        self.worker = worker

    def acquire_worker(self, hardware):
        return self.worker

    def release_worker(self, worker) -> None:
        if worker is not self.worker:
            raise ValueError("unexpected worker")


class RunStalled(BaseException):
    pass


class ProgressWatchdog:
    """Bound wall time without a trace event, including worker HTTP waits."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self.expired = False

    def __enter__(self):
        self.previous_handler = signal.getsignal(signal.SIGALRM)
        self.previous_timer = signal.getitimer(signal.ITIMER_REAL)
        signal.signal(signal.SIGALRM, self._expire)
        self.reset()
        return self

    def reset(self) -> None:
        if not self.expired:
            signal.setitimer(signal.ITIMER_REAL, self.seconds)

    def _expire(self, signum, frame) -> None:
        self.expired = True
        signal.setitimer(signal.ITIMER_REAL, 0)
        raise RunStalled(f"no search event for {self.seconds:g} seconds")

    def __exit__(self, exc_type, exc, traceback) -> None:
        signal.setitimer(signal.ITIMER_REAL, *self.previous_timer)
        signal.signal(signal.SIGALRM, self.previous_handler)


class ProgressTrace:
    def __init__(self, store: SQLiteTraceStore, watchdog: ProgressWatchdog) -> None:
        self.store = store
        self.watchdog = watchdog

    def __getattr__(self, name):
        return getattr(self.store, name)

    def emit(self, event_type, payload) -> None:
        self.store.emit(event_type, payload)
        self.watchdog.reset()
        if event_type == "generation":
            print(
                f"B_gen={payload.get('b_gen')} status={payload.get('proposal_status')} "
                f"reason={payload.get('invalid_reason')}", flush=True,
            )
        elif event_type == "new_global_best":
            print(f"new best reward={payload.get('reward')}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--budget", type=int, default=25)
    parser.add_argument("--arms", nargs="+", choices=("seq", "mcts", "combined"))
    parser.add_argument("--label", default="")
    parser.add_argument("--stall-timeout", type=float, default=360)
    arguments = parser.parse_args()
    if (arguments.budget < 1 or arguments.stall_timeout <= 0
            or len(set(arguments.seeds)) != len(arguments.seeds)):
        parser.error("budget and stall timeout must be positive; seeds must be unique")
    if arguments.label and (not arguments.label.isalnum() or len(arguments.label) > 24):
        parser.error("label must be a short alphanumeric value")
    if arguments.arms and len(set(arguments.arms)) != len(arguments.arms):
        parser.error("arms must be unique")
    base = os.environ.get("ENMAAS_BASE_URL", "").rstrip("/")
    if not base.startswith("https://") or not os.environ.get("ENMAAS_KEY"):
        parser.error("ENMAAS_BASE_URL and ENMAAS_KEY are required")
    url = base if base.endswith("/messages") else f"{base}/messages"
    OUTPUT.mkdir(exist_ok=True)
    repository = capture_repository_state(ROOT)
    bundle = package_bundle(ROOT)
    bundle_hash = hashlib.sha256(bundle).hexdigest()
    root = load_triton_bf16_gemm_root()
    root_hash = hashlib.sha256(root.source.encode()).hexdigest()
    combined_strategies = parse_strategies(load_data(COMBINED_CATALOG))
    combined_catalog_hash = hashlib.sha256(COMBINED_CATALOG.read_bytes()).hexdigest()
    config = NebiusConfig(
        image=IMAGE, project_id=PROJECT_ID, subnet_id=SUBNET_ID,
        ssh_public_key=Path.home() / ".ssh/nebius.pub",
        ssh_private_key=Path.home() / ".ssh/nebius", username="sanjay",
        startup_timeout_seconds=1200, backend="triton_h100",
        project_git_commit=repository.commit, project_dirty_tree=repository.dirty,
        container_digest=image_digest_from_reference(IMAGE),
    )
    provider = create_nebius_provider(
        config, readiness_progress=lambda stage, elapsed: print(
            f"worker {stage}: {elapsed:.1f}s", flush=True,
        ),
    )
    hardware = HardwareSpec(
        "H100", form_factor="SXM", minimum_compute_capability="9.0",
        minimum_tool_versions={"triton": "3.9.0"},
        required_profilers=("ncu",),
    )
    for seed in arguments.seeds:
        order = tuple(arguments.arms) if arguments.arms else (
            ("seq", "mcts") if seed % 2 else ("mcts", "seq")
        )
        suffix = f"-{arguments.label}" if arguments.label else ""
        paths = {
            arm: OUTPUT / f"{arm}-glm53-s{seed}-b{arguments.budget}{suffix}.sqlite"
            for arm in order
        }
        for path in paths.values():
            if path.exists():
                parser.error(f"refusing to overwrite trace: {path}")
        print(f"starting seed={seed}, order={order}", flush=True)
        with provider.worker_for_run(hardware) as worker:
            overlay_nebius_worker(worker, config.ssh_private_key, config.username, bundle)
            reused = ReusedWorkerProvider(worker)
            for arm in order:
                client = MessagesClient(MessagesConfig(
                    model=MODEL, url=url, max_output_tokens=16_384,
                    timeout_seconds=300, thinking_mode="disabled", history_turns=0,
                    instructions=(
                        SEQUENTIAL_TRITON_INSTRUCTIONS if arm == "seq" else
                        DEFAULT_GENERATION_INSTRUCTIONS
                    ),
                ))
                if arm == "seq":
                    generator = SequentialKernelGenerator(
                        client, instructions_text=SEQUENTIAL_TRITON_INSTRUCTIONS,
                        prompt_schema_version=SEQUENTIAL_TRITON_PROMPT_SCHEMA_VERSION,
                    )
                else:
                    generator = LLMKernelGenerator(client)
                metadata = {
                    "comparison": "triton_h100_search_b25_v1",
                    "comparison_arm": arm,
                    "worker_image": IMAGE,
                    "worker_code_overlay_sha256": bundle_hash,
                    "root_source_sha256": root_hash,
                    "messages_thinking": "disabled",
                    "messages_timeout_seconds": 300,
                    "search_stall_timeout_seconds": arguments.stall_timeout,
                    "messages_endpoint_sha256": hashlib.sha256(url.encode()).hexdigest(),
                    "git_commit": repository.commit,
                    "dirty_tree": repository.dirty,
                }
                print(f"starting seed={seed} arm={arm}", flush=True)
                with SQLiteTraceStore(paths[arm]) as store:
                    with ProgressWatchdog(arguments.stall_timeout) as watchdog:
                        trace = ProgressTrace(store, watchdog)
                        if arm in {"mcts", "combined"}:
                            execution = run_mcts_search(
                                provider=reused, hardware=hardware, workload=BF16_GEMM_WORKLOAD,
                                root_program=root, generator=generator,
                                strategies=(
                                    combined_strategies if arm == "combined" else
                                    (pure_triton_freeform_strategy(),)
                                ),
                                prior_provider=UniformStrategyPrior(),
                                generation_budget=arguments.budget, trace=trace,
                                mcts_config=MCTSConfig(
                                    max_repairs=1, k_max=3, max_depth=10,
                                    include_parent_profile=False,
                                    profile_metric_set="lightweight_v1",
                                ),
                                seed=seed, model_name=MODEL,
                                run_metadata={
                                    **metadata,
                                    "triton_action_policy": (
                                        "combined_v20" if arm == "combined" else "freeform_only_v1"
                                    ),
                                    **(
                                        {"strategy_catalog_sha256": combined_catalog_hash}
                                        if arm == "combined" else {}
                                    ),
                                },
                            )
                        else:
                            execution = run_sequential_search(
                                provider=reused, hardware=hardware, workload=BF16_GEMM_WORKLOAD,
                                root_program=root, generator=generator,
                                generation_budget=arguments.budget, trace=trace,
                                config=SequentialConfig(
                                    max_repairs=1, profile_metric_set="lightweight_v1",
                                    profile_visibility="none", compiler_detail="none",
                                    dataflow_view="none", dataflow_description="none",
                                ),
                                seed=seed, model_name=MODEL,
                                run_metadata={
                                    **metadata,
                                    "sequential_strategy_choice": generator.choice_provenance,
                                },
                            )
                best = execution.result.best.evaluation
                assert best.program is not None and best.benchmark is not None
                source_path = paths[arm].with_name(paths[arm].stem + "-best.py")
                source_path.write_text(best.program.source, encoding="utf-8")
                result = {
                    "seed": seed, "arm": arm, "budget": arguments.budget,
                    "run_id": execution.run_id,
                    "generations": execution.result.generations,
                    "best_median_us": best.benchmark.median_us,
                    "trace": paths[arm].name,
                    "best_source": source_path.name,
                }
                paths[arm].with_suffix(".json").write_text(
                    json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8",
                )
                print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
