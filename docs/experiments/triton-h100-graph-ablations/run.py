"""Paired Triton H100 context and search-policy ablations on Nebius."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import signal
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from kernel_mcts.benchmarks import BF16_GEMM_WORKLOAD, load_triton_bf16_gemm_root
from kernel_mcts.config import load_data, parse_strategies
from kernel_mcts.generation import GenerationResult
from kernel_mcts.llm import DEFAULT_GENERATION_INSTRUCTIONS
from kernel_mcts.llm_generation import LLMKernelGenerator, build_generation_prompt
from kernel_mcts.messages_client import MessagesClient, MessagesConfig
from kernel_mcts.nebius import NebiusConfig, create_nebius_provider
from kernel_mcts.orchestration import run_mcts_search, run_sequential_search
from kernel_mcts.persistence import SQLiteTraceStore
from kernel_mcts.priors import UniformStrategyPrior
from kernel_mcts.provenance import capture_repository_state, image_digest_from_reference
from kernel_mcts.providers import HardwareSpec
from kernel_mcts.search import MCTSConfig
from kernel_mcts.sequential import SequentialConfig
from kernel_mcts.sequential_generation import (
    SEQUENTIAL_TRITON_INSTRUCTIONS, SEQUENTIAL_TRITON_PROMPT_SCHEMA_VERSION,
    SequentialKernelGenerator, build_sequential_prompt,
)
from kernel_mcts.strategy_applicability import applicable_strategies
from kernel_mcts.triton_freeform import pure_triton_freeform_strategy


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
IMAGE = (
    "docker.io/saarora/gpu-kernel-mcts@sha256:"
    "1e02c618f5a99b7f212522e0623131c3c26fb27c76834401b4a6bed54fad7874"
)
MODEL = "rits/zai-org/glm-5-3"
PROJECT_ID = "project-e00aymjmpr00grzcky46ba"
SUBNET_ID = "vpcsubnet-e00a30bhyztjvq7yaz"
CATALOG = ROOT / "configs/triton_h100_pure_local_v20.json"
PAIRED = runpy.run_path(str(HERE.parent / "triton-h100-freeform-compare/run_paired.py"))
ReusedWorkerProvider = PAIRED["ReusedWorkerProvider"]
ProgressWatchdog = PAIRED["ProgressWatchdog"]
ProgressTrace = PAIRED["ProgressTrace"]


class GatewayDeadlineExceeded(Exception):
    pass


@contextmanager
def request_deadline(seconds: float):
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)

    def on_alarm(signum, frame):
        raise GatewayDeadlineExceeded

    signal.signal(signal.SIGALRM, on_alarm)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous_handler)


class BoundedGatewayGenerator:
    """Count a transient gateway failure as one invalid generation call."""

    def __init__(self, inner, *, is_mcts: bool, strategies=()) -> None:
        self.inner = inner
        self.is_mcts = is_mcts
        self.strategies = strategies
        self.consecutive_failures = 0

    def generate(self, request):
        started = perf_counter()
        try:
            with request_deadline(120):
                result = self.inner.generate(request)
        except (RuntimeError, GatewayDeadlineExceeded) as error:
            message = (
                "Messages API hard deadline exceeded"
                if isinstance(error, GatewayDeadlineExceeded) else str(error)
            )
            if message not in {
                "Messages API returned HTTP 502", "Messages API returned HTTP 503",
                "Messages API returned HTTP 504", "Messages API transport failed",
                "Messages API hard deadline exceeded",
            }:
                raise
            self.consecutive_failures += 1
            if self.consecutive_failures >= 3:
                raise RuntimeError("Messages gateway had repeated transient failures") from None
            if self.is_mcts:
                prompt = build_generation_prompt(request)
            else:
                eligible = applicable_strategies(
                    self.strategies, request.parent, request.workload, request.hardware,
                ) if self.strategies else ()
                menu = tuple(
                    f"{item.id}: {item.description}; {item.prompt_for(request.parent.backend)}"
                    for item in eligible
                )
                prompt = build_sequential_prompt(request, menu)
            print(f"transient gateway failure: {message}", flush=True)
            return GenerationResult(
                generation_id=f"gateway-error-{uuid4()}", raw_output="", program=None,
                prompt_hash=hashlib.sha256(prompt.encode()).hexdigest(),
                prompt_text=prompt, latency_seconds=perf_counter() - started,
                metadata={
                    "generator": "llm", "llm_call": True,
                    "proposal_mechanism": "llm_generation", "model": MODEL,
                    "thinking_mode": "disabled", "gateway_error": message,
                },
            )
        self.consecutive_failures = 0
        return result


def arms_for_group(group: str) -> tuple[str, ...]:
    if group == "context":
        return tuple(f"mcts-n{n}-c{c}" for n in (0, 1) for c in (0, 1))
    if group == "behavior":
        return ("mcts-n0-c0", "mcts-strategy", "seq-free-h0", "seq-strategy-h0", "seq-free-h2")
    if group == "depth":
        return ("seq-free-h0",)
    raise ValueError(group)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=("context", "behavior", "depth"), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--label", default="")
    parser.add_argument("--output-dir", type=Path, default=HERE / "output")
    parser.add_argument("--stall-timeout", type=float, default=360)
    args = parser.parse_args()
    if args.budget < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("budget must be positive and seeds must be unique")
    if args.label and (not args.label.isalnum() or len(args.label) > 24):
        parser.error("label must be short and alphanumeric")
    base = os.environ.get("ENMAAS_BASE_URL", "").rstrip("/")
    if not base.startswith("https://") or not os.environ.get("ENMAAS_KEY"):
        parser.error("ENMAAS_BASE_URL and ENMAAS_KEY are required")
    url = base if base.endswith("/messages") else f"{base}/messages"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    repository = capture_repository_state(ROOT)
    root = load_triton_bf16_gemm_root()
    root_hash = hashlib.sha256(root.source.encode()).hexdigest()
    strategies = parse_strategies(load_data(CATALOG))
    catalog_hash = hashlib.sha256(CATALOG.read_bytes()).hexdigest()
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
        minimum_tool_versions={"triton": "3.9.0"}, required_profilers=("ncu",),
    )
    for seed in args.seeds:
        requested = arms_for_group(args.group)
        order = requested if seed % 2 else tuple(reversed(requested))
        pending = []
        for arm in order:
            prefix = f"{args.group}-" if args.group in {"behavior", "depth"} else ""
            suffix = f"-{args.label}" if args.label else ""
            path = args.output_dir / f"{prefix}{arm}-glm53-s{seed}-b{args.budget}{suffix}.sqlite"
            result_path = path.with_suffix(".json")
            if result_path.exists():
                print(f"skip completed {arm} seed={seed}", flush=True)
            elif path.exists():
                parser.error(f"incomplete trace exists; inspect before resuming: {path}")
            else:
                pending.append((arm, path))
        if not pending:
            continue
        print(f"starting seed={seed}, arms={[arm for arm, _ in pending]}", flush=True)
        with provider.worker_for_run(hardware) as worker:
            reused = ReusedWorkerProvider(worker)
            for arm, path in pending:
                is_mcts = arm.startswith("mcts")
                ncu = arm.endswith("n1-c0") or arm.endswith("n1-c1")
                compiler = arm.endswith("c1")
                strategy = "strategy" in arm
                history = 2 if arm.endswith("h2") else 0
                client = MessagesClient(MessagesConfig(
                    model=MODEL, url=url, max_output_tokens=16_384,
                    timeout_seconds=300, thinking_mode="disabled", history_turns=history,
                    instructions=(DEFAULT_GENERATION_INSTRUCTIONS if is_mcts else
                                  SEQUENTIAL_TRITON_INSTRUCTIONS),
                ))
                generator_impl = (
                    LLMKernelGenerator(client) if is_mcts else
                    SequentialKernelGenerator(
                        client,
                        strategy_catalog=strategies if strategy else (),
                        strategy_menu=tuple(
                            f"{item.id}: {item.description}; {item.prompt_for('triton_h100')}"
                            for item in strategies
                        ) if strategy else (),
                        catalog_sha256=catalog_hash if strategy else None,
                        instructions_text=SEQUENTIAL_TRITON_INSTRUCTIONS,
                        prompt_schema_version=SEQUENTIAL_TRITON_PROMPT_SCHEMA_VERSION,
                    )
                )
                generator = BoundedGatewayGenerator(
                    generator_impl, is_mcts=is_mcts,
                    strategies=strategies if strategy else (),
                )
                metadata = {
                    "comparison": "triton_h100_graph_ablations_v1",
                    "comparison_group": args.group,
                    "comparison_arm": arm,
                    "worker_image": IMAGE,
                    "worker_code_overlay": False,
                    "root_source_sha256": root_hash,
                    "messages_thinking": "disabled",
                    "messages_history_turns": history,
                    "messages_timeout_seconds": 300,
                    "search_stall_timeout_seconds": args.stall_timeout,
                    "messages_endpoint_sha256": hashlib.sha256(url.encode()).hexdigest(),
                    "git_commit": repository.commit,
                    "dirty_tree": repository.dirty,
                    "strategy_catalog_sha256": catalog_hash if strategy else None,
                    "ncu_prompt_visible": ncu,
                    "compiler_prompt_visible": compiler,
                    "profile_collection": "spill_v1",
                    "dataflow_prompt_view": "none",
                    "hardware_graph_prompt_view": "none",
                }
                print(f"starting seed={seed} arm={arm}", flush=True)
                with SQLiteTraceStore(path) as store:
                    with ProgressWatchdog(args.stall_timeout) as watchdog:
                        trace = ProgressTrace(store, watchdog)
                        if is_mcts:
                            execution = run_mcts_search(
                                provider=reused, hardware=hardware,
                                workload=BF16_GEMM_WORKLOAD, root_program=root,
                                generator=generator,
                                strategies=strategies if strategy else (pure_triton_freeform_strategy(),),
                                prior_provider=UniformStrategyPrior(),
                                generation_budget=args.budget, trace=trace,
                                mcts_config=MCTSConfig(
                                    max_repairs=1, k_max=3, max_depth=10,
                                    include_parent_profile=ncu,
                                    profile_prompt_view="ncu_only",
                                    include_compiler_summary=compiler,
                                    profile_metric_set="spill_v1",
                                ),
                                seed=seed, model_name=MODEL, run_metadata=metadata,
                            )
                        else:
                            execution = run_sequential_search(
                                provider=reused, hardware=hardware,
                                workload=BF16_GEMM_WORKLOAD, root_program=root,
                                generator=generator,
                                generation_budget=args.budget, trace=trace,
                                config=SequentialConfig(
                                    max_repairs=1, profile_metric_set="spill_v1",
                                    profile_visibility="none", compiler_detail="none",
                                    dataflow_view="none", dataflow_description="none",
                                ),
                                seed=seed, model_name=MODEL, run_metadata={
                                    **metadata,
                                    "sequential_strategy_choice": generator_impl.choice_provenance,
                                },
                                strategy_catalog=strategies if strategy else (),
                            )
                best = execution.result.best.evaluation
                assert best.program is not None and best.benchmark is not None
                source_path = path.with_name(path.stem + "-best.py")
                source_path.write_text(best.program.source, encoding="utf-8")
                result = {
                    "seed": seed, "arm": arm, "budget": args.budget,
                    "run_id": execution.run_id,
                    "generations": execution.result.generations,
                    "best_median_us": best.benchmark.median_us,
                    "trace": path.name, "best_source": source_path.name,
                }
                path.with_suffix(".json").write_text(
                    json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8",
                )
                print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
