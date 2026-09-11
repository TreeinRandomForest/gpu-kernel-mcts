from __future__ import annotations

import argparse
import os
from pathlib import Path

from .benchmarks import BF16_GEMM_WORKLOAD, load_bf16_gemm_root
from .domain import Strategy
from .orchestration import run_mcts_search
from .persistence import SQLiteTraceStore
from .priors import UniformStrategyPrior
from .providers import HardwareSpec, RunPodConfig
from .runpod import create_runpod_provider
from .runpod_cli import ReadinessProgress
from .runpod_volume import resolve_reusable_volume
from .search import MCTSConfig
from .smoke import SmokeKernelGenerator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one deterministic MCTS smoke iteration on an H100 worker."
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--api-key-env", default="RUNPOD_API_KEY")
    parser.add_argument("--gpu-type", default="NVIDIA H100 80GB HBM3")
    parser.add_argument("--network-volume-id")
    parser.add_argument("--data-center-id")
    parser.add_argument("--auto-volume", action="store_true")
    parser.add_argument("--preferred-data-center-id")
    parser.add_argument("--volume-name", default="gpu-kernel-mcts")
    parser.add_argument("--generation-budget", type=int, default=1)
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
    if arguments.generation_budget != 1:
        parser.error("the deterministic smoke run requires --generation-budget=1")
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
            data_center_ids=(data_center_id,),
        ),
        readiness_progress=progress,
    )
    try:
        with SQLiteTraceStore(arguments.trace) as trace:
            execution = run_mcts_search(
                provider=provider,
                hardware=HardwareSpec(
                    "H100",
                    form_factor="SXM",
                    minimum_compute_capability="9.0",
                ),
                workload=BF16_GEMM_WORKLOAD,
                root_program=load_bf16_gemm_root(),
                strategies=(
                    Strategy(
                        "deterministic_smoke",
                        "Generate the fixed orchestration smoke candidate",
                        {"cuda_cpp": "fixed smoke candidate"},
                    ),
                ),
                generator=SmokeKernelGenerator(),
                prior_provider=UniformStrategyPrior(),
                generation_budget=1,
                trace=trace,
                mcts_config=MCTSConfig(k_max=1, max_repairs=0),
                seed=arguments.seed,
                run_id=arguments.run_id,
            )
    finally:
        progress.finish()

    result = execution.result
    print(
        f"Smoke search completed: run_id={execution.run_id}, "
        f"iterations={result.iterations}, B_gen={result.generations}, "
        f"nodes={len(result.nodes)}, best_reward={result.best.reward:.6g}"
    )
    print(f"SQLite trace: {arguments.trace.resolve()}")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
