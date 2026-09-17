from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Mapping, TextIO

from .autotuning import TuningConfig, parse_tuning_parameters
from .benchmarks import BF16_GEMM_WORKLOAD
from .domain import KernelProgram
from .nebius import NebiusConfig, create_nebius_provider
from .orchestration import run_standalone_autotuning
from .persistence import SQLiteTraceStore
from .provenance import capture_repository_state, image_digest_from_reference
from .providers import HardwareSpec
from .runpod_cli import ReadinessProgress


class AutotuneCLIProgress:
    def __init__(
        self,
        trace: SQLiteTraceStore,
        readiness: ReadinessProgress,
        stream: TextIO = sys.stdout,
    ) -> None:
        self._trace = trace
        self._readiness = readiness
        self._stream = stream

    def start_run(self, *args) -> None:
        self._trace.start_run(*args)

    def emit(self, event_type: str, payload: Mapping[str, object]) -> None:
        self._trace.emit(event_type, payload)
        if event_type == "environment_manifest":
            self._readiness.finish()
            self._write("Worker ready; evaluating tuning baseline")
        elif event_type == "tuning_started":
            self._write(
                f"Starting autotuning: method={payload.get('method')}, "
                f"budget={payload.get('budget')}"
            )
        elif event_type == "tuning_trial":
            evaluation = payload.get("evaluation")
            status = evaluation.get("status") if isinstance(evaluation, Mapping) else None
            reward = evaluation.get("reward") if isinstance(evaluation, Mapping) else None
            self._write(
                f"Tuning trial: B_tune={payload.get('b_tune')}, "
                f"status={status}, reward={reward}"
            )

    def _write(self, message: str) -> None:
        self._stream.write(f"{message}\n")
        self._stream.flush()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Autotune one annotated CUDA BF16 GEMM on a Nebius H100 SXM."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--tuning-budget", type=int, default=20)
    parser.add_argument("--tuning-method", choices=("random", "grid"), default="random")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--run-id")
    parser.add_argument("--nebius-project-id", required=True)
    parser.add_argument("--nebius-subnet-id", required=True)
    parser.add_argument("--nebius-username", required=True)
    parser.add_argument("--nebius-ssh-public-key", type=Path, required=True)
    parser.add_argument("--nebius-ssh-private-key", type=Path, required=True)
    parser.add_argument("--nebius-platform", default="gpu-h100-sxm")
    parser.add_argument("--nebius-preset", default="1gpu-16vcpu-200gb")
    parser.add_argument("--confirm-create-and-terminate", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if not arguments.confirm_create_and_terminate:
        parser.error("--confirm-create-and-terminate is required")
    if arguments.tuning_budget < 1:
        parser.error("--tuning-budget must be positive")
    for path, label in (
        (arguments.output, "output"),
        (arguments.trace, "trace"),
    ):
        if path.exists():
            parser.error(f"refusing to overwrite existing {label}: {path}")
    if not arguments.input.is_file():
        parser.error(f"input kernel does not exist: {arguments.input}")
    program = KernelProgram(arguments.input.read_text(encoding="utf-8"))
    try:
        parameters = parse_tuning_parameters(program)
    except ValueError as error:
        parser.error(str(error))
    if not parameters:
        parser.error("input kernel has no KERNEL_MCTS_TUNE annotations")

    repository = capture_repository_state(Path(__file__).resolve().parents[2])
    image_digest = image_digest_from_reference(arguments.image)
    progress = ReadinessProgress("autotuning worker")
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
        ),
        readiness_progress=progress,
    )
    metadata: dict[str, object] = {"worker_image": arguments.image}
    if repository.commit is not None:
        metadata["git_commit"] = repository.commit
    if repository.dirty is not None:
        metadata["dirty_tree"] = repository.dirty
    if image_digest is not None:
        metadata["worker_image_digest"] = image_digest
    try:
        with SQLiteTraceStore(arguments.trace) as trace:
            reporting_trace = AutotuneCLIProgress(trace, progress)
            execution = run_standalone_autotuning(
                provider=provider,
                hardware=HardwareSpec(
                    "H100",
                    form_factor="SXM",
                    minimum_compute_capability="9.0",
                ),
                workload=BF16_GEMM_WORKLOAD,
                program=program,
                tuning_config=TuningConfig(
                    arguments.tuning_budget,
                    arguments.tuning_method,
                    arguments.seed,
                ),
                trace=reporting_trace,
                run_id=arguments.run_id,
                run_metadata=metadata,
            )
    finally:
        progress.finish()

    best = execution.tuning.best
    if best is None or best.program is None:
        raise RuntimeError("autotuning produced no valid configuration")
    arguments.output.write_text(best.program.source, encoding="utf-8")
    print(
        f"Autotuning completed: run_id={execution.run_id}, "
        f"B_tune={execution.tuning.used}, baseline_reward={execution.baseline.reward}, "
        f"best_reward={best.reward}, improved={execution.tuning.improved}"
    )
    print(f"SQLite trace: {arguments.trace.resolve()}")
    print(f"Tuned kernel: {arguments.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
