from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from .benchmarks import BF16_GEMM_WORKLOAD, load_bf16_gemm_root
from .domain import EvaluationResult, KernelProgram, ProposalStatus
from .orchestration import run_candidate_evaluation
from .providers import HardwareSpec, RunPodConfig
from .runpod import create_runpod_provider
from .runpod_cli import ReadinessProgress
from .runpod_volume import resolve_reusable_volume
from .serialization import serialize_evaluation, serialize_workload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile and correctness-test one CUDA candidate on an H100 worker."
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--api-key-env", default="RUNPOD_API_KEY")
    parser.add_argument("--gpu-type", default="NVIDIA H100 80GB HBM3")
    parser.add_argument("--network-volume-id")
    parser.add_argument("--data-center-id")
    parser.add_argument("--auto-volume", action="store_true")
    parser.add_argument("--preferred-data-center-id")
    parser.add_argument("--volume-name", default="gpu-kernel-mcts")
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
    if not arguments.candidate.is_file():
        parser.error(f"candidate file does not exist: {arguments.candidate}")
    if arguments.report.exists():
        parser.error(f"refusing to overwrite existing report: {arguments.report}")
    api_key = os.environ.get(arguments.api_key_env)
    if not api_key:
        parser.error(f"environment variable {arguments.api_key_env!r} is not set")
    network_volume_id, data_center_id = resolve_reusable_volume(
        parser, arguments, api_key
    )

    hardware = HardwareSpec(
        "H100",
        form_factor="SXM",
        minimum_compute_capability="9.0",
    )
    progress = ReadinessProgress("candidate evaluation worker")
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
        execution = run_candidate_evaluation(
            provider=provider,
            hardware=hardware,
            workload=BF16_GEMM_WORKLOAD,
            root_program=load_bf16_gemm_root(),
            candidate_program=KernelProgram(
                arguments.candidate.read_text(encoding="utf-8"),
                backend="cuda_cpp",
            ),
            run_id=arguments.run_id,
        )
    finally:
        progress.finish()

    report = {
        "run_id": execution.run_id,
        "candidate_path": str(arguments.candidate.resolve()),
        "workload": serialize_workload(BF16_GEMM_WORKLOAD),
        "requested_hardware": asdict(hardware),
        "environment_manifest": execution.environment_manifest,
        "root_evaluation": serialize_evaluation(execution.root),
        "candidate_evaluation": serialize_evaluation(execution.candidate),
    }
    arguments.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _print_evaluation("Root", execution.root)
    _print_evaluation("Candidate", execution.candidate)
    print(f"Evaluation report: {arguments.report.resolve()}")
    return 0 if execution.candidate.status == ProposalStatus.VALID else 1


def _print_evaluation(label: str, evaluation: EvaluationResult) -> None:
    correctness = evaluation.correctness
    benchmark = evaluation.benchmark
    print(
        f"{label}: status={evaluation.status.value}, "
        f"compile={evaluation.compile_status.value}, "
        f"correctness={evaluation.correctness_status.value}, "
        f"invalid_reason={evaluation.invalid_reason.value if evaluation.invalid_reason else None}"
    )
    if correctness is not None:
        print(
            f"{label} correctness: max_error={correctness.maximum_error}, "
            f"mean_error={correctness.mean_error}, "
            f"failed_test_id={correctness.failed_test_id}"
        )
    if benchmark is not None:
        print(
            f"{label} benchmark: median={benchmark.median_us} us, "
            f"mean={benchmark.mean_us} us, samples={len(benchmark.timings_us)}, "
            f"reward={evaluation.reward}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
