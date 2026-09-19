from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import socket
import subprocess
from typing import Sequence

from .cute_baseline import (
    DEFAULT_EXAMPLE,
    run_same_worker_comparison,
    run_hopper_bf16_comparable,
    run_hopper_bf16_feasibility,
)
from .benchmarks import BF16_GEMM_WORKLOAD
from .cute_tuning import run_cute_schedule_tuning
from .cute_backend import CuteBackendConfig, CuTeDSLBackend
from .cute_program import PinnedCuteGemmRenderer, REFERENCE_CUTE_GEMM
from .evaluation import BackendKernelEvaluator, EvaluationContext
from .serialization import serialize_environment_manifest, serialize_evaluation
from .vendor_baselines import VendorBaselineConfig, VendorBaselineSuite
from .worker_service import capture_environment_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run fixed BF16 GEMM baselines on one H100 worker."
    )
    parser.add_argument("--example", type=Path, default=DEFAULT_EXAMPLE)
    parser.add_argument(
        "--mode",
        choices=("comparison", "comparable", "feasibility", "tune", "backend"),
        default="comparison",
    )
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.mode == "comparison":
        result = _run_comparison(arguments.example)
    elif arguments.mode == "tune":
        result = _run_tuning(arguments.example)
    elif arguments.mode == "backend":
        result = _run_backend_evaluation()
    elif arguments.mode == "comparable":
        result = run_hopper_bf16_comparable(arguments.example)
    else:
        result = run_hopper_bf16_feasibility(arguments.example)
    encoded = json.dumps(result, sort_keys=True)
    if arguments.output is not None:
        arguments.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


def _run_comparison(example: Path):
    import cutlass
    import torch

    environment = dict(os.environ)
    environment.setdefault("KERNEL_MCTS_WORKER_ID", socket.gethostname())
    environment.setdefault("KERNEL_MCTS_PROVIDER", "standalone")
    manifest = capture_environment_manifest(environment)
    manifest = replace(
        manifest,
        toolchain_versions={
            **manifest.toolchain_versions,
            **_driver_version(),
        },
        library_versions={
            **manifest.library_versions,
            "cutlass_dsl": str(cutlass.__version__),
            "pytorch": str(torch.__version__),
        },
        operating_state=_gpu_operating_state(),
    )
    suite = VendorBaselineSuite(
        VendorBaselineConfig(
            artifact_root=Path("/tmp/kernel-mcts-cute-comparison"),
            cutlass_path=Path(environment.get("CUTLASS_PATH", "/opt/cutlass")),
        )
    )
    return run_same_worker_comparison(
        lambda: suite.run(BF16_GEMM_WORKLOAD),
        lambda: run_hopper_bf16_comparable(example),
        serialize_environment_manifest(manifest),
    )


def _run_tuning(example: Path):
    import cutlass
    import torch

    environment = dict(os.environ)
    environment.setdefault("KERNEL_MCTS_WORKER_ID", socket.gethostname())
    environment.setdefault("KERNEL_MCTS_PROVIDER", "standalone")
    manifest = capture_environment_manifest(environment)
    manifest = replace(
        manifest,
        toolchain_versions={**manifest.toolchain_versions, **_driver_version()},
        library_versions={
            **manifest.library_versions,
            "cutlass_dsl": str(cutlass.__version__),
            "pytorch": str(torch.__version__),
        },
        operating_state=_gpu_operating_state(),
    )
    suite = VendorBaselineSuite(
        VendorBaselineConfig(
            artifact_root=Path("/tmp/kernel-mcts-cute-tuning"),
            cutlass_path=Path(environment.get("CUTLASS_PATH", "/opt/cutlass")),
        )
    )
    vendor = suite.run(BF16_GEMM_WORKLOAD)
    return run_cute_schedule_tuning(
        lambda schedule: run_hopper_bf16_comparable(example, schedule=schedule),
        vendor["cublas"],
        serialize_environment_manifest(manifest),
    )


def _run_backend_evaluation():
    environment = dict(os.environ)
    environment.setdefault("KERNEL_MCTS_WORKER_ID", socket.gethostname())
    environment.setdefault("KERNEL_MCTS_PROVIDER", "standalone")
    manifest = capture_environment_manifest(environment)
    program = PinnedCuteGemmRenderer().render(REFERENCE_CUTE_GEMM)
    backend = CuTeDSLBackend(
        CuteBackendConfig(artifact_root=Path("/tmp/kernel-mcts-cute-backend"))
    )
    root_compilation = backend.compile(program, BF16_GEMM_WORKLOAD)
    if not root_compilation.success or root_compilation.artifact is None:
        raise RuntimeError(f"CuTe reference failed JIT: {root_compilation.stderr}")
    root_correctness = backend.check_correctness(
        root_compilation.artifact, BF16_GEMM_WORKLOAD
    )
    if not root_correctness.success:
        raise RuntimeError("CuTe reference failed correctness")
    root_benchmark = backend.benchmark(root_compilation.artifact, BF16_GEMM_WORKLOAD)
    evaluator = BackendKernelEvaluator(
        backend=backend,
        root_benchmark=root_benchmark,
        context=EvaluationContext(
            worker_id=manifest.worker_id,
            environment_manifest_id=manifest.manifest_id,
            launch_config={
                "tile_shape_mn": [REFERENCE_CUTE_GEMM.tile_m, REFERENCE_CUTE_GEMM.tile_n],
                "cluster_shape_mn": [
                    REFERENCE_CUTE_GEMM.cluster_m,
                    REFERENCE_CUTE_GEMM.cluster_n,
                ],
            },
            hardware_toolchain={
                "gpu_model": manifest.gpu_model,
                "compute_capability": manifest.compute_capability,
                "form_factor": manifest.form_factor,
                "toolchain_versions": manifest.toolchain_versions,
                "library_versions": manifest.library_versions,
            },
        ),
    )
    evaluation = evaluator.evaluate(program, BF16_GEMM_WORKLOAD)
    return {
        "status": "ok",
        "representation": REFERENCE_CUTE_GEMM.as_dict(),
        "configuration_hash": REFERENCE_CUTE_GEMM.configuration_hash,
        "environment_manifest": serialize_environment_manifest(manifest),
        "evaluation": serialize_evaluation(evaluation),
    }


def _driver_version() -> dict[str, str]:
    try:
        value = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError, IndexError):
        return {}
    return {"driver": value}


def _gpu_operating_state() -> dict[str, object]:
    fields = (
        "clocks.current.graphics",
        "clocks.current.memory",
        "temperature.gpu",
        "power.draw",
        "power.limit",
    )
    try:
        values = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={','.join(fields)}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.splitlines()[0].split(",")
    except (OSError, subprocess.SubprocessError, IndexError):
        return {}
    return {name: value.strip() for name, value in zip(fields, values)}


if __name__ == "__main__":
    raise SystemExit(main())
