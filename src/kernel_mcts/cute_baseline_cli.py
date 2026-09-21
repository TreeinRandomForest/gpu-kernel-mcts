from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import socket
import subprocess
from typing import Mapping, Sequence

from .cute_baseline import (
    DEFAULT_EXAMPLE,
    run_same_worker_comparison,
    run_hopper_bf16_comparable,
    run_hopper_bf16_feasibility,
)
from .benchmarks import BF16_GEMM_WORKLOAD
from .cute_tuning import (
    run_cute_pipeline_interaction_tuning,
    run_cute_schedule_tuning,
)
from .cute_backend import CuteBackendConfig, CuTeDSLBackend
from .cute_program import PinnedCuteGemmRenderer, REFERENCE_CUTE_GEMM
from .cute_mutations import enumerate_cute_mutations
from .cute_diagnostics import (
    artifact_changes,
    discover_cache_roots,
    loaded_module_paths,
    select_fingerprint_candidate,
    snapshot_files,
)
from .cute_capabilities import inspect_cute_structural_capabilities
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
        choices=(
            "comparison",
            "comparable",
            "feasibility",
            "tune",
            "pipeline-tune",
            "backend",
            "backend-profile",
            "design-space",
            "diagnostic",
            "structural-capabilities",
        ),
        default="comparison",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--profile-set",
        choices=("lightweight_v1", "diagnostic_v2"),
        default="lightweight_v1",
    )
    parser.add_argument("--pipeline-stages", type=int, choices=(2, 3, 4))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.mode == "comparison":
        result = _run_comparison(arguments.example)
    elif arguments.mode == "tune":
        result = _run_tuning(arguments.example)
    elif arguments.mode == "pipeline-tune":
        result = _run_pipeline_tuning(arguments.example)
    elif arguments.mode == "backend":
        result = _run_backend_evaluation(
            pipeline_stages=arguments.pipeline_stages
        )
    elif arguments.mode == "backend-profile":
        result = _run_backend_evaluation(
            arguments.profile_set,
            pipeline_stages=arguments.pipeline_stages,
        )
    elif arguments.mode == "design-space":
        result = _describe_design_space()
    elif arguments.mode == "diagnostic":
        result = _run_artifact_diagnostic(arguments.example)
    elif arguments.mode == "structural-capabilities":
        result = inspect_cute_structural_capabilities(arguments.example)
    elif arguments.mode == "comparable":
        result = run_hopper_bf16_comparable(arguments.example)
    else:
        result = run_hopper_bf16_feasibility(arguments.example)
    encoded = json.dumps(result, sort_keys=True)
    if arguments.output is not None:
        arguments.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


def _describe_design_space() -> Mapping[str, object]:
    proposals = enumerate_cute_mutations(REFERENCE_CUTE_GEMM)
    return {
        "status": "ok",
        "backend": "cute_dsl",
        "workload": BF16_GEMM_WORKLOAD.benchmark_id,
        "parent_representation": REFERENCE_CUTE_GEMM.as_dict(),
        "parent_configuration_hash": REFERENCE_CUTE_GEMM.configuration_hash,
        "proposal_count": len(proposals),
        "proposals": [proposal.as_dict() for proposal in proposals],
        "budget": {"b_gen": 0, "b_tune": 0},
    }


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


def _run_pipeline_tuning(example: Path):
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
    )
    suite = VendorBaselineSuite(
        VendorBaselineConfig(
            artifact_root=Path("/tmp/kernel-mcts-cute-pipeline-tuning"),
            cutlass_path=Path(environment.get("CUTLASS_PATH", "/opt/cutlass")),
        )
    )
    vendor = suite.run(BF16_GEMM_WORKLOAD)
    return run_cute_pipeline_interaction_tuning(
        lambda program: run_hopper_bf16_comparable(
            example,
            schedule=program.schedule,
            pipeline_stages=program.pipeline_stages,
        ),
        vendor["cublas"],
        serialize_environment_manifest(manifest),
    )


def _run_backend_evaluation(
    profile_metric_set: str | None = None,
    *,
    pipeline_stages: int | None = None,
):
    import cutlass
    import torch

    environment = dict(os.environ)
    environment.setdefault("KERNEL_MCTS_WORKER_ID", socket.gethostname())
    environment.setdefault("KERNEL_MCTS_PROVIDER", "standalone")
    manifest = capture_environment_manifest(environment)
    manifest = _enrich_cute_manifest(
        manifest,
        cutlass_version=str(cutlass.__version__),
        pytorch_version=str(torch.__version__),
        driver_version=_driver_version(),
    )
    representation = replace(
        REFERENCE_CUTE_GEMM,
        pipeline_stages=pipeline_stages,
    )
    program = PinnedCuteGemmRenderer().render(representation)
    backend = CuTeDSLBackend(
        CuteBackendConfig(
            artifact_root=Path("/tmp/kernel-mcts-cute-backend"),
            architecture=f"sm_{manifest.compute_capability.replace('.', '')}",
            ncu_version=manifest.profiler_versions.get("ncu"),
        )
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
                "tile_shape_mn": [representation.tile_m, representation.tile_n],
                "cluster_shape_mn": [
                    representation.cluster_m,
                    representation.cluster_n,
                ],
                "pipeline_stages": representation.pipeline_stages,
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
    assert evaluation.compilation is not None
    report = dict(
        _backend_evaluation_report(
            manifest,
            root_compilation,
            evaluation,
            representation=representation,
        )
    )
    if profile_metric_set is not None:
        report["profile"] = dict(
            evaluator.lightweight_profile(
                evaluation,
                BF16_GEMM_WORKLOAD,
                profile_metric_set,
            )
        )
        report["profile_did_not_change_reward"] = evaluation.reward == 0.0
    return report


def _run_artifact_diagnostic(example: Path):
    environment = dict(os.environ)
    roots = discover_cache_roots(environment)
    before_files = snapshot_files(roots)
    before_modules = set(loaded_module_paths())
    result = dict(
        run_hopper_bf16_comparable(
            example,
            capture_jit_diagnostics=True,
        )
    )
    after_files = snapshot_files(roots)
    after_modules = set(loaded_module_paths())
    changes = artifact_changes(before_files, after_files)
    return {
        "status": "ok",
        "benchmark_id": "bf16_gemm_4096_h100",
        "cache_roots": [str(root) for root in roots],
        "files_before": len(before_files),
        "files_after": len(after_files),
        "artifact_changes": [item.as_dict() for item in changes],
        "fingerprint_candidate": select_fingerprint_candidate(changes),
        "new_loaded_modules": sorted(after_modules - before_modules),
        "kernel_callable": result.pop("jit_diagnostics", {}),
        "evaluation": result,
    }


def _backend_evaluation_report(
    manifest,
    root_compilation,
    evaluation,
    *,
    representation=REFERENCE_CUTE_GEMM,
):
    assert root_compilation.artifact is not None
    assert evaluation.compilation is not None
    return {
        "status": "ok",
        "representation": representation.as_dict(),
        "configuration_hash": representation.configuration_hash,
        "environment_manifest": serialize_environment_manifest(manifest),
        "initial_jit_evaluation": {
            "artifact_id": root_compilation.artifact.artifact_id,
            "duration_seconds": root_compilation.duration_seconds,
            "stdout": root_compilation.stdout,
            "stderr": root_compilation.stderr,
            "artifact_paths": list(root_compilation.artifact_paths),
        },
        "cache_validation": {
            "artifact_reused": (
                evaluation.compilation.artifact_id
                == root_compilation.artifact.artifact_id
            ),
            "evaluator_compile_duration_seconds": evaluation.compilation.duration_seconds,
            "evaluator_compile_stdout": evaluation.compilation.stdout,
        },
        "evaluation": serialize_evaluation(evaluation),
    }


def _enrich_cute_manifest(
    manifest,
    *,
    cutlass_version: str,
    pytorch_version: str,
    driver_version: dict[str, str],
):
    return replace(
        manifest,
        toolchain_versions={
            **manifest.toolchain_versions,
            **driver_version,
        },
        library_versions={
            **manifest.library_versions,
            "cutlass_dsl": cutlass_version,
            "pytorch": pytorch_version,
        },
    )


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
