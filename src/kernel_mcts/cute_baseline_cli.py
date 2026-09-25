from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
from types import ModuleType
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
from .cute_independent_tma import INDEPENDENT_TMA_DEBUG_STAGES
from .cute_backend import CuteBackendConfig, CuTeDSLBackend
from .cute_program import (
    CUTE_GEMM_SCHEMA_VERSION,
    PinnedCuteGemmRenderer,
    REFERENCE_CUTE_GEMM,
)
from .cute_schedule import CuteSchedule
from .cute_source_transform import validate_pinned_cute_gemm_source
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
            "canonical-epilogue-validation",
            "canonical-swizzle-validation",
            "design-space",
            "diagnostic",
            "structural-capabilities",
            "wgmma-inflight-diagnostic",
            "tma-copy-diagnostic",
            "epilogue-stage-diagnostic",
            "smem-swizzle-diagnostic",
            "independent-lowering-bindings",
            "independent-tma-copy",
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
    parser.add_argument(
        "--wgmma-configuration",
        choices=("pinned_default", "single_warp_group"),
        default="pinned_default",
    )
    parser.add_argument(
        "--wgmma-inflight-groups", type=int, choices=(1, 2), default=1
    )
    parser.add_argument(
        "--tma-load-policy",
        choices=("auto_multicast", "non_multicast"),
        default="auto_multicast",
    )
    parser.add_argument("--epilogue-stages", type=int, choices=(2, 3, 4))
    parser.add_argument(
        "--shared-memory-swizzle",
        choices=("heuristic", "sw64"),
        default="heuristic",
    )
    parser.add_argument("--tile-m", type=int)
    parser.add_argument("--tile-n", type=int)
    parser.add_argument("--cluster-m", type=int)
    parser.add_argument("--cluster-n", type=int)
    parser.add_argument(
        "--independent-tma-stage",
        choices=INDEPENDENT_TMA_DEBUG_STAGES,
        default="cluster_ab_multicast",
    )
    parser.add_argument(
        "--independent-repository-contract",
        action="store_true",
        help="use canonical inputs, cuBLAS reference, and repository timing policy",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if (
        arguments.mode != "wgmma-inflight-diagnostic"
        and arguments.wgmma_inflight_groups != 1
    ):
        raise ValueError(
            "--wgmma-inflight-groups is available only in "
            "wgmma-inflight-diagnostic mode"
        )
    if (
        arguments.mode != "independent-tma-copy"
        and arguments.independent_tma_stage != "cluster_ab_multicast"
    ):
        raise ValueError(
            "--independent-tma-stage is available only in independent-tma-copy mode"
        )
    if (
        arguments.mode != "independent-tma-copy"
        and arguments.independent_repository_contract
    ):
        raise ValueError(
            "--independent-repository-contract is available only in "
            "independent-tma-copy mode"
        )
    if (
        arguments.mode != "tma-copy-diagnostic"
        and arguments.tma_load_policy != "auto_multicast"
    ):
        raise ValueError(
            "--tma-load-policy is available only in tma-copy-diagnostic mode"
        )
    if (
        arguments.mode
        not in ("backend", "backend-profile", "epilogue-stage-diagnostic")
        and arguments.epilogue_stages is not None
    ):
        raise ValueError(
            "--epilogue-stages is available only in epilogue-stage-diagnostic mode"
        )
    if (
        arguments.mode == "epilogue-stage-diagnostic"
        and arguments.epilogue_stages is None
    ):
        raise ValueError("epilogue-stage-diagnostic requires --epilogue-stages")
    if (
        arguments.mode not in ("backend", "backend-profile")
        and arguments.shared_memory_swizzle != "heuristic"
    ):
        raise ValueError(
            "--shared-memory-swizzle is available only in backend modes"
        )
    schedule_values = (
        arguments.tile_m,
        arguments.tile_n,
        arguments.cluster_m,
        arguments.cluster_n,
    )
    if arguments.mode not in ("backend", "backend-profile") and any(
        value is not None for value in schedule_values
    ):
        raise ValueError(
            "typed schedule arguments are available only in backend modes"
        )
    if any(value is not None for value in schedule_values) and any(
        value is None for value in schedule_values
    ):
        raise ValueError(
            "--tile-m, --tile-n, --cluster-m, and --cluster-n must be provided together"
        )
    canonical_schedule = (
        CuteSchedule(*schedule_values)
        if all(value is not None for value in schedule_values)
        else REFERENCE_CUTE_GEMM.schedule
    )
    if arguments.mode == "comparison":
        result = _run_comparison(arguments.example)
    elif arguments.mode == "tune":
        result = _run_tuning(arguments.example)
    elif arguments.mode == "pipeline-tune":
        result = _run_pipeline_tuning(arguments.example)
    elif arguments.mode == "backend":
        result = _run_backend_evaluation(
            schedule=canonical_schedule,
            pipeline_stages=arguments.pipeline_stages,
            epilogue_stages=arguments.epilogue_stages,
            shared_memory_swizzle=arguments.shared_memory_swizzle,
            wgmma_configuration=arguments.wgmma_configuration,
        )
    elif arguments.mode == "backend-profile":
        result = _run_backend_evaluation(
            arguments.profile_set,
            schedule=canonical_schedule,
            pipeline_stages=arguments.pipeline_stages,
            epilogue_stages=arguments.epilogue_stages,
            shared_memory_swizzle=arguments.shared_memory_swizzle,
            wgmma_configuration=arguments.wgmma_configuration,
        )
    elif arguments.mode == "canonical-epilogue-validation":
        result = _run_canonical_epilogue_validation()
    elif arguments.mode == "canonical-swizzle-validation":
        result = _run_canonical_swizzle_validation()
    elif arguments.mode == "design-space":
        result = _describe_design_space()
    elif arguments.mode == "diagnostic":
        result = _run_artifact_diagnostic(arguments.example)
    elif arguments.mode == "structural-capabilities":
        result = inspect_cute_structural_capabilities(arguments.example)
    elif arguments.mode == "wgmma-inflight-diagnostic":
        result = _run_wgmma_inflight_diagnostic(
            arguments.example,
            pipeline_stages=arguments.pipeline_stages,
            wgmma_inflight_groups=arguments.wgmma_inflight_groups,
        )
    elif arguments.mode == "tma-copy-diagnostic":
        result = _run_tma_copy_diagnostic(
            arguments.example,
            tma_load_policy=arguments.tma_load_policy,
        )
    elif arguments.mode == "epilogue-stage-diagnostic":
        result = _run_epilogue_stage_diagnostic(
            arguments.example,
            epilogue_stages=arguments.epilogue_stages,
        )
    elif arguments.mode == "smem-swizzle-diagnostic":
        result = _run_smem_swizzle_diagnostic(arguments.example)
    elif arguments.mode == "independent-lowering-bindings":
        result = _run_independent_lowering_bindings()
    elif arguments.mode == "independent-tma-copy":
        if arguments.pipeline_stages not in (None, 2, 3):
            raise ValueError(
                "independent-tma-copy supports pipeline stages 2 or 3"
            )
        result = _run_independent_tma_copy(
            arguments.independent_tma_stage,
            pipeline_stages=arguments.pipeline_stages or 3,
            repository_contract=arguments.independent_repository_contract,
        )
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


def _run_independent_lowering_bindings() -> Mapping[str, object]:
    from .cute_independent import make_independent_cute_gemm
    from .cute_independent_lowering import lower_independent_cute_gemm_static

    kernel = make_independent_cute_gemm()
    lowering = lower_independent_cute_gemm_static(kernel)
    namespace: dict[str, object] = {}
    exec(compile(lowering.source, "independent_cute_lowering.py", "exec"), namespace)
    required_api_bindings = namespace["required_api_bindings"]
    if not callable(required_api_bindings):
        raise TypeError("generated required_api_bindings must be callable")
    bindings = required_api_bindings()
    return {
        "status": "ok",
        "phase": "static_lowering_bindings",
        "cutlass_version": str(namespace["cutlass"].__version__),
        "binding_count": len(bindings),
        "all_bindings_callable": all(callable(binding) for binding in bindings),
        **lowering.as_dict(),
    }


def _run_independent_tma_copy(
    debug_stage: str,
    *,
    pipeline_stages: int = 3,
    repository_contract: bool = False,
) -> Mapping[str, object]:
    from .cute_independent import make_independent_cute_gemm
    from .cute_independent_tma import render_independent_tma_copy_diagnostic

    rendered = render_independent_tma_copy_diagnostic(
        make_independent_cute_gemm(pipeline_stages=pipeline_stages),
        debug_stage=debug_stage,
        repository_contract=repository_contract,
    )
    with tempfile.TemporaryDirectory(
        prefix="kernel-mcts-independent-tma-"
    ) as directory:
        module = _load_generated_module(
            rendered.source,
            module_name=f"kernel_mcts_independent_tma_{rendered.source_hash[:16]}",
            path=Path(directory) / "independent_tma_copy.py",
        )
        try:
            run_diagnostic = getattr(module, "run_diagnostic", None)
            if not callable(run_diagnostic):
                raise TypeError("generated run_diagnostic must be callable")
            result = run_diagnostic()
        finally:
            sys.modules.pop(module.__name__, None)
    if not isinstance(result, Mapping):
        raise TypeError("generated TMA diagnostic must return a mapping")
    return {**result, "source_hash": rendered.source_hash}


def _load_generated_module(
    source: str,
    *,
    module_name: str,
    path: Path,
) -> ModuleType:
    """Materialize generated CuTe source so its AST is inspectable during JIT."""

    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import generated CuTe module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


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
    schedule: CuteSchedule = REFERENCE_CUTE_GEMM.schedule,
    pipeline_stages: int | None = None,
    epilogue_stages: int | None = None,
    shared_memory_swizzle: str = "heuristic",
    wgmma_configuration: str = "pinned_default",
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
        tile_m=schedule.tile_m,
        tile_n=schedule.tile_n,
        cluster_m=schedule.cluster_m,
        cluster_n=schedule.cluster_n,
        pipeline_stages=pipeline_stages,
        epilogue_stages=epilogue_stages,
        shared_memory_swizzle=shared_memory_swizzle,
        wgmma_configuration=wgmma_configuration,
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
                "epilogue_stages": representation.epilogue_stages,
                "shared_memory_swizzle": representation.shared_memory_swizzle,
                "wgmma_configuration": representation.wgmma_configuration,
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


def _run_canonical_epilogue_validation() -> Mapping[str, object]:
    schedule = CuteSchedule(128, 256, 2, 1)
    variants = {
        str(stage): _run_backend_evaluation(
            schedule=schedule,
            epilogue_stages=stage,
        )
        for stage in (2, 3)
    }
    checks: dict[str, bool] = {}
    for stage, report in variants.items():
        representation = report["representation"]
        evaluation = report["evaluation"]
        cache_validation = report["cache_validation"]
        assert isinstance(representation, Mapping)
        assert isinstance(evaluation, Mapping)
        assert isinstance(cache_validation, Mapping)
        checks[f"stage_{stage}_schema_v{CUTE_GEMM_SCHEMA_VERSION}"] = representation.get(
            "schema_version"
        ) == CUTE_GEMM_SCHEMA_VERSION
        checks[f"stage_{stage}_canonical_schedule"] = (
            representation.get("tile_m"),
            representation.get("tile_n"),
            representation.get("cluster_m"),
            representation.get("cluster_n"),
            representation.get("epilogue_stages"),
        ) == (128, 256, 2, 1, int(stage))
        checks[f"stage_{stage}_cache_reused"] = (
            cache_validation.get("artifact_reused") is True
        )
        checks[f"stage_{stage}_valid"] = evaluation.get("status") == "VALID"
        checks[f"stage_{stage}_correct"] = (
            evaluation.get("correctness_status") == "PASS"
        )

    configuration_hashes = {
        report["configuration_hash"] for report in variants.values()
    }
    runtime_fingerprints = set()
    for report in variants.values():
        evaluation = report["evaluation"]
        assert isinstance(evaluation, Mapping)
        metadata = evaluation.get("metadata")
        runtime_fingerprint = (
            metadata.get("runtime_fingerprint")
            if isinstance(metadata, Mapping)
            else None
        )
        runtime_fingerprints.add(
            runtime_fingerprint.get("sha256")
            if isinstance(runtime_fingerprint, Mapping)
            else None
        )
    checks["distinct_configuration_hashes"] = len(configuration_hashes) == 2
    checks["distinct_runtime_fingerprints"] = (
        None not in runtime_fingerprints and len(runtime_fingerprints) == 2
    )
    return {
        "status": "ok" if all(checks.values()) else "validation_failed",
        "validation_kind": "canonical_epilogue_stages",
        "checks": checks,
        "variants": variants,
    }


def _run_canonical_swizzle_validation() -> Mapping[str, object]:
    schedule = CuteSchedule(128, 256, 2, 1)
    variants = {
        policy: _run_backend_evaluation(
            "diagnostic_v2",
            schedule=schedule,
            shared_memory_swizzle=policy,
        )
        for policy in ("heuristic", "sw64")
    }
    checks: dict[str, bool] = {}
    runtime_fingerprints = set()
    for policy, report in variants.items():
        representation = report["representation"]
        evaluation = report["evaluation"]
        cache_validation = report["cache_validation"]
        assert isinstance(representation, Mapping)
        assert isinstance(evaluation, Mapping)
        assert isinstance(cache_validation, Mapping)
        checks[f"{policy}_schema_v{CUTE_GEMM_SCHEMA_VERSION}"] = (
            representation.get("schema_version") == CUTE_GEMM_SCHEMA_VERSION
        )
        checks[f"{policy}_canonical_state"] = (
            representation.get("tile_m"),
            representation.get("tile_n"),
            representation.get("cluster_m"),
            representation.get("cluster_n"),
            representation.get("shared_memory_swizzle"),
        ) == (128, 256, 2, 1, policy)
        checks[f"{policy}_cache_reused"] = (
            cache_validation.get("artifact_reused") is True
        )
        checks[f"{policy}_valid"] = evaluation.get("status") == "VALID"
        checks[f"{policy}_correct"] = (
            evaluation.get("correctness_status") == "PASS"
        )
        checks[f"{policy}_profiled"] = (
            isinstance(report.get("profile"), Mapping)
            and report.get("profile_did_not_change_reward") is True
        )
        metadata = evaluation.get("metadata")
        runtime_fingerprint = (
            metadata.get("runtime_fingerprint")
            if isinstance(metadata, Mapping)
            else None
        )
        runtime_fingerprints.add(
            runtime_fingerprint.get("sha256")
            if isinstance(runtime_fingerprint, Mapping)
            else None
        )

    configuration_hashes = {
        report["configuration_hash"] for report in variants.values()
    }
    checks["distinct_configuration_hashes"] = len(configuration_hashes) == 2
    checks["distinct_runtime_fingerprints"] = (
        None not in runtime_fingerprints and len(runtime_fingerprints) == 2
    )
    return {
        "status": "ok" if all(checks.values()) else "validation_failed",
        "validation_kind": "canonical_shared_memory_swizzle",
        "checks": checks,
        "variants": variants,
    }


def _run_wgmma_inflight_diagnostic(
    example: Path,
    *,
    pipeline_stages: int | None,
    wgmma_inflight_groups: int,
) -> Mapping[str, object]:
    if (
        pipeline_stages is not None
        and wgmma_inflight_groups >= pipeline_stages
    ):
        raise ValueError(
            "diagnostic WGMMA in-flight groups must be smaller than pipeline stages"
        )
    result = dict(
        run_hopper_bf16_comparable(
            example,
            pipeline_stages=pipeline_stages,
            wgmma_inflight_groups=wgmma_inflight_groups,
            capture_jit_diagnostics=True,
        )
    )
    result["diagnostic_control"] = {
        "name": "wgmma_inflight_groups",
        "value": wgmma_inflight_groups,
        "canonical_search_state": False,
    }
    return result


def _run_tma_copy_diagnostic(
    example: Path,
    *,
    tma_load_policy: str,
) -> Mapping[str, object]:
    validate_pinned_cute_gemm_source(example)
    schedule = CuteSchedule(128, 256, 2, 1)
    result = dict(
        run_hopper_bf16_comparable(
            example,
            schedule=schedule,
            tma_load_policy=tma_load_policy,
            capture_jit_diagnostics=True,
        )
    )
    result["diagnostic_control"] = {
        "name": "tma_load_policy",
        "value": tma_load_policy,
        "pinned_behavior": "auto_multicast",
        "cluster_shape_mn": [schedule.cluster_m, schedule.cluster_n],
        "canonical_search_state": False,
    }
    return result


def _run_epilogue_stage_diagnostic(
    example: Path,
    *,
    epilogue_stages: int,
) -> Mapping[str, object]:
    schedule = CuteSchedule(128, 256, 2, 1)
    diagnostic_control = {
        "name": "epilogue_stages",
        "value": epilogue_stages,
        "pinned_behavior": 4,
        "cluster_shape_mn": [schedule.cluster_m, schedule.cluster_n],
        "canonical_search_state": False,
    }
    try:
        validate_pinned_cute_gemm_source(example)
        result = dict(
            run_hopper_bf16_comparable(
                example,
                schedule=schedule,
                epilogue_stages=epilogue_stages,
                capture_jit_diagnostics=True,
            )
        )
    except Exception as error:
        return {
            "status": "diagnostic_failed",
            "candidate_status": "NOT_ADMITTED",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "diagnostic_control": diagnostic_control,
        }
    result["diagnostic_control"] = diagnostic_control
    return result


def _run_smem_swizzle_diagnostic(example: Path) -> Mapping[str, object]:
    validate_pinned_cute_gemm_source(example)
    schedule = CuteSchedule(128, 256, 2, 1)
    variants: dict[str, Mapping[str, object]] = {}
    for policy in ("heuristic", "forced_sw64"):
        control = {
            "name": "smem_swizzle_policy",
            "value": policy,
            "scope": (
                "layout-atom heuristic; for this BF16 workload A/B change from "
                "SW128 to SW64 while the epilogue remains SW64"
            ),
            "tile_shape_mn": [schedule.tile_m, schedule.tile_n],
            "cluster_shape_mn": [schedule.cluster_m, schedule.cluster_n],
            "canonical_search_state": False,
        }
        try:
            result = dict(
                run_hopper_bf16_comparable(
                    example,
                    schedule=schedule,
                    smem_swizzle_policy=policy,
                    capture_jit_diagnostics=True,
                )
            )
            result["diagnostic_control"] = control
        except Exception as error:
            result = {
                "status": "diagnostic_failed",
                "candidate_status": "NOT_ADMITTED",
                "error_type": type(error).__name__,
                "error_message": str(error),
                "diagnostic_control": control,
            }
        variants[policy] = result
    return {
        "status": "ok",
        "diagnostic": "smem_swizzle_policy",
        "interpretation": (
            "Diagnostic only; forced SW64 is not canonical CuTe state or an "
            "MCTS mutation until H100 validation succeeds."
        ),
        "variants": variants,
    }


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
