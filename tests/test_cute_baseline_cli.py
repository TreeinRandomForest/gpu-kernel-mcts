from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from kernel_mcts.cute_baseline_cli import (
    _backend_evaluation_report,
    _run_canonical_epilogue_validation,
    _run_canonical_swizzle_validation,
    _describe_design_space,
    _enrich_cute_manifest,
    _run_wgmma_inflight_diagnostic,
    _run_tma_copy_diagnostic,
    _run_epilogue_stage_diagnostic,
    _run_smem_swizzle_diagnostic,
    _load_generated_module,
    build_parser,
    main,
)
from kernel_mcts.domain import (
    BenchmarkResult,
    CompilationEvidence,
    CompileStatus,
    CorrectnessStatus,
    EvaluationResult,
    KernelProgram,
    ProposalStatus,
)
from kernel_mcts.providers import EnvironmentManifest


def test_cli_accepts_standalone_tuning_mode(tmp_path) -> None:
    output = tmp_path / "tuning.json"

    arguments = build_parser().parse_args(
        ["--mode", "tune", "--output", str(output)]
    )

    assert arguments.mode == "tune"
    assert arguments.output == output


def test_cli_accepts_pipeline_interaction_tuning_mode() -> None:
    arguments = build_parser().parse_args(["--mode", "pipeline-tune"])

    assert arguments.mode == "pipeline-tune"


def test_cli_accepts_backend_evaluation_mode() -> None:
    arguments = build_parser().parse_args(["--mode", "backend"])

    assert arguments.mode == "backend"


def test_cli_accepts_bounded_pipeline_stage_evaluation() -> None:
    arguments = build_parser().parse_args(
        ["--mode", "backend", "--pipeline-stages", "3"]
    )

    assert arguments.pipeline_stages == 3


def test_cli_accepts_single_warp_group_evaluation() -> None:
    arguments = build_parser().parse_args(
        ["--mode", "backend", "--wgmma-configuration", "single_warp_group"]
    )

    assert arguments.wgmma_configuration == "single_warp_group"


def test_cli_accepts_two_wgmma_inflight_groups() -> None:
    arguments = build_parser().parse_args(
        ["--mode", "wgmma-inflight-diagnostic", "--wgmma-inflight-groups", "2"]
    )

    assert arguments.mode == "wgmma-inflight-diagnostic"
    assert arguments.wgmma_inflight_groups == 2


def test_wgmma_inflight_diagnostic_is_not_canonical_state(monkeypatch) -> None:
    calls = []

    def run_comparable(*args, **kwargs):
        calls.append((args, kwargs))
        return {"status": "ok"}

    monkeypatch.setattr(
        "kernel_mcts.cute_baseline_cli.run_hopper_bf16_comparable",
        run_comparable,
    )

    result = _run_wgmma_inflight_diagnostic(
        Path("example.py"), pipeline_stages=3, wgmma_inflight_groups=2
    )

    assert result["diagnostic_control"]["canonical_search_state"] is False
    assert calls[0][1]["wgmma_inflight_groups"] == 2
    assert calls[0][1]["capture_jit_diagnostics"] is True


def test_cli_rejects_inflight_override_in_canonical_backend_mode() -> None:
    with pytest.raises(ValueError, match="only in wgmma-inflight-diagnostic"):
        main(["--mode", "backend", "--wgmma-inflight-groups", "2"])


def test_tma_copy_diagnostic_is_not_canonical_state(monkeypatch) -> None:
    calls = []

    def run_comparable(*args, **kwargs):
        calls.append((args, kwargs))
        return {"status": "ok"}

    monkeypatch.setattr(
        "kernel_mcts.cute_baseline_cli.run_hopper_bf16_comparable",
        run_comparable,
    )
    monkeypatch.setattr(
        "kernel_mcts.cute_baseline_cli.validate_pinned_cute_gemm_source",
        lambda _path: "pinned-hash",
    )

    result = _run_tma_copy_diagnostic(
        Path("example.py"), tma_load_policy="non_multicast"
    )

    assert result["diagnostic_control"]["canonical_search_state"] is False
    assert result["diagnostic_control"]["cluster_shape_mn"] == [2, 1]
    assert calls[0][1]["tma_load_policy"] == "non_multicast"
    assert calls[0][1]["schedule"].as_dict() == {
        "tile_m": 128,
        "tile_n": 256,
        "cluster_m": 2,
        "cluster_n": 1,
    }


def test_cli_rejects_tma_override_in_canonical_backend_mode() -> None:
    with pytest.raises(ValueError, match="only in tma-copy-diagnostic"):
        main(["--mode", "backend", "--tma-load-policy", "non_multicast"])


def test_epilogue_stage_diagnostic_is_not_canonical_state(monkeypatch) -> None:
    calls = []

    def run_comparable(*args, **kwargs):
        calls.append((args, kwargs))
        return {"status": "ok"}

    monkeypatch.setattr(
        "kernel_mcts.cute_baseline_cli.run_hopper_bf16_comparable",
        run_comparable,
    )
    monkeypatch.setattr(
        "kernel_mcts.cute_baseline_cli.validate_pinned_cute_gemm_source",
        lambda _path: "pinned-hash",
    )

    result = _run_epilogue_stage_diagnostic(
        Path("example.py"), epilogue_stages=3
    )

    assert result["diagnostic_control"]["canonical_search_state"] is False
    assert result["diagnostic_control"]["pinned_behavior"] == 4
    assert calls[0][1]["epilogue_stages"] == 3
    assert calls[0][1]["schedule"].as_dict() == {
        "tile_m": 128,
        "tile_n": 256,
        "cluster_m": 2,
        "cluster_n": 1,
    }


def test_epilogue_stage_diagnostic_serializes_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        "kernel_mcts.cute_baseline_cli.validate_pinned_cute_gemm_source",
        lambda _path: "pinned-hash",
    )
    monkeypatch.setattr(
        "kernel_mcts.cute_baseline_cli.run_hopper_bf16_comparable",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("JIT failed")),
    )

    result = _run_epilogue_stage_diagnostic(
        Path("example.py"), epilogue_stages=2
    )

    assert result["status"] == "diagnostic_failed"
    assert result["candidate_status"] == "NOT_ADMITTED"
    assert result["error_type"] == "RuntimeError"
    assert result["error_message"] == "JIT failed"


def test_smem_swizzle_diagnostic_compares_heuristic_and_sw64(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        "kernel_mcts.cute_baseline_cli.validate_pinned_cute_gemm_source",
        lambda _path: "pinned-hash",
    )

    def run_comparable(*args, **kwargs):
        calls.append((args, kwargs))
        return {"status": "ok", "benchmark": {"median_us": 1.0}}

    monkeypatch.setattr(
        "kernel_mcts.cute_baseline_cli.run_hopper_bf16_comparable",
        run_comparable,
    )

    result = _run_smem_swizzle_diagnostic(Path("example.py"))

    assert result["status"] == "ok"
    assert set(result["variants"]) == {"heuristic", "forced_sw64"}
    assert [call[1]["smem_swizzle_policy"] for call in calls] == [
        "heuristic",
        "forced_sw64",
    ]
    assert all(call[1]["capture_jit_diagnostics"] is True for call in calls)
    assert all(
        variant["diagnostic_control"]["canonical_search_state"] is False
        for variant in result["variants"].values()
    )


def test_cli_routes_validated_epilogue_state_through_canonical_backend(
    monkeypatch,
) -> None:
    calls = []

    def run_backend(*args, **kwargs):
        calls.append((args, kwargs))
        return {"status": "ok"}

    monkeypatch.setattr(
        "kernel_mcts.cute_baseline_cli._run_backend_evaluation", run_backend
    )

    result = main(
        [
            "--mode",
            "backend",
            "--tile-m",
            "128",
            "--tile-n",
            "256",
            "--cluster-m",
            "2",
            "--cluster-n",
            "1",
            "--epilogue-stages",
            "3",
        ]
    )

    assert result == 0
    assert calls[0][1]["schedule"].as_dict() == {
        "tile_m": 128,
        "tile_n": 256,
        "cluster_m": 2,
        "cluster_n": 1,
    }
    assert calls[0][1]["epilogue_stages"] == 3
    assert calls[0][1]["shared_memory_swizzle"] == "heuristic"


def test_cli_routes_sw64_state_through_canonical_backend(monkeypatch) -> None:
    calls = []

    def run_backend(*args, **kwargs):
        calls.append((args, kwargs))
        return {"status": "ok"}

    monkeypatch.setattr(
        "kernel_mcts.cute_baseline_cli._run_backend_evaluation", run_backend
    )

    result = main(
        [
            "--mode",
            "backend",
            "--tile-m",
            "128",
            "--tile-n",
            "256",
            "--cluster-m",
            "2",
            "--cluster-n",
            "1",
            "--shared-memory-swizzle",
            "sw64",
        ]
    )

    assert result == 0
    assert calls[0][1]["shared_memory_swizzle"] == "sw64"


def test_canonical_epilogue_validation_checks_both_cached_states(monkeypatch) -> None:
    def run_backend(*, schedule, epilogue_stages):
        assert schedule.as_dict() == {
            "tile_m": 128,
            "tile_n": 256,
            "cluster_m": 2,
            "cluster_n": 1,
        }
        return {
            "representation": {
                "schema_version": 3,
                **schedule.as_dict(),
                "epilogue_stages": epilogue_stages,
            },
            "configuration_hash": f"configuration-{epilogue_stages}",
            "cache_validation": {"artifact_reused": True},
            "evaluation": {
                "status": "VALID",
                "correctness_status": "PASS",
                "metadata": {
                    "runtime_fingerprint": {"sha256": f"binary-{epilogue_stages}"}
                },
            },
        }

    monkeypatch.setattr(
        "kernel_mcts.cute_baseline_cli._run_backend_evaluation", run_backend
    )

    report = _run_canonical_epilogue_validation()

    assert report["status"] == "ok"
    assert set(report["variants"]) == {"2", "3"}
    assert all(report["checks"].values())


def test_canonical_epilogue_validation_reports_identical_runtime_artifacts(
    monkeypatch,
) -> None:
    def run_backend(*, schedule, epilogue_stages):
        return {
            "representation": {
                "schema_version": 3,
                **schedule.as_dict(),
                "epilogue_stages": epilogue_stages,
            },
            "configuration_hash": f"configuration-{epilogue_stages}",
            "cache_validation": {"artifact_reused": True},
            "evaluation": {
                "status": "VALID",
                "correctness_status": "PASS",
                "metadata": {
                    "runtime_fingerprint": {"sha256": "same-binary"}
                },
            },
        }

    monkeypatch.setattr(
        "kernel_mcts.cute_baseline_cli._run_backend_evaluation", run_backend
    )

    report = _run_canonical_epilogue_validation()

    assert report["status"] == "validation_failed"
    assert report["checks"]["distinct_runtime_fingerprints"] is False


def test_canonical_swizzle_validation_checks_cache_artifacts_and_profiles(
    monkeypatch,
) -> None:
    def run_backend(profile_set, *, schedule, shared_memory_swizzle):
        assert profile_set == "diagnostic_v2"
        return {
            "representation": {
                "schema_version": 3,
                **schedule.as_dict(),
                "shared_memory_swizzle": shared_memory_swizzle,
            },
            "configuration_hash": f"configuration-{shared_memory_swizzle}",
            "cache_validation": {"artifact_reused": True},
            "evaluation": {
                "status": "VALID",
                "correctness_status": "PASS",
                "metadata": {
                    "runtime_fingerprint": {
                        "sha256": f"binary-{shared_memory_swizzle}"
                    }
                },
            },
            "profile": {"metric_set": "diagnostic_v2"},
            "profile_did_not_change_reward": True,
        }

    monkeypatch.setattr(
        "kernel_mcts.cute_baseline_cli._run_backend_evaluation", run_backend
    )

    report = _run_canonical_swizzle_validation()

    assert report["status"] == "ok"
    assert set(report["variants"]) == {"heuristic", "sw64"}
    assert all(report["checks"].values())


def test_cli_requires_complete_canonical_schedule() -> None:
    with pytest.raises(ValueError, match="must be provided together"):
        main(["--mode", "backend", "--cluster-m", "2"])


def test_cli_rejects_canonical_schedule_outside_backend_modes() -> None:
    with pytest.raises(ValueError, match="only in backend modes"):
        main(
            [
                "--mode",
                "comparison",
                "--tile-m",
                "128",
                "--tile-n",
                "256",
                "--cluster-m",
                "2",
                "--cluster-n",
                "1",
            ]
        )


def test_epilogue_diagnostic_requires_stage_argument() -> None:
    with pytest.raises(ValueError, match="requires --epilogue-stages"):
        main(["--mode", "epilogue-stage-diagnostic"])


def test_cli_accepts_artifact_diagnostic_mode() -> None:
    arguments = build_parser().parse_args(["--mode", "diagnostic"])

    assert arguments.mode == "diagnostic"


def test_cli_accepts_structural_capabilities_mode() -> None:
    arguments = build_parser().parse_args(["--mode", "structural-capabilities"])

    assert arguments.mode == "structural-capabilities"


def test_cli_accepts_backend_profile_mode_and_metric_set() -> None:
    arguments = build_parser().parse_args(
        ["--mode", "backend-profile", "--profile-set", "diagnostic_v2"]
    )

    assert arguments.mode == "backend-profile"
    assert arguments.profile_set == "diagnostic_v2"


def test_design_space_report_contains_typed_mutations_without_budget() -> None:
    arguments = build_parser().parse_args(["--mode", "design-space"])
    report = _describe_design_space()

    assert arguments.mode == "design-space"
    assert report["proposal_count"] == 6
    assert report["budget"] == {"b_gen": 0, "b_tune": 0}
    assert all(
        proposal["proposal_mechanism"] == "typed_mutation"
        for proposal in report["proposals"]
    )


def test_cli_accepts_independent_lowering_binding_diagnostic() -> None:
    arguments = build_parser().parse_args(
        ["--mode", "independent-lowering-bindings"]
    )

    assert arguments.mode == "independent-lowering-bindings"


def test_cli_accepts_independent_tma_copy_diagnostic() -> None:
    arguments = build_parser().parse_args(
        [
            "--mode",
            "independent-tma-copy",
            "--independent-tma-stage",
            "compile_only",
        ]
    )

    assert arguments.mode == "independent-tma-copy"
    assert arguments.independent_tma_stage == "compile_only"


def test_generated_module_loader_materializes_inspectable_source(
    tmp_path: Path,
) -> None:
    path = tmp_path / "generated.py"
    module = _load_generated_module(
        "def value():\n    return 7\n",
        module_name="kernel_mcts_test_generated_module",
        path=path,
    )

    try:
        assert path.read_text(encoding="utf-8").startswith("def value")
        assert module.value() == 7
        assert module.value.__code__.co_filename == str(path)
    finally:
        sys.modules.pop(module.__name__, None)


def test_backend_manifest_adds_cute_libraries_and_driver() -> None:
    manifest = EnvironmentManifest(
        worker_id="worker",
        provider="standalone",
        gpu_model="H100",
        compute_capability="9.0",
        captured_at="2026-09-19T00:00:00+00:00",
        toolchain_versions={"cuda_toolkit": "12.9.1"},
    )

    enriched = _enrich_cute_manifest(
        manifest,
        cutlass_version="4.5.1",
        pytorch_version="2.8.0+cu129",
        driver_version={"driver": "580.173.02"},
    )

    assert enriched.toolchain_versions["driver"] == "580.173.02"
    assert enriched.library_versions == {
        "cutlass_dsl": "4.5.1",
        "pytorch": "2.8.0+cu129",
    }


def test_backend_report_separates_initial_jit_from_cached_lookup() -> None:
    manifest = EnvironmentManifest(
        worker_id="worker",
        provider="standalone",
        gpu_model="H100",
        compute_capability="9.0",
        captured_at="2026-09-19T00:00:00+00:00",
    )
    initial = SimpleNamespace(
        artifact=SimpleNamespace(artifact_id="artifact"),
        duration_seconds=12.5,
        stdout="initial CuTe JIT/evaluation",
        stderr="",
        artifact_paths=("/tmp/candidate.py",),
    )
    evaluation = EvaluationResult(
        ProposalStatus.VALID,
        KernelProgram("source", "cute_dsl"),
        "state",
        0.0,
        BenchmarkResult((100.0,), 100.0),
        compile_status=CompileStatus.SUCCESS,
        correctness_status=CorrectnessStatus.PASS,
        compilation=CompilationEvidence(
            "artifact", "reused cached CuTe JIT artifact", "", 0.001
        ),
    )

    report = _backend_evaluation_report(manifest, initial, evaluation)

    assert report["initial_jit_evaluation"]["duration_seconds"] == 12.5
    assert report["cache_validation"] == {
        "artifact_reused": True,
        "evaluator_compile_duration_seconds": 0.001,
        "evaluator_compile_stdout": "reused cached CuTe JIT artifact",
    }
