from pathlib import Path
from types import SimpleNamespace

import pytest

from kernel_mcts.cute_baseline_cli import (
    _backend_evaluation_report,
    _describe_design_space,
    _enrich_cute_manifest,
    _run_wgmma_inflight_diagnostic,
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
