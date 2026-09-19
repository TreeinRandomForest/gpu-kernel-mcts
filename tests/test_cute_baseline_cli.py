from types import SimpleNamespace

from kernel_mcts.cute_baseline_cli import (
    _backend_evaluation_report,
    _enrich_cute_manifest,
    build_parser,
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


def test_cli_accepts_backend_evaluation_mode() -> None:
    arguments = build_parser().parse_args(["--mode", "backend"])

    assert arguments.mode == "backend"


def test_cli_accepts_artifact_diagnostic_mode() -> None:
    arguments = build_parser().parse_args(["--mode", "diagnostic"])

    assert arguments.mode == "diagnostic"


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
