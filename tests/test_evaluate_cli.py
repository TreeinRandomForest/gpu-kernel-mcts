from __future__ import annotations

import json

import pytest

from kernel_mcts.domain import (
    BenchmarkResult,
    CompileStatus,
    CorrectnessEvidence,
    CorrectnessStatus,
    EvaluationResult,
    InvalidReason,
    KernelProgram,
    ProposalStatus,
)
from kernel_mcts.evaluate_cli import main
from kernel_mcts.orchestration import CandidateEvaluationExecution


def valid(program: KernelProgram, reward: float) -> EvaluationResult:
    return EvaluationResult(
        ProposalStatus.VALID,
        program,
        f"state:{program.source}",
        reward,
        BenchmarkResult((10.0, 11.0), 10.5, mean_us=10.5),
        compile_status=CompileStatus.SUCCESS,
        correctness_status=CorrectnessStatus.PASS,
        correctness=CorrectnessEvidence(0.01, 0.001),
    )


def arguments(candidate, report) -> list[str]:
    return [
        "--image",
        "worker:v1",
        "--candidate",
        str(candidate),
        "--report",
        str(report),
        "--network-volume-id",
        "volume-1",
        "--data-center-id",
        "EUR-IS-3",
        "--confirm-create-and-terminate",
    ]


def test_evaluation_cli_writes_complete_valid_report(
    tmp_path, monkeypatch, capsys
) -> None:
    candidate = tmp_path / "candidate.cu"
    report = tmp_path / "evaluation.json"
    candidate.write_text("candidate source")
    root = valid(KernelProgram("root source"), 0.0)
    evaluated = valid(KernelProgram("candidate source"), 0.5)
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
    monkeypatch.setattr(
        "kernel_mcts.evaluate_cli.resolve_reusable_volume",
        lambda parser, values, key: ("volume-1", "EUR-IS-3"),
    )
    monkeypatch.setattr(
        "kernel_mcts.evaluate_cli.create_runpod_provider", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        "kernel_mcts.evaluate_cli.run_candidate_evaluation",
        lambda **kwargs: CandidateEvaluationExecution(
            "evaluation-run",
            root,
            evaluated,
            {"manifest_id": "manifest-1", "gpu_model": "H100"},
        ),
    )

    result = main(arguments(candidate, report))

    assert result == 0
    payload = json.loads(report.read_text())
    assert payload["run_id"] == "evaluation-run"
    assert payload["candidate_evaluation"]["status"] == "VALID"
    assert payload["candidate_evaluation"]["correctness"]["maximum_error"] == 0.01
    assert payload["candidate_evaluation"]["benchmark"]["median_us"] == 10.5
    assert payload["environment_manifest"]["manifest_id"] == "manifest-1"
    output = capsys.readouterr().out
    assert "Candidate: status=VALID, compile=SUCCESS, correctness=PASS" in output
    assert "reward=0.5" in output


def test_evaluation_cli_returns_failure_for_invalid_candidate(
    tmp_path, monkeypatch
) -> None:
    candidate = tmp_path / "candidate.cu"
    report = tmp_path / "evaluation.json"
    candidate.write_text("bad source")
    root = valid(KernelProgram("root source"), 0.0)
    invalid = EvaluationResult(
        ProposalStatus.INVALID,
        program=KernelProgram("bad source"),
        invalid_reason=InvalidReason.COMPILE_FAILURE,
        compile_status=CompileStatus.FAIL,
    )
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
    monkeypatch.setattr(
        "kernel_mcts.evaluate_cli.resolve_reusable_volume",
        lambda parser, values, key: ("volume-1", "EUR-IS-3"),
    )
    monkeypatch.setattr(
        "kernel_mcts.evaluate_cli.create_runpod_provider", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        "kernel_mcts.evaluate_cli.run_candidate_evaluation",
        lambda **kwargs: CandidateEvaluationExecution(
            "evaluation-run",
            root,
            invalid,
            {"manifest_id": "manifest-1"},
        ),
    )

    assert main(arguments(candidate, report)) == 1
    assert json.loads(report.read_text())["candidate_evaluation"]["invalid_reason"] == (
        "COMPILE_FAILURE"
    )


def test_evaluation_cli_refuses_existing_report_before_provisioning(
    tmp_path, monkeypatch, capsys
) -> None:
    candidate = tmp_path / "candidate.cu"
    report = tmp_path / "evaluation.json"
    candidate.write_text("candidate source")
    report.write_text("existing")
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")

    with pytest.raises(SystemExit):
        main(arguments(candidate, report))

    assert "refusing to overwrite" in capsys.readouterr().err
    assert report.read_text() == "existing"
