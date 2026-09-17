from __future__ import annotations

from types import SimpleNamespace

import pytest

from kernel_mcts.autotune_cli import main
from kernel_mcts.autotuning import TuningResult
from kernel_mcts.domain import BenchmarkResult, EvaluationResult, KernelProgram, ProposalStatus
from kernel_mcts.provenance import RepositoryState


def test_standalone_autotune_cli_validates_before_provisioning(tmp_path, monkeypatch) -> None:
    source = tmp_path / "kernel.cu"
    source.write_text("kernel")
    monkeypatch.setattr(
        "kernel_mcts.autotune_cli.create_nebius_provider",
        lambda *args, **kwargs: pytest.fail("must validate before provisioning"),
    )

    with pytest.raises(SystemExit):
        main(
            [
                "--input", str(source), "--output", str(tmp_path / "out.cu"),
                "--trace", str(tmp_path / "trace.sqlite"), "--image", "worker:v1",
                "--nebius-project-id", "project", "--nebius-subnet-id", "subnet",
                "--nebius-username", "user", "--nebius-ssh-public-key", str(source),
                "--nebius-ssh-private-key", str(source), "--confirm-create-and-terminate",
            ]
        )


def test_standalone_autotune_cli_wires_tuner_and_exports_best(tmp_path, monkeypatch) -> None:
    source = tmp_path / "kernel.cu"
    source.write_text(
        "// KERNEL_MCTS_TUNE TILE=64,128\n#define TILE 64\nkernel\n"
    )
    output = tmp_path / "tuned.cu"
    trace = tmp_path / "trace.sqlite"
    captured = {}
    baseline = EvaluationResult(
        ProposalStatus.VALID,
        KernelProgram(source.read_text()),
        "base", 1.0, BenchmarkResult((10.0,), 10.0),
    )
    best = EvaluationResult(
        ProposalStatus.VALID,
        KernelProgram(source.read_text().replace("TILE 64", "TILE 128")),
        "best", 2.0, BenchmarkResult((5.0,), 5.0),
    )

    monkeypatch.setattr(
        "kernel_mcts.autotune_cli.capture_repository_state",
        lambda path: RepositoryState("a" * 40, False),
    )
    monkeypatch.setattr(
        "kernel_mcts.autotune_cli.create_nebius_provider",
        lambda config, **kwargs: captured.setdefault("provider_config", config) or object(),
    )

    def fake_run(**values):
        captured["run"] = values
        tuning = TuningResult((), (), best, {"TILE": 128}, 1.0)
        return SimpleNamespace(run_id="tune-run", baseline=baseline, tuning=tuning)

    monkeypatch.setattr("kernel_mcts.autotune_cli.run_standalone_autotuning", fake_run)

    assert main(
        [
            "--input", str(source), "--output", str(output), "--trace", str(trace),
            "--image", "worker:v1", "--tuning-budget", "7", "--tuning-method", "grid",
            "--nebius-project-id", "project", "--nebius-subnet-id", "subnet",
            "--nebius-username", "user", "--nebius-ssh-public-key", str(source),
            "--nebius-ssh-private-key", str(source), "--confirm-create-and-terminate",
        ]
    ) == 0

    assert "#define TILE 128" in output.read_text()
    assert captured["run"]["tuning_config"].budget == 7
    assert captured["run"]["tuning_config"].method == "grid"
