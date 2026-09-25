from __future__ import annotations

import json

from kernel_mcts import cute_backend_runner
from kernel_mcts.cute_program import REFERENCE_CUTE_GEMM
from kernel_mcts.cute_independent import make_independent_cute_gemm
from kernel_mcts.cute_independent_tma import IndependentTmaDiagnosticSource


def test_runner_forwards_typed_pipeline_stages(monkeypatch, capsys) -> None:
    calls = []

    def run_comparable(**keywords):
        calls.append(keywords)
        return {"status": "ok"}

    monkeypatch.setattr(
        cute_backend_runner,
        "run_hopper_bf16_comparable",
        run_comparable,
    )
    representation = {
        **REFERENCE_CUTE_GEMM.as_dict(),
        "pipeline_stages": 3,
    }

    result = cute_backend_runner.main(
        ["--representation-json", json.dumps(representation)]
    )

    assert result == 0
    assert calls[0]["pipeline_stages"] == 3
    assert calls[0]["epilogue_stages"] is None
    assert calls[0]["wgmma_configuration"] == "pinned_default"
    assert calls[0]["smem_swizzle_policy"] == "heuristic"
    assert calls[0]["schedule"] == REFERENCE_CUTE_GEMM.schedule
    assert json.loads(capsys.readouterr().out) == {"status": "ok"}


def test_runner_forwards_canonical_sw64_policy(monkeypatch, capsys) -> None:
    calls = []

    def run_comparable(**keywords):
        calls.append(keywords)
        return {"status": "ok"}

    monkeypatch.setattr(
        cute_backend_runner,
        "run_hopper_bf16_comparable",
        run_comparable,
    )
    representation = {
        **REFERENCE_CUTE_GEMM.as_dict(),
        "cluster_m": 2,
        "cluster_n": 1,
        "shared_memory_swizzle": "sw64",
    }

    result = cute_backend_runner.main(
        ["--representation-json", json.dumps(representation)]
    )

    assert result == 0
    assert calls[0]["smem_swizzle_policy"] == "forced_sw64"
    assert json.loads(capsys.readouterr().out) == {"status": "ok"}


def test_runner_executes_independent_repository_contract(monkeypatch, capsys) -> None:
    calls = []

    def render(program, **keywords):
        calls.append((program, keywords))
        return IndependentTmaDiagnosticSource(
            program.configuration_hash,
            "def run_diagnostic():\n    return {'status': 'ok'}\n",
        )

    monkeypatch.setattr(
        cute_backend_runner,
        "render_independent_tma_copy_diagnostic",
        render,
    )
    representation = make_independent_cute_gemm()

    result = cute_backend_runner.main(
        [
            "--representation-kind",
            "independent",
            "--representation-json",
            representation.canonical_json(),
        ]
    )

    assert result == 0
    assert calls == [
        (
            representation,
            {
                "debug_stage": "wgmma_full_workload",
                "repository_contract": True,
                "profile_single_launch": False,
            },
        )
    ]
    assert json.loads(capsys.readouterr().out) == {"status": "ok"}
