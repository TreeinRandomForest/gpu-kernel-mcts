from __future__ import annotations

import json

from kernel_mcts import cute_backend_runner
from kernel_mcts.cute_program import REFERENCE_CUTE_GEMM


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
    assert calls[0]["wgmma_configuration"] == "pinned_default"
    assert calls[0]["wgmma_inflight_groups"] == 1
    assert calls[0]["schedule"] == REFERENCE_CUTE_GEMM.schedule
    assert json.loads(capsys.readouterr().out) == {"status": "ok"}
