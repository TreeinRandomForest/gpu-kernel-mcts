"""Controller safeguards for the CUDA Graph ablation runner."""

from __future__ import annotations

import runpy
from pathlib import Path
from types import SimpleNamespace


RUNNER = runpy.run_path(str(
    Path(__file__).resolve().parents[1]
    / "docs/experiments/triton-h100-graph-ablations/run.py"
))


def test_context_and_behavior_arms_are_distinct() -> None:
    arms_for_group = RUNNER["arms_for_group"]
    assert arms_for_group("context") == (
        "mcts-n0-c0", "mcts-n0-c1", "mcts-n1-c0", "mcts-n1-c1",
    )
    assert len(arms_for_group("behavior")) == 5
    assert arms_for_group("depth") == ("seq-free-h0",)


def test_transient_gateway_failure_is_recorded_as_invalid_generation(monkeypatch) -> None:
    class FailingGenerator:
        def generate(self, request):
            raise RuntimeError("Messages API transport failed")

    wrapper = RUNNER["BoundedGatewayGenerator"](FailingGenerator(), is_mcts=True)
    monkeypatch.setitem(wrapper.generate.__globals__, "build_generation_prompt", lambda request: "prompt")
    result = wrapper.generate(SimpleNamespace())
    assert result.program is None
    assert result.prompt_text == "prompt"
    assert result.metadata["llm_call"] is True
    assert result.metadata["gateway_error"] == "Messages API transport failed"
    assert wrapper.consecutive_failures == 1
