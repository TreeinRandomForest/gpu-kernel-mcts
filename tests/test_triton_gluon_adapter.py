from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernel_mcts.benchmarks import load_triton_bf16_gemm_gluon_reference
from kernel_mcts import triton_harness


def test_gluon_reference_preserves_validated_kernel_body_and_search_launcher() -> None:
    adapted = load_triton_bf16_gemm_gluon_reference()
    original_path = (Path(__file__).resolve().parents[1] /
                     "docs/experiments/triton-h100-gluon-wait1/kernel.py")
    original = original_path.read_text(encoding="utf-8")
    adapted_tree = ast.parse(adapted.source)
    original_tree = ast.parse(original)

    def function(tree, name):
        return next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == name)

    assert adapted.backend == "triton_h100"
    assert ast.dump(function(adapted_tree, "gemm")) == ast.dump(function(original_tree, "gemm"))
    launcher = function(adapted_tree, "launch")
    assert [argument.arg for argument in launcher.args.args] == ["A", "B", "C", "M", "N", "K"]
    assert any(isinstance(node, ast.Attribute) and node.attr == "T"
               and isinstance(node.value, ast.Name) and node.value.id == "B"
               for node in ast.walk(launcher))


def test_triton_profile_harness_marks_only_target_launch(monkeypatch) -> None:
    events = []
    cuda = SimpleNamespace(
        is_available=lambda: True,
        synchronize=lambda: events.append("synchronize"),
        nvtx=SimpleNamespace(range_push=lambda name: events.append(("push", name)),
                             range_pop=lambda: events.append("pop")),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    monkeypatch.setattr(triton_harness, "_inputs", lambda *args: (object(), object(), object()))
    monkeypatch.setattr(triton_harness, "_candidate", lambda path: (
        lambda *args: events.append("launch")))
    args = argparse.Namespace(M=4096, N=4096, K=4096, seed=0,
                              candidate=Path("candidate.py"), mode="profile",
                              warmups=0)

    assert triton_harness.run(args) == {"status": "ok"}
    assert events == ["launch", "synchronize", "synchronize",
                      ("push", "KERNEL_MCTS_TARGET"), "launch", "pop", "synchronize"]


def test_triton_profile_harness_closes_range_after_candidate_error(monkeypatch) -> None:
    events = []
    cuda = SimpleNamespace(
        is_available=lambda: True,
        synchronize=lambda: None,
        nvtx=SimpleNamespace(range_push=lambda name: events.append("push"),
                             range_pop=lambda: events.append("pop")),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    monkeypatch.setattr(triton_harness, "_inputs", lambda *args: (object(), object(), object()))
    calls = iter((None, RuntimeError("launch failed")))

    def launch(*args):
        outcome = next(calls)
        if isinstance(outcome, Exception):
            raise outcome

    monkeypatch.setattr(triton_harness, "_candidate", lambda path: launch)
    args = argparse.Namespace(M=4096, N=4096, K=4096, seed=0,
                              candidate=Path("candidate.py"), mode="profile",
                              warmups=0)

    with pytest.raises(RuntimeError, match="launch failed"):
        triton_harness.run(args)
    assert events == ["push", "pop"]
