from __future__ import annotations

import importlib.util
import sqlite3
import time
from pathlib import Path

import pytest

from kernel_mcts.domain import KernelProgram, ShapeCase, Strategy, WorkloadContract
from kernel_mcts.generation import GenerationRequest


def _runner_module():
    path = Path(__file__).resolve().parents[1] / "docs/experiments/triton-h100-context-factorial/run.py"
    spec = importlib.util.spec_from_file_location("triton_context_factorial_run", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resume_only_skips_a_closed_full_budget_trace(tmp_path) -> None:
    runner = _runner_module()
    path = tmp_path / "corrected-b10-s1-B.sqlite"
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE search_runs (ended_at TEXT, final_b_gen INTEGER);
            CREATE TABLE generations (generation_id TEXT);
            INSERT INTO search_runs VALUES ('2026-11-02', 10);
        """)
        connection.executemany("INSERT INTO generations VALUES (?)", [(str(i),) for i in range(10)])
    assert runner.completed_trace(path, 10)
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE search_runs SET final_b_gen = 3")
    assert not runner.completed_trace(path, 10)


def test_resume_archives_partial_trace_without_deleting_it(tmp_path, monkeypatch) -> None:
    runner = _runner_module()
    monkeypatch.setattr(runner, "OUTPUT", tmp_path)
    path = tmp_path / "corrected-b10-s1-B.sqlite"
    path.write_bytes(b"partial trace")
    path.with_suffix(".json").write_text("partial summary")
    archived = runner.archive_interrupted_trace(path)
    assert archived.read_bytes() == b"partial trace"
    assert archived.with_suffix(".json").read_text() == "partial summary"
    assert not path.exists()


def test_transient_gateway_error_is_a_budgeted_invalid_generation() -> None:
    runner = _runner_module()

    class Client:
        def complete(self, prompt):
            raise RuntimeError("Messages API returned HTTP 504")

    request = GenerationRequest(
        parent=KernelProgram("source", "triton_h100"),
        strategy=Strategy("tile", "tile", {"triton_h100": "tile"}),
        workload=WorkloadContract("toy", "gemm", "bf16",
                                  (ShapeCase({"M": 1}, 1.0),), 0.0, 0.0),
        hardware={"gpu_model": "H100"}, profile=None,
    )
    generator = runner.BoundedGatewayGenerator(Client())
    first = generator.generate(request)
    assert first.program is None
    assert first.prompt_text is not None
    assert first.metadata["gateway_error"] == "Messages API returned HTTP 504"
    assert first.metadata["llm_call"] is True
    generator.generate(request)
    with pytest.raises(RuntimeError, match="repeated transient failures"):
        generator.generate(request)


def test_hard_gateway_deadline_returns_an_invalid_generation() -> None:
    runner = _runner_module()

    class StalledClient:
        def complete(self, prompt):
            time.sleep(1)
            raise AssertionError("deadline did not interrupt the request")

    request = GenerationRequest(
        parent=KernelProgram("source", "triton_h100"),
        strategy=Strategy("tile", "tile", {"triton_h100": "tile"}),
        workload=WorkloadContract("toy", "gemm", "bf16",
                                  (ShapeCase({"M": 1}, 1.0),), 0.0, 0.0),
        hardware={"gpu_model": "H100"}, profile=None,
    )
    generator = runner.BoundedGatewayGenerator(StalledClient(), request_deadline_seconds=0.02)
    result = generator.generate(request)
    assert result.program is None
    assert result.metadata["gateway_error"] == "Messages API hard deadline exceeded"
    assert result.latency_seconds is not None and result.latency_seconds < 0.5
