from __future__ import annotations

import sqlite3
import subprocess

import pytest

from kernel_mcts.persistence import SQLiteTraceStore
from kernel_mcts.trace_viz import (
    TraceGraphOptions,
    TraceVisualizationError,
    render_trace_dot,
    write_trace_visualization,
)


def _trace_database(tmp_path):
    path = tmp_path / "trace.sqlite"
    SQLiteTraceStore(path).close()
    with sqlite3.connect(path) as connection:
        connection.execute(
            """INSERT INTO search_runs(
                run_id, benchmark_id, algorithm, config_json, started_at,
                ended_at, best_node_id, generation_budget, final_b_gen,
                final_b_prior, final_iterations, model_name
            ) VALUES ('run-1', 'bf16', 'mcts', '{}', '2026-01-01',
                      '2026-01-02', 'child', 5, 3, 0, 2, 'test-model')"""
        )
        connection.executemany(
            """INSERT INTO nodes(
                run_id, node_id, state_key, program_text, backend_type, reward,
                workload_json, hardware_json, benchmark_json, profile_json,
                metadata_json, launch_config_json
            ) VALUES ('run-1', ?, ?, 'omitted source', 'cuda_cpp', ?, '{}', '{}',
                      ?, NULL, '{}', '{}')""",
            [
                ("root", "root-key", 0.0, '{"median_us": 100.0}'),
                ("child", "child-key", 0.693147, '{"median_us": 50.0}'),
            ],
        )
        connection.executemany(
            """INSERT INTO strategy_edges(
                run_id, parent_node_id, strategy_id, prior, visits, value_sum,
                q_mean, q_max, proposal_count, generation_attempt_count,
                repair_generation_count, valid_proposal_count, invalid_proposal_count
            ) VALUES ('run-1', 'root', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                ("coalesce", 0.6, 3, 2.0, 0.6667, 0.8, 2, 2, 0, 1, 1),
                ("tile", 0.4, 1, 0.693147, 0.693147, 0.693147, 1, 1, 0, 1, 0),
            ],
        )
        connection.executemany(
            """INSERT INTO realization_edges(
                run_id, parent_node_id, strategy_id, child_node_id,
                descents, value_sum, q_mean
            ) VALUES ('run-1', 'root', ?, 'child', ?, ?, ?)""",
            [
                ("coalesce", 2, 1.386294, 0.693147),
                ("tile", 1, 0.693147, 0.693147),
            ],
        )
        connection.execute(
            """INSERT INTO generations(
                generation_id, run_id, b_gen, parent_node_id, strategy_id,
                repair_attempt, proposal_status, invalid_reason, compile_status,
                correctness_status, metadata_json, reused_node
            ) VALUES ('generation-invalid', 'run-1', 1, 'root', 'coalesce',
                      0, 'INVALID', 'COMPILE_FAILURE', 'FAILED',
                      'NOT_ATTEMPTED', '{}', 0)"""
        )
        connection.executemany(
            """INSERT INTO search_events(run_id, event_type, payload_json, created_at)
               VALUES ('run-1', ?, ?, '2026-01-01')""",
            [
                ("run_started", '{"root_node_id":"root"}'),
                ("run_completed", '{}'),
            ],
        )
    return path


def test_dot_renders_dag_strategy_statistics_and_failed_proposals(tmp_path) -> None:
    path = _trace_database(tmp_path)

    dot, run_id = render_trace_dot(path)

    assert run_id == "run-1"
    assert "COMPLETED" in dot
    assert "B_gen=3/5" in dot
    assert "speedup=2.000x" in dot
    assert "Q_mean=0.6667 | Q_max=0.8000" in dot
    assert "PUCT" in dot
    assert "UCB | descents=2" in dot
    assert "B_gen=1 | INVALID" in dot
    assert "reason=COMPILE_FAILURE" in dot
    assert dot.count('  "child" [label=') == 1
    assert dot.count('-> "child"') == 2
    assert "omitted source" not in dot


def test_dot_filters_realizations_by_visits_and_shortest_depth(tmp_path) -> None:
    path = _trace_database(tmp_path)

    dot, _ = render_trace_dot(
        path,
        run_id="run-1",
        options=TraceGraphOptions(max_depth=1, min_edge_visits=2),
    )

    assert "strategy:root:coalesce" in dot
    assert "strategy:root:tile" not in dot
    assert "B_gen=1 | INVALID" not in dot


def test_png_render_invokes_graphviz_without_shell(tmp_path, monkeypatch) -> None:
    path = _trace_database(tmp_path)
    dot_path = tmp_path / "search.dot"
    png_path = tmp_path / "search.png"
    captured = {}

    monkeypatch.setattr("kernel_mcts.trace_viz.shutil.which", lambda name: "/usr/bin/dot")

    def run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        png_path.write_bytes(b"png")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("kernel_mcts.trace_viz.subprocess.run", run)

    selected = write_trace_visualization(path, dot_path, png_path=png_path)

    assert selected == "run-1"
    assert dot_path.read_text(encoding="utf-8").startswith("digraph mcts_search")
    assert png_path.read_bytes() == b"png"
    assert captured["command"] == [
        "/usr/bin/dot",
        "-Tpng",
        str(dot_path),
        "-o",
        str(png_path),
    ]
    assert captured["kwargs"]["check"] is False


def test_missing_graphviz_keeps_dot_and_reports_requirement(tmp_path, monkeypatch) -> None:
    path = _trace_database(tmp_path)
    dot_path = tmp_path / "search.dot"
    monkeypatch.setattr("kernel_mcts.trace_viz.shutil.which", lambda name: None)

    with pytest.raises(TraceVisualizationError, match="Graphviz 'dot'"):
        write_trace_visualization(path, dot_path, png_path=tmp_path / "search.png")

    assert dot_path.is_file()


def test_visualization_refuses_to_overwrite_outputs(tmp_path) -> None:
    path = _trace_database(tmp_path)
    dot_path = tmp_path / "search.dot"
    dot_path.write_text("existing", encoding="utf-8")

    with pytest.raises(TraceVisualizationError, match="refusing to overwrite"):
        write_trace_visualization(path, dot_path)

    assert dot_path.read_text(encoding="utf-8") == "existing"


def test_unknown_run_is_rejected(tmp_path) -> None:
    path = _trace_database(tmp_path)

    with pytest.raises(TraceVisualizationError, match="run ID not found"):
        render_trace_dot(path, run_id="missing")
