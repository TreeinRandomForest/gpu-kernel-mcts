from __future__ import annotations

import sqlite3

import pytest

from kernel_mcts.persistence import SQLiteTraceStore
from kernel_mcts.trace_browser import TraceBrowserError, TraceCatalog


def _browser_trace(tmp_path):
    path = tmp_path / "run.sqlite"
    SQLiteTraceStore(path).close()
    with sqlite3.connect(path) as connection:
        connection.execute(
            """INSERT INTO search_runs(
                run_id, benchmark_id, algorithm, config_json, started_at,
                ended_at, best_node_id, generation_budget, final_b_gen,
                final_iterations, model_name, hardware_json
            ) VALUES ('run-1', 'bf16', 'mcts', '{"c_puct":12}', '2026-01-01',
                      '2026-01-02', 'right', 10, 5, 4, 'test-model',
                      '{"gpu":"H100"}')"""
        )
        nodes = [
            ("root", "root-key", 0.0, "root source\n", '{"median_us":100}', None),
            (
                "left",
                "left-key",
                0.4,
                "left source\n",
                '{"median_us":67}',
                '{"metrics":{"occupancy":30,"dram":10}}',
            ),
            (
                "right",
                "right-key",
                0.7,
                "right source\n",
                '{"median_us":50}',
                '{"metrics":{"occupancy":45,"dram":8}}',
            ),
        ]
        connection.executemany(
            """INSERT INTO nodes(
                run_id, node_id, state_key, reward, program_text, benchmark_json,
                profile_json, backend_type, workload_json, hardware_json,
                metadata_json, launch_config_json
            ) VALUES ('run-1', ?, ?, ?, ?, ?, ?, 'cuda_cpp', '{}', '{}', '{}',
                      '{"block":[256,1,1]}')""",
            nodes,
        )
        strategies = [
            ("root", "tile", 0.5, 2, 0.5, 0.8),
            ("root", "stage", 0.5, 1, 0.4, 0.7),
            ("left", "merge", 1.0, 1, 0.7, 0.7),
        ]
        connection.executemany(
            """INSERT INTO strategy_edges(
                run_id, parent_node_id, strategy_id, prior, visits, value_sum,
                q_mean, q_max, proposal_count, generation_attempt_count,
                repair_generation_count, valid_proposal_count, invalid_proposal_count
            ) VALUES ('run-1', ?, ?, ?, ?, ?, ?, ?, 1, 1, 0, 1, 0)""",
            [(*row[:4], row[4] * row[3], row[4], row[5]) for row in strategies],
        )
        connection.executemany(
            """INSERT INTO realization_edges(
                run_id, parent_node_id, strategy_id, child_node_id,
                descents, value_sum, q_mean
            ) VALUES ('run-1', ?, ?, ?, ?, ?, ?)""",
            [
                ("root", "tile", "left", 2, 0.8, 0.4),
                ("root", "stage", "right", 1, 0.7, 0.7),
                ("left", "merge", "right", 1, 0.7, 0.7),
            ],
        )
        connection.execute(
            """INSERT INTO generations(
                generation_id, run_id, b_gen, parent_node_id, strategy_id,
                repair_attempt, proposal_status, invalid_reason, compile_status,
                correctness_status, metadata_json, reused_node, created_node_id
            ) VALUES ('g-left', 'run-1', 1, 'root', 'tile', 0, 'VALID', NULL,
                      'SUCCESS', 'PASS', '{}', 0, 'left')"""
        )
        connection.executemany(
            """INSERT INTO search_events(run_id, event_type, payload_json, created_at)
               VALUES ('run-1', ?, ?, '2026-01-01')""",
            [("run_started", '{"root_node_id":"root"}'), ("run_completed", "{}")],
        )
    return path


def test_catalog_lists_runs_and_graph_without_source(tmp_path) -> None:
    _browser_trace(tmp_path)
    catalog = TraceCatalog(tmp_path)

    traces = catalog.list_traces()
    assert len(traces) == 1
    assert traces[0]["name"] == "run.sqlite"
    assert traces[0]["runs"][0]["best_speedup"] == pytest.approx(2.0137527)

    graph = catalog.graph(traces[0]["trace_id"], "run-1")
    assert graph["root_node_id"] == "root"
    assert graph["best_node_id"] == "right"
    assert graph["historical_scores_available"] is False
    assert [node["depth"] for node in graph["nodes"] if node["node_id"] == "right"] == [1]
    assert "program_text" not in graph["nodes"][0]


def test_node_and_comparison_include_source_profile_diff_and_shortest_path(tmp_path) -> None:
    _browser_trace(tmp_path)
    catalog = TraceCatalog(tmp_path)
    trace_id = catalog.list_traces()[0]["trace_id"]

    node = catalog.node(trace_id, "run-1", "left")
    assert node["program_text"] == "left source\n"
    assert node["profile"]["metrics"]["occupancy"] == 30
    assert node["generations"][0]["strategy_id"] == "tile"

    comparison = catalog.compare(trace_id, "run-1", "root", "right")
    assert comparison["relationship"]["kind"] == "A_ANCESTOR_OF_B"
    assert [step["strategy_id"] for step in comparison["relationship"]["path"]] == [
        "stage"
    ]
    assert "-root source" in comparison["source_diff"]
    assert "+right source" in comparison["source_diff"]

    profile = catalog.compare(trace_id, "run-1", "left", "right")[
        "profile_comparison"
    ]
    occupancy = next(item for item in profile if item["metric"] == "metrics.occupancy")
    assert occupancy["delta"] == 15
    assert occupancy["percent_change"] == 50


def test_catalog_reports_invalid_sqlite_and_rejects_unknown_trace_id(tmp_path) -> None:
    (tmp_path / "other.sqlite").write_text("not sqlite", encoding="utf-8")
    catalog = TraceCatalog(tmp_path)

    traces = catalog.list_traces()
    assert traces[0]["runs"] == []
    assert "error" in traces[0]
    with pytest.raises(TraceBrowserError, match="trace file not found"):
        catalog.graph("not-an-id", "run-1")


def test_graph_exposes_exact_selection_decision_candidates(tmp_path) -> None:
    path = _browser_trace(tmp_path)
    with sqlite3.connect(path) as connection:
        connection.execute(
            """INSERT INTO iterations(
                run_id, iteration, status, b_gen, b_prior
            ) VALUES ('run-1', 1, 'VALID', 1, 0)"""
        )
        connection.execute(
            """INSERT INTO selection_decisions(
                run_id, iteration, step, node_id, selected_strategy_id,
                selection_mode, selected_child_node_id, total_action_visits,
                c_puct, c_ucb, c_pw, alpha_pw, k_max, existing_children,
                allowed_children
            ) VALUES ('run-1', 1, 0, 'root', 'tile', 'UCB', 'left', 3,
                      12, 1, 1, 0.5, 4, 1, 1)"""
        )
        connection.executemany(
            """INSERT INTO puct_candidates(
                run_id, iteration, step, strategy_id, prior, visits, q_mean,
                q_max, exploit_term, explore_term, total_score, selected
            ) VALUES ('run-1', 1, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                ("tile", 0.5, 2, 0.4, 0.8, 0.4, 1.0, 1.4, 1),
                ("stage", 0.5, 1, 0.7, 0.7, 0.7, 0.6, 1.3, 0),
            ],
        )
        connection.execute(
            """INSERT INTO ucb_candidates(
                run_id, iteration, step, child_node_id, descents, q_mean,
                exploit_term, explore_term, total_score, selected
            ) VALUES ('run-1', 1, 0, 'left', 2, 0.4, 0.4, 0.5, 0.9, 1)"""
        )

    catalog = TraceCatalog(tmp_path)
    trace_id = catalog.list_traces()[0]["trace_id"]
    graph = catalog.graph(trace_id, "run-1")

    assert graph["historical_scores_available"] is True
    decision = graph["selection_decisions"][0]
    assert decision["selected_strategy_id"] == "tile"
    assert [item["strategy_id"] for item in decision["puct_candidates"]] == [
        "tile",
        "stage",
    ]
    assert decision["ucb_candidates"][0]["selected"] == 1


def test_graph_analysis_builds_playback_and_strategy_summaries(tmp_path) -> None:
    path = _browser_trace(tmp_path)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE generations SET iteration = 1 WHERE generation_id = 'g-left'"
        )
        connection.execute(
            """INSERT INTO generations(
                generation_id, run_id, iteration, b_gen, parent_node_id,
                strategy_id, repair_attempt, proposal_status, invalid_reason,
                compile_status, correctness_status, metadata_json, reused_node
            ) VALUES ('g-repair', 'run-1', 2, 2, 'root', 'tile', 1,
                      'INVALID', 'COMPILE_FAILURE', 'FAIL', 'NOT_TESTED', '{}', 0)"""
        )
        connection.executemany(
            """INSERT INTO iterations(
                run_id, iteration, status, selected_strategy_id, leaf_node_id,
                backed_up_reward, b_gen, b_prior
            ) VALUES ('run-1', ?, ?, ?, ?, ?, ?, 0)""",
            [
                (1, "VALID", "tile", "left", 0.4, 1),
                (2, "INVALID", "tile", None, None, 2),
                (3, "VALID", "stage", "right", 0.7, 3),
            ],
        )

    catalog = TraceCatalog(tmp_path)
    trace_id = catalog.list_traces()[0]["trace_id"]
    graph = catalog.graph(trace_id, "run-1")

    assert next(node for node in graph["nodes"] if node["node_id"] == "left")[
        "created_iteration"
    ] == 1
    assert next(
        edge
        for edge in graph["realizations"]
        if edge["parent_node_id"] == "root" and edge["child_node_id"] == "left"
    )["created_iteration"] == 1
    timeline = graph["analysis"]["timeline"]
    assert [item["cumulative_best_reward"] for item in timeline] == [0.4, 0.4, 0.7]
    assert timeline[-1]["cumulative_best_node_id"] == "right"
    tile = next(
        item for item in graph["analysis"]["strategies"] if item["strategy_id"] == "tile"
    )
    assert tile["calls"] == 2
    assert tile["repair_calls"] == 1
    assert tile["valid_calls"] == 1
    assert tile["invalid_calls"] == 1
    assert tile["completed_proposals"] == 2
    assert tile["valid_outcomes"] == 1


def test_trace_directory_must_exist(tmp_path) -> None:
    with pytest.raises(TraceBrowserError, match="does not exist"):
        TraceCatalog(tmp_path / "missing")
