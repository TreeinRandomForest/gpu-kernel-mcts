from __future__ import annotations

import io
import hashlib
import json
import sqlite3
import shutil
import zipfile

import pytest

from kernel_mcts.benchmarks.bf16_gemm import (
    BF16_GEMM_WORKLOAD,
    load_bf16_gemm_smoke_candidate,
)
from kernel_mcts.dataflow import bf16_gemm_dataflow
from kernel_mcts.persistence import SQLiteTraceStore
from kernel_mcts.resource_graph import h100_sxm_resource_graph, ncu_resource_observations
from kernel_mcts.trace_browser import TraceBrowserError, TraceCatalog
from kernel_mcts.serialization import serialize_workload


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


def test_evidence_matches_selected_source_report_by_source_and_workload(tmp_path) -> None:
    path = _browser_trace(tmp_path)
    source = "import triton\n# selected source\n"
    source_sha = hashlib.sha256(source.encode()).hexdigest()
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE search_runs SET benchmark_id = ? WHERE run_id = 'run-1'",
                           ("bf16_gemm_4096_h100",))
        connection.execute("UPDATE nodes SET backend_type = ?, program_text = ? "
                           "WHERE run_id = 'run-1' AND node_id = 'right'",
                           ("triton_h100", source))
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "analysis.json").write_text(json.dumps({
        "schema_version": 1, "benchmark_id": "bf16_gemm_4096_h100",
        "worker_image": "worker:test", "metric_scope": "whole_kernel_NCU_replay",
        "observations": [{
            "variant": "selected", "source_sha256": source_sha,
            "report_sha256": "a" * 64, "sass_local_load_sites": 3,
            "sass_local_store_sites": 2, "metrics": {"local_load_sectors": 80},
            "top_local_source_lines": [{"line": 4, "source": "tl.dot", "load_sites": 3}],
        }],
    }))
    catalog = TraceCatalog(tmp_path, reports)
    trace_id = catalog.list_traces()[0]["trace_id"]
    before = path.stat().st_mtime_ns
    selected = catalog.evidence(trace_id, "run-1", "right")["source_report"]
    assert selected["provenance"] == "separate_selected_node_measurement"
    assert selected["sass_local_load_sites"] == 3
    assert selected["metrics"]["local_load_sectors"] == 80
    assert catalog.evidence(trace_id, "run-1", "left")["source_report"] is None
    assert path.stat().st_mtime_ns == before


def test_hardware_view_reads_persisted_graph_and_profile_overlay(tmp_path) -> None:
    path = _browser_trace(tmp_path)
    manifest = {
        "manifest_id": "h100-worker", "gpu_model": "NVIDIA H100 80GB HBM3",
        "form_factor": "SXM", "compute_capability": "9.0", "gpu_count": 1,
        "total_memory_bytes": 85_520_809_984, "mig_configuration": {},
    }
    graph = h100_sxm_resource_graph(manifest)
    profile = {
        "profiler": "ncu", "metric_set": "lightweight_v1",
        "metrics": {
            "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": {
                "value": 1.4, "unit": "%"
            }
        },
    }
    observation = ncu_resource_observations(graph, profile)
    workload = serialize_workload(BF16_GEMM_WORKLOAD)
    launch = workload["metadata"]["abi"]["launch"]
    dataflow = bf16_gemm_dataflow(load_bf16_gemm_smoke_candidate().source, workload, launch)
    with sqlite3.connect(path) as connection:
        connection.execute(
            """INSERT INTO environment_manifests(manifest_id, worker_id, provider, gpu_model,
               compute_capability, form_factor, captured_at, manifest_json)
               VALUES (?, 'worker', 'nebius', ?, '9.0', 'SXM', '2026-01-01', ?)""",
            (manifest["manifest_id"], manifest["gpu_model"], json.dumps(manifest)),
        )
        connection.execute(
            "UPDATE search_runs SET environment_manifest_id = ? WHERE run_id = 'run-1'",
            (manifest["manifest_id"],),
        )
        connection.execute(
            """INSERT INTO hardware_resource_graphs
               (run_id, manifest_id, graph_schema_version, graph_json, created_at)
               VALUES ('run-1', ?, 2, ?, '2026-01-01')""",
            (manifest["manifest_id"], json.dumps(graph)),
        )
        connection.execute(
            """UPDATE nodes SET program_text = ?, workload_json = ?, profile_json = ?,
               metadata_json = ?, launch_config_json = ?
               WHERE run_id = 'run-1' AND node_id = 'right'""",
            (
                load_bf16_gemm_smoke_candidate().source,
                json.dumps(workload),
                json.dumps(profile),
                json.dumps({"compiler_context": {
                    "architecture": "sm_90",
                    "ptxas": {"status": "parsed", "resources": {
                        "registers_per_thread": 31,
                        "static_shared_memory_bytes": 1024,
                    }},
                }}),
                json.dumps(launch),
            ),
        )
        connection.execute(
            """INSERT INTO hardware_resource_observations
               (run_id, node_id, graph_schema_version, observation_json, created_at)
               VALUES ('run-1', 'right', 2, ?, '2026-01-01')""",
            (json.dumps(observation),),
        )
        connection.execute(
            """INSERT INTO node_dataflow_graphs
               (run_id, node_id, graph_schema_version, graph_json, created_at)
               VALUES ('run-1', 'right', 3, ?, '2026-01-01')""",
            (json.dumps(dataflow),),
        )

    catalog = TraceCatalog(tmp_path)
    trace_id = catalog.list_traces()[0]["trace_id"]
    hardware = catalog.hardware(trace_id, "run-1")
    assert hardware["source"] == "persisted"
    assert len(hardware["graph"]["nodes"]) == 8
    assert len(hardware["graph"]["links"]) == 9
    assert hardware["observations"][0]["node_id"] == "right"
    assert hardware["observations"][0]["observations"][0]["value"] == 1.4
    right = next(row for row in hardware["kernels"] if row["node_id"] == "right")
    checks = {row["constraint_id"]: row for row in right["constraint_checks"]}
    assert checks["registers_per_thread"]["observed"]["value"] == 31
    assert checks["threads_per_block"]["observed"]["value"] == 256
    assert right["dataflow_source"] == "persisted"
    assert right["dataflow"]["summary"]["source_global_load_bytes"] == 256 * 1024**3
    root = next(row for row in hardware["kernels"] if row["node_id"] == "root")
    assert root["dataflow_source"] == "derived_from_node"
    assert root["dataflow"]["kind"] == "logical_and_static_sites"
    assert root["dataflow"]["summary"]["source_global_load_bytes"] is None

    # A stored v2 dataflow graph is rederived for display without writing.
    with sqlite3.connect(path) as connection:
        old_dataflow = {**dataflow, "schema_version": 2}
        connection.execute(
            "UPDATE node_dataflow_graphs SET graph_schema_version = 2, graph_json = ?",
            (json.dumps(old_dataflow),),
        )
    before = path.stat().st_mtime_ns
    hardware = catalog.hardware(trace_id, "run-1")
    right = next(row for row in hardware["kernels"] if row["node_id"] == "right")
    assert right["dataflow_source"] == "derived_from_node"
    assert right["dataflow"]["schema_version"] == 3
    assert path.stat().st_mtime_ns == before

    # A stored v1 graph is upgraded for display, without rewriting the trace.
    with sqlite3.connect(path) as connection:
        old_graph = {**graph, "schema_version": 1, "constraints": [graph["constraints"][-1]]}
        connection.execute(
            "UPDATE hardware_resource_graphs SET graph_schema_version = 1, graph_json = ?",
            (json.dumps(old_graph),),
        )
    before = path.stat().st_mtime_ns
    hardware = catalog.hardware(trace_id, "run-1")
    assert hardware["source"] == "derived_from_manifest"
    assert hardware["graph"]["schema_version"] == 2
    assert path.stat().st_mtime_ns == before

    # Old traces can be viewed without a schema migration or file write.
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TABLE node_dataflow_graphs")
        connection.execute("DROP TABLE hardware_resource_observations")
        connection.execute("DROP TABLE hardware_resource_graphs")
        connection.execute("PRAGMA user_version = 10")
    before = path.stat().st_mtime_ns
    hardware = catalog.hardware(trace_id, "run-1")
    assert hardware["source"] == "derived_from_manifest"
    assert hardware["graph"]["manifest_id"] == "h100-worker"
    assert hardware["observations"][0]["observations"][0]["value"] == 1.4
    right = next(row for row in hardware["kernels"] if row["node_id"] == "right")
    assert right["constraint_checks"][0]["observed"]["value"] == 31
    assert right["dataflow_source"] == "derived_from_node"
    assert right["dataflow"]["template"] == "direct_load_candidate"
    assert path.stat().st_mtime_ns == before


def test_hardware_view_derives_generated_cuda_node_graph_read_only(tmp_path) -> None:
    import hashlib

    path = _browser_trace(tmp_path)
    manifest = {
        "manifest_id": "h100-worker", "gpu_model": "NVIDIA H100 80GB HBM3",
        "form_factor": "SXM", "compute_capability": "9.0", "gpu_count": 1,
        "total_memory_bytes": 85_520_809_984, "mig_configuration": {},
    }
    source = load_bf16_gemm_smoke_candidate().source + "\n// generated variant\n"
    ptx = (b".version 8.7\n.target sm_90\n"
           b".visible .entry bf16_gemm_root(.param .u64 p0) {\n"
           b"ld.param.u64 %rd1, [p0];\nld.global.u32 %r1, [%rd1];\nret;\n}\n")
    sha256 = hashlib.sha256(ptx).hexdigest()
    with sqlite3.connect(path) as connection:
        connection.execute(
            """INSERT INTO environment_manifests(manifest_id, worker_id, provider, gpu_model,
               compute_capability, form_factor, captured_at, manifest_json)
               VALUES (?, 'worker', 'nebius', ?, '9.0', 'SXM', '2026-01-01', ?)""",
            (manifest["manifest_id"], manifest["gpu_model"], json.dumps(manifest)),
        )
        connection.execute(
            "UPDATE search_runs SET environment_manifest_id = ? WHERE run_id = 'run-1'",
            (manifest["manifest_id"],),
        )
        connection.execute(
            """UPDATE nodes SET program_text = ?, metadata_json = ?
               WHERE run_id = 'run-1' AND node_id = 'right'""",
            (source, json.dumps({"compiler_context": {
                "entry_point": "bf16_gemm_root",
                "ptx": {"artifacts": [{"sha256": sha256, "storage_status": "stored"}]},
                "sass": {
                "status": "parsed", "categories": {"global_load": 2, "global_store": 1}
            }}})),
        )
        connection.execute(
            "INSERT INTO compiler_artifacts(sha256, kind, size_bytes, content) VALUES (?, 'ptx', ?, ?)",
            (sha256, len(ptx), ptx),
        )
        connection.execute(
            "INSERT INTO node_compiler_artifacts(run_id, node_id, sha256) VALUES ('run-1', 'right', ?)",
            (sha256,),
        )
    before = path.stat().st_mtime_ns
    catalog = TraceCatalog(tmp_path)
    hardware = catalog.hardware(catalog.list_traces()[0]["trace_id"], "run-1")
    right = next(row for row in hardware["kernels"] if row["node_id"] == "right")
    assert right["dataflow_source"] == "derived_from_node"
    assert right["dataflow"]["kind"] == "logical_and_static_sites"
    assert right["dataflow"]["summary"]["static_sass_instruction_sites"]["global_load"] == 2
    assert right["dataflow"]["compiled_ptx"]["ptx_sha256"] == sha256
    assert right["dataflow"]["compiled_ptx"]["instruction_count"] == 3
    assert path.stat().st_mtime_ns == before

def test_hardware_view_handles_run_without_supported_gpu_graph(tmp_path) -> None:
    _browser_trace(tmp_path)
    catalog = TraceCatalog(tmp_path)
    trace_id = catalog.list_traces()[0]["trace_id"]
    assert catalog.hardware(trace_id, "run-1") == {
        "graph": None, "source": None, "observations": [], "kernels": []
    }


def test_reconciliation_evidence_reads_verified_ptx_without_writing(tmp_path) -> None:
    path = _browser_trace(tmp_path)
    raw = (b".version 8.7\n.target sm_90\n"
           b".visible .entry kernel(.param .u64 p0) {\n"
           b"ld.param.u64 %rd1, [p0];\nld.global.u32 %r1, [%rd1];\nret;\n}\n")
    digest = hashlib.sha256(raw).hexdigest()
    context = {
        "entry_point": "kernel",
        "ptx": {"artifacts": [{"sha256": digest, "storage_status": "stored"}]},
        "ptxas": {"status": "parsed", "resources": {"registers_per_thread": 32}},
        "sass": {"status": "parsed", "categories": {"global_load": 1}},
    }
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE nodes SET metadata_json = ? WHERE run_id = 'run-1' AND node_id = 'right'",
            (json.dumps({"compiler_context": context}),),
        )
        connection.execute(
            "INSERT INTO compiler_artifacts(sha256, kind, size_bytes, content) VALUES (?, 'ptx', ?, ?)",
            (digest, len(raw), raw),
        )
        connection.execute(
            "INSERT INTO node_compiler_artifacts(run_id, node_id, sha256) VALUES ('run-1', 'right', ?)",
            (digest,),
        )
    before = path.stat().st_mtime_ns
    catalog = TraceCatalog(tmp_path)
    trace_id = catalog.list_traces()[0]["trace_id"]
    evidence = catalog.evidence(trace_id, "run-1", "right")
    assert evidence["ptx"] == {"status": "stored", "sha256": digest, "text": raw.decode()}
    assert evidence["compiler"]["sass"]["categories"]["global_load"] == 1
    assert evidence["profile"]["metrics"]["occupancy"] == 45
    assert evidence["dataflow"]["views"]["paths"]["entry_point"] == "kernel"
    assert path.stat().st_mtime_ns == before

    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE compiler_artifacts SET content = ? WHERE sha256 = ?", (b"altered", digest)
        )
    evidence = catalog.evidence(trace_id, "run-1", "right")
    assert evidence["ptx"]["status"] == "hash_mismatch"
    assert evidence["ptx"]["text"] is None


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


def test_catalog_reads_trace_created_before_mutation_budget_columns(tmp_path) -> None:
    path = _browser_trace(tmp_path)
    with sqlite3.connect(path) as connection:
        connection.execute("ALTER TABLE search_runs DROP COLUMN mutation_budget")
        connection.execute("ALTER TABLE search_runs DROP COLUMN final_b_mut")
        connection.execute("ALTER TABLE generations DROP COLUMN b_mut")
        connection.execute("ALTER TABLE generations DROP COLUMN api_instructions")
        connection.execute("ALTER TABLE iterations DROP COLUMN b_mut")

    catalog = TraceCatalog(tmp_path)
    trace = catalog.list_traces()[0]

    assert "error" not in trace
    assert trace["runs"][0]["mutation_budget"] is None
    assert trace["runs"][0]["b_mut"] == 0
    graph = catalog.graph(trace["trace_id"], "run-1")
    assert graph["analysis"]["timeline"] == []
    node = catalog.node(trace["trace_id"], "run-1", "left")
    assert node["generations"][0]["b_mut"] == 0
    assert node["generations"][0]["api_instructions"] is None


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


def test_cross_run_comparison_aligns_timelines_strategies_and_configuration(tmp_path) -> None:
    first = _browser_trace(tmp_path)
    with sqlite3.connect(first) as connection:
        connection.execute(
            """INSERT INTO iterations(
                run_id, iteration, status, selected_strategy_id, leaf_node_id,
                backed_up_reward, b_gen, b_prior
            ) VALUES ('run-1', 1, 'VALID', 'stage', 'right', 0.7, 5, 0)"""
        )
    second = tmp_path / "second.sqlite"
    shutil.copyfile(first, second)
    with sqlite3.connect(second) as connection:
        connection.execute(
            """UPDATE search_runs SET config_json = '{"c_puct":6}',
                      generation_budget = 20, final_b_gen = 10,
                      model_name = 'other-model'
               WHERE run_id = 'run-1'"""
        )
        connection.execute("UPDATE nodes SET reward = 1.0 WHERE node_id = 'right'")
        connection.execute(
            """UPDATE iterations SET backed_up_reward = 1.0, b_gen = 10
               WHERE run_id = 'run-1' AND iteration = 1"""
        )

    catalog = TraceCatalog(tmp_path)
    traces = {entry["name"]: entry for entry in catalog.list_traces()}
    comparison = catalog.compare_runs(
        traces["run.sqlite"]["trace_id"],
        "run-1",
        traces["second.sqlite"]["trace_id"],
        "run-1",
    )

    assert comparison["comparable_workload"] is True
    assert comparison["comparable_hardware"] is True
    assert comparison["timeline_a"][-1]["b_gen"] == 5
    assert comparison["timeline_b"][-1]["b_gen"] == 10
    assert comparison["run_b"]["best_speedup"] == pytest.approx(2.7182818)
    assert comparison["summary_deltas"]["best_reward"] == pytest.approx(0.3)
    differences = {item["field"]: item for item in comparison["differences"]}
    assert differences["config.c_puct"] == {"field": "config.c_puct", "a": 12, "b": 6}
    assert differences["model_name"]["b"] == "other-model"
    assert next(
        item for item in comparison["strategies"] if item["strategy_id"] == "stage"
    )["visits_a"] == 1
    occupancy = next(
        item
        for item in comparison["best_profile_comparison"]
        if item["metric"] == "metrics.occupancy"
    )
    assert occupancy["delta"] == 0


def test_export_bundle_contains_sources_diff_profiles_path_and_decisions(tmp_path) -> None:
    path = _browser_trace(tmp_path)
    with sqlite3.connect(path) as connection:
        connection.execute(
            """INSERT INTO iterations(
                run_id, iteration, status, selected_strategy_id, leaf_node_id,
                backed_up_reward, b_gen, b_prior
            ) VALUES ('run-1', 1, 'VALID', 'stage', 'right', 0.7, 1, 0)"""
        )
        connection.execute(
            """INSERT INTO selection_decisions(
                run_id, iteration, step, node_id, selected_strategy_id,
                selection_mode, selected_child_node_id, total_action_visits,
                c_puct, c_ucb, c_pw, alpha_pw, k_max, existing_children,
                allowed_children
            ) VALUES ('run-1', 1, 0, 'root', 'stage', 'EXPAND', 'right', 0,
                      12, 1, 1, 0.5, 4, 0, 1)"""
        )

    catalog = TraceCatalog(tmp_path)
    trace_id = catalog.list_traces()[0]["trace_id"]
    filename, payload = catalog.export_bundle(trace_id, "run-1", "root", "right")

    assert filename.endswith(".zip")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        assert set(archive.namelist()) == {
            "manifest.json",
            "report.md",
            "node-a.cu",
            "node-b.cu",
            "source.diff",
            "node-a.json",
            "node-b.json",
            "profile-a.json",
            "profile-b.json",
            "relationship.json",
            "decisions.json",
            "run.json",
        }
        assert archive.read("node-a.cu") == b"root source\n"
        assert archive.read("node-b.cu") == b"right source\n"
        assert b"-root source" in archive.read("source.diff")
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["format_version"] == 1
        assert manifest["relationship"]["kind"] == "A_ANCESTOR_OF_B"
        assert manifest["relevant_decision_count"] == 1
        decisions = json.loads(archive.read("decisions.json"))
        assert decisions[0]["selected_strategy_id"] == "stage"
        assert b"stage" in archive.read("report.md")


def test_browser_exposes_cute_representation_and_uses_python_export(tmp_path) -> None:
    path = _browser_trace(tmp_path)
    representation = {"schema_version": 1, "tile_m": 128, "tile_n": 256}
    with sqlite3.connect(path) as connection:
        connection.execute(
            """UPDATE nodes SET backend_type = 'cute_dsl', representation_json = ?,
                      representation_schema_version = 1,
                      configuration_hash = 'config-hash'
               WHERE run_id = 'run-1' AND node_id = 'left'""",
            (json.dumps(representation),),
        )

    catalog = TraceCatalog(tmp_path)
    trace_id = catalog.list_traces()[0]["trace_id"]
    node = catalog.node(trace_id, "run-1", "left")
    assert node["representation"] == representation
    assert node["representation_schema_version"] == 1
    assert node["configuration_hash"] == "config-hash"

    _, payload = catalog.export_bundle(trace_id, "run-1", "root", "left")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        assert "node-a.cu" in archive.namelist()
        assert "node-b.py" in archive.namelist()
