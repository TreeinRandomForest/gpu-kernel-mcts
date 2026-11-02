"""Compressed views must preserve evidence scope and exact PTX edges."""

from __future__ import annotations

import pytest

from kernel_mcts.overlay_view import exact_neighborhood, summarize_overlay


def _overlay():
    def node(identifier, operation):
        return {"id": identifier, "hardware": {
            "operation_class": operation,
            "candidate_graph_links": ["shared_registers"] if operation == "shared_read" else [],
        }}
    return {
        "source_sha256": "source", "ptx_sha256": "ptx",
        "nodes": [node("a", "shared_read"), node("b", "tensor_compute"),
                  node("c", "shared_read"), node("d", "tensor_compute")],
        "edges": [{"source": "a", "target": "b", "kind": "ptx_def_use"}],
        "path_gaps": {}, "ptx_uncertainty": ["memory_alias_or_runtime_path_unknown"],
    }


def test_summary_counts_observed_edges_without_connecting_unrelated_sites():
    summary = summarize_overlay(_overlay())
    assert summary["site_count"] == 4
    assert summary["sites_without_proven_edges"] == 2
    assert summary["candidate_link_sites"] == {"shared_registers": 2}
    assert summary["observed_class_edge_pairs"] == [
        {"source_class": "shared_read", "target_class": "tensor_compute", "edge_count": 1}
    ]


def test_exact_neighborhood_excludes_other_members_of_same_class():
    neighborhood = exact_neighborhood(_overlay(), "a", hops=1)
    assert [node["id"] for node in neighborhood["nodes"]] == ["a", "b"]
    assert neighborhood["edges"] == _overlay()["edges"]
    limited = exact_neighborhood(_overlay(), "a", hops=1, max_nodes=1)
    assert limited["truncated"] is True
    assert limited["edges"] == []


def test_invalid_edge_or_anchor_rejected():
    with pytest.raises(ValueError, match="anchor absent"):
        exact_neighborhood(_overlay(), "missing")
    overlay = _overlay()
    overlay["edges"].append({"source": "a", "target": "missing"})
    with pytest.raises(ValueError, match="missing node"):
        summarize_overlay(overlay)
