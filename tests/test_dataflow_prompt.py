from __future__ import annotations

import json

import pytest

from kernel_mcts import dataflow_prompt


def graph() -> dict[str, object]:
    return {"compiled_ptx": {
        "status": "partial", "ptx_sha256": "abc", "entry_point": "kernel",
        "instruction_count": 5, "block_count": 2,
        "uncertainty": ["memory_alias_or_runtime_path_unknown"],
        "nodes": [
            {"id": "param_0", "kind": "parameter", "opcode": None, "line": None, "block": None},
            {"id": "site_1", "kind": "parameter_load", "opcode": "ld.param.u64", "line": 4, "block": 0},
            {"id": "site_2", "kind": "load", "opcode": "ld.global.u32", "address_space": "global", "line": 5, "block": 0},
            {"id": "site_3", "kind": "arithmetic", "opcode": "add.u32", "line": 6, "block": 0},
            {"id": "site_4", "kind": "store", "opcode": "st.shared.u32", "address_space": "shared", "line": 7, "block": 0},
            {"id": "site_5", "kind": "load", "opcode": "ld.shared.u32", "address_space": "shared", "line": 10, "block": 1},
        ],
        "edges": [
            {"source": "param_0", "target": "site_1", "kind": "parameter_provenance", "register": None},
            {"source": "site_1", "target": "site_2", "kind": "ptx_def_use", "register": "%rd1"},
            {"source": "site_2", "target": "site_3", "kind": "ptx_def_use", "register": "%r1"},
            {"source": "site_3", "target": "site_4", "kind": "ptx_def_use", "register": "%r2"},
        ],
    }}


@pytest.mark.parametrize("view", ["counts", "paths", "subgraph", "full"])
def test_views_keep_proven_edges_and_unknown_memory_connection(view: str) -> None:
    result = dataflow_prompt.prompt_dataflow_view(graph(), view)
    assert result["view"] == view
    assert result["status"] == "partial"
    assert result["static_site_counts"]["load"] == 2
    assert result["static_memory_site_counts"]["load.global"] == 1
    assert result["static_memory_site_counts"]["load.shared"] == 1
    assert result["proven_edge_counts"]["ptx_def_use"] == 3
    assert result["uncertainty"] == ["memory_alias_or_runtime_path_unknown"]
    assert "measured bytes" in result["limits"]
    if view == "counts":
        assert "paths" not in result and "adjacency" not in result
    elif view == "paths":
        paths = result["paths"]
        assert any([site[0] for site in path["sites"]] ==
                   ["param_0", "site_1", "site_2", "site_3", "site_4"] for path in paths)
        assert not any("site_5" in [site[0] for site in path["sites"]]
                       and "site_4" in [site[0] for site in path["sites"]] for path in paths)
    else:
        adjacency = result["adjacency"]
        assert adjacency["site_fields"][0] == "id"
        assert ["site_4", "site_5", "ptx_def_use", None] not in adjacency["edges"]
        if view == "full":
            assert len(adjacency["sites"]) == 6
            assert len(adjacency["edges"]) == 4
        else:
            assert adjacency["omitted_sites"] == 0
    json.dumps(result)


def test_full_view_reports_size_limit_without_partial_adjacency(monkeypatch) -> None:
    monkeypatch.setattr(dataflow_prompt, "MAX_FULL_CHARS", 5)
    result = dataflow_prompt.prompt_dataflow_view(graph(), "full")
    assert result["status"] == "prompt_size_limit"
    assert "adjacency" not in result
    assert result["full_adjacency_chars"] > result["max_adjacency_chars"]


def test_unavailable_view_and_invalid_choice() -> None:
    assert dataflow_prompt.prompt_dataflow_view(None, "paths") == {
        "schema_version": 1, "view": "paths", "status": "unavailable"
    }
    with pytest.raises(ValueError, match="dataflow prompt view"):
        dataflow_prompt.prompt_dataflow_view(graph(), "none")
