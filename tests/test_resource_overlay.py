"""Generic PTX-to-resource annotation, including attention-style operations."""

from __future__ import annotations

import pytest

from kernel_mcts.resource_overlay import overlay_ptx_dataflow


LINKS = ("hbm_l2", "l2_l1", "l1_registers", "shared_registers",
         "l1_shared", "l2_shared", "registers_scalar", "shared_tensor",
         "tensor_registers")


def _hardware() -> dict:
    return {"purpose": "diagnostic_only", "manifest_id": "test-h100",
            "links": [{"id": link} for link in LINKS]}


def test_attention_style_operations_keep_paths_and_uncertainty():
    graph = {
        "status": "partial", "ptx_sha256": "abc", "entry_point": "attention",
        "uncertainty": ["control_flow_join_or_loop"],
        "nodes": [
            {"id": "q", "kind": "load", "address_space": "global",
             "opcode": "ld.global.cg.b32"},
            {"id": "score", "kind": "arithmetic", "opcode": "ex2.approx.f32"},
            {"id": "reduce", "kind": "arithmetic", "opcode": "shfl.sync.bfly.b32"},
            {"id": "shared", "kind": "load", "address_space": "shared",
             "opcode": "ld.shared.u32"},
            {"id": "matrix_store", "kind": "unsupported", "address_space": None,
             "opcode": "stmatrix.sync.aligned.m8n8.x4.shared.b16"},
            {"id": "mma", "kind": "tensor", "opcode": "wgmma.mma_async.sync.aligned"},
            {"id": "wait", "kind": "tensor", "opcode": "wgmma.wait_group.sync.aligned"},
            {"id": "out", "kind": "store", "address_space": "global",
             "opcode": "st.global.b32"},
        ],
        "edges": [{"source": "q", "target": "score", "kind": "ptx_def_use"},
                  {"source": "score", "target": "reduce", "kind": "ptx_def_use"}],
    }
    result = overlay_ptx_dataflow(
        graph, _hardware(), source_sha256="source-hash",
        run_id="run", node_id="node",
        whole_kernel_evidence={"scope": "whole_kernel",
                               "ncu": {"dram_bytes": 100}}
    )
    by_id = {node["id"]: node["hardware"] for node in result["nodes"]}
    assert by_id["q"]["path_gap"] == "l1_bypass_load_route_not_modeled"
    assert by_id["score"]["candidate_graph_links"] == ["registers_scalar"]
    assert by_id["reduce"]["operation_class"] == "warp_exchange"
    assert by_id["shared"]["candidate_graph_links"] == ["shared_registers"]
    assert by_id["matrix_store"]["operation_class"] == "shared_write"
    assert by_id["mma"]["candidate_graph_links"] == ["shared_tensor", "tensor_registers"]
    assert by_id["wait"]["operation_class"] == "synchronization"
    assert result["edges"] == graph["edges"]
    assert result["whole_kernel_evidence"]["ncu"]["dram_bytes"] == 100
    assert result["ptx_uncertainty"] == ["control_flow_join_or_loop"]
    assert result["binding"] == "per_node"
    assert result["source_sha256"] == "source-hash"


def test_overlay_rejects_site_scoped_ncu_evidence():
    graph = {"status": "ok", "nodes": [], "edges": []}
    with pytest.raises(ValueError, match="whole-kernel scope"):
        overlay_ptx_dataflow(graph, _hardware(), source_sha256="source-hash",
                             whole_kernel_evidence={"scope": "per_instruction"})


def test_overlay_rejects_unpaired_node_identity():
    graph = {"status": "ok", "nodes": [], "edges": []}
    with pytest.raises(ValueError, match="paired node identity"):
        overlay_ptx_dataflow(graph, _hardware(), source_sha256="source-hash",
                             run_id="run")
