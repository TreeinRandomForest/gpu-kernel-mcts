"""Generic PTX-to-resource annotation, including attention-style operations."""

from __future__ import annotations

import pytest

from kernel_mcts.resource_overlay import overlay_ptx_dataflow


LINKS = ("hbm_l2", "l2_l1", "l1_registers", "shared_registers",
         "l1_shared", "l2_shared", "registers_scalar", "shared_tensor",
         "tensor_registers", "l2_registers", "l2_shared_cp_async",
         "registers_tensor", "registers_warp_exchange")


def _hardware() -> dict:
    return {"purpose": "diagnostic_only", "manifest_id": "test-h100",
            "topology_revision": 2,
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
    assert by_id["q"]["candidate_graph_links"] == ["hbm_l2", "l2_registers"]
    assert by_id["q"]["path_gap"] is None
    assert by_id["score"]["candidate_graph_links"] == ["registers_scalar"]
    assert by_id["reduce"]["operation_class"] == "warp_exchange"
    assert by_id["reduce"]["candidate_graph_links"] == ["registers_warp_exchange"]
    assert by_id["shared"]["candidate_graph_links"] == ["shared_registers"]
    assert by_id["matrix_store"]["operation_class"] == "shared_write"
    assert by_id["mma"]["candidate_graph_links"] == ["shared_tensor", "tensor_registers"]
    assert by_id["wait"]["operation_class"] == "synchronization"
    assert result["edges"] == graph["edges"]
    assert result["whole_kernel_evidence"]["ncu"]["dram_bytes"] == 100
    assert result["ptx_uncertainty"] == ["control_flow_join_or_loop"]
    assert result["binding"] == "per_node"
    assert result["source_sha256"] == "source-hash"
    assert result["hardware_graph_topology_revision"] == 2


def test_attention_copy_and_register_tensor_paths_do_not_claim_l1_or_shared_operand():
    graph = {
        "status": "partial", "nodes": [
            {"id": "copy", "kind": "async_copy",
             "opcode": "cp.async.cg.shared.global"},
            {"id": "mma", "kind": "tensor",
             "opcode": "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"},
        ], "edges": [],
    }
    result = overlay_ptx_dataflow(graph, _hardware(), source_sha256="source-hash")
    by_id = {node["id"]: node["hardware"] for node in result["nodes"]}
    assert by_id["copy"]["candidate_graph_links"] == ["hbm_l2", "l2_shared_cp_async"]
    assert "l1_shared" not in by_id["copy"]["candidate_graph_links"]
    assert by_id["mma"]["candidate_graph_links"] == ["registers_tensor", "tensor_registers"]
    assert "shared_tensor" not in by_id["mma"]["candidate_graph_links"]


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
