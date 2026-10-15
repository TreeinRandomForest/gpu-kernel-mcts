from __future__ import annotations

from kernel_mcts.benchmarks.bf16_gemm import (
    BF16_GEMM_WORKLOAD,
    load_bf16_gemm_root,
    load_bf16_gemm_smoke_candidate,
)
from kernel_mcts.dataflow import bf16_gemm_dataflow, cuda_node_dataflow
from kernel_mcts.serialization import serialize_workload


WORKLOAD = serialize_workload(BF16_GEMM_WORKLOAD)
LAUNCH = WORKLOAD["metadata"]["abi"]["launch"]


def test_tiled_root_dataflow_records_explicit_reuse_and_barriers() -> None:
    graph = bf16_gemm_dataflow(load_bf16_gemm_root().source, WORKLOAD, LAUNCH)
    assert graph is not None
    assert graph["schema_version"] == 3
    assert graph["template"] == "shared_tiled_root"
    assert graph["summary"]["source_global_load_bytes"] == 16 * 1024**3
    assert graph["summary"]["source_global_store_bytes"] == 32 * 1024**2
    assert graph["summary"]["barriers_per_cta"] == 512
    assert graph["summary"]["shared_reuse_per_loaded_operand"] == 16
    a_load = next(edge for edge in graph["edges"] if edge["id"] == "a_global_load")
    assert a_load["source_level_quantity"]["value"] == 8 * 1024**3
    assert a_load["hardware_links"] == ["l2_l1", "l1_registers", "shared_registers"]
    assert a_load["measured_link_bytes"] is None
    assert {node["memory_space"] for node in graph["nodes"]} == {
        "global", "shared", "register", "compute"
    }


def test_direct_candidate_dataflow_has_no_explicit_shared_staging() -> None:
    graph = bf16_gemm_dataflow(load_bf16_gemm_smoke_candidate().source, WORKLOAD, LAUNCH)
    assert graph is not None
    assert graph["template"] == "direct_load_candidate"
    assert graph["summary"]["source_global_load_bytes"] == 256 * 1024**3
    assert graph["summary"]["barriers_per_cta"] == 0
    assert graph["synchronization"] == []
    assert not any(node["memory_space"] == "shared" for node in graph["nodes"])
    a_load = next(edge for edge in graph["edges"] if edge["id"] == "a_global_load")
    assert a_load["target"] == "a_registers"


def test_dataflow_refuses_modified_source_or_unverified_contract() -> None:
    source = load_bf16_gemm_root().source
    assert bf16_gemm_dataflow(source + "\n", WORKLOAD, LAUNCH) is None
    assert bf16_gemm_dataflow(source, WORKLOAD, {**LAUNCH, "block": [32, 8, 1]}) is None
    assert bf16_gemm_dataflow(source, {**WORKLOAD, "benchmark_id": "other"}, LAUNCH) is None


def test_generated_cuda_node_separates_logical_tensors_from_static_sites() -> None:
    source = load_bf16_gemm_root().source + "\n// generated variant\n"
    context = {"sass": {"status": "parsed", "categories": {
        "global_load": 3, "global_store": 1, "shared_load": 2,
        "shared_store": 1, "tensor": 4, "arithmetic": 2,
    }}}
    graph = cuda_node_dataflow(source, WORKLOAD, LAUNCH, context)
    assert graph is not None
    assert graph["kind"] == "logical_and_static_sites"
    assert graph["evidence_status"] == "sass_categories"
    assert graph["summary"]["static_sass_instruction_sites"]["tensor"] == 4
    assert graph["summary"]["source_global_load_bytes"] is None
    assert {node["id"] for node in graph["nodes"] if not node["id"].startswith("sass_")} == {
        "a_contract", "b_contract", "logical_gemm", "c_contract"
    }
    assert {node["id"] for node in graph["nodes"] if node["id"].startswith("sass_")} == {
        "sass_global_load", "sass_global_store", "sass_shared_load", "sass_shared_store", "sass_tensor"
    }
    assert {edge["operation"] for edge in graph["edges"]} == {"logical_operand", "logical_output"}
    assert all(not edge["hardware_links"] for edge in graph["edges"])
    assert all(edge["source_level_quantity"]["value"] is None for edge in graph["edges"])
    assert all(edge["measured_link_bytes"] is None for edge in graph["edges"])


def test_generated_node_without_sass_keeps_unknowns_explicit() -> None:
    graph = cuda_node_dataflow("__global__ void kernel() {}", {}, {}, None)
    assert graph is not None
    assert graph["evidence_status"] == "structure_only"
    assert graph["summary"]["static_sass_instruction_sites"] is None
    assert not any(node["memory_space"] == "shared" for node in graph["nodes"])
