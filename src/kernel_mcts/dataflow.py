"""Bounded dataflow evidence for valid CUDA nodes."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from importlib.resources import files
from typing import Mapping


SCHEMA_VERSION = 1
TILE = 16
BF16_BYTES = 2


@lru_cache(maxsize=1)
def _known_sources() -> dict[str, str]:
    package = files("kernel_mcts.benchmarks").joinpath("kernels")
    return {
        "shared_tiled_root": package.joinpath("bf16_gemm_root.cu").read_text(encoding="utf-8"),
        "direct_load_candidate": package.joinpath("bf16_gemm_smoke_candidate.cu").read_text(encoding="utf-8"),
    }


def _fixed_shape(workload: Mapping[str, object], launch: Mapping[str, object]) -> tuple[int, int, int] | None:
    """Require the benchmark and launch contract actually inspected for this model."""
    if (
        workload.get("benchmark_id") != "bf16_gemm_4096_h100"
        or workload.get("operation") != "gemm"
        or workload.get("dtype") != "bfloat16"
        or launch.get("block") != [16, 16, 1]
        or launch.get("grid") != ["ceil_div(N, 16)", "ceil_div(M, 16)", 1]
        or launch.get("dynamic_shared_memory_bytes") != 0
    ):
        return None
    shapes = workload.get("shapes")
    metadata = workload.get("metadata")
    if not isinstance(shapes, list) or len(shapes) != 1 or not isinstance(metadata, Mapping):
        return None
    shape = shapes[0].get("dimensions") if isinstance(shapes[0], Mapping) else None
    tensors = metadata.get("tensors")
    if shape != {"M": 4096, "N": 4096, "K": 4096} or not isinstance(tensors, Mapping):
        return None
    if (
        metadata.get("accumulation_dtype") != "float32"
        or metadata.get("output_dtype") != "bfloat16"
        or any(
            not isinstance(tensors.get(name), Mapping)
            or tensors[name].get("layout") != layout
            or tensors[name].get("shape") != expected_shape
            for name, layout, expected_shape in (
                ("A", "row_major", ["M", "K"]),
                ("B", "column_major", ["K", "N"]),
                ("C", "row_major", ["M", "N"]),
            )
        )
    ):
        return None
    return (4096, 4096, 4096)


def _node(node_id: str, label: str, space: str, resource: str | None, column: int, lane: int) -> dict[str, object]:
    return {
        "id": node_id,
        "label": label,
        "memory_space": space,
        "hardware_resource_id": resource,
        "column": column,
        "lane": lane,
    }


def _edge(
    edge_id: str,
    source: str,
    target: str,
    operation: str,
    count: int | None,
    unit: str,
    hardware_links: list[str],
    *,
    note: str = "",
) -> dict[str, object]:
    return {
        "id": edge_id,
        "source": source,
        "target": target,
        "operation": operation,
        "source_level_quantity": {
            "value": count,
            "unit": unit,
            "basis": (
                "fixed_shape_source_operations"
                if unit == "source operations" else "fixed_shape_source_element_references"
            ) if count is not None else "unknown",
        },
        "hardware_links": hardware_links,
        "measured_link_bytes": None,
        "note": note,
    }


def bf16_gemm_dataflow(
    source: str,
    workload: Mapping[str, object],
    launch_config: Mapping[str, object],
) -> dict[str, object] | None:
    """Model only exact packaged sources under the fixed 4096³ workload.

    Counts describe source-level element references, not DRAM/L2 transactions or
    measured bandwidth. Cache effects and compiler lowering remain unknown.
    """
    if not isinstance(source, str) or _fixed_shape(workload, launch_config) is None:
        return None
    template = next((name for name, text in _known_sources().items() if source == text), None)
    if template is None:
        return None

    m = n = k = 4096
    outputs = m * n
    ctas = (m // TILE) * (n // TILE)
    k_tiles = k // TILE
    macs = outputs * k
    operand_refs = macs * BF16_BYTES
    output_bytes = outputs * BF16_BYTES
    shared_tiled = template == "shared_tiled_root"
    global_operand_bytes = ctas * k_tiles * TILE * TILE * BF16_BYTES if shared_tiled else operand_refs
    first_target_a = "a_shared" if shared_tiled else "a_registers"
    first_target_b = "b_shared" if shared_tiled else "b_registers"
    first_path = ["l2_l1", "l1_registers", "shared_registers"] if shared_tiled else ["l2_l1", "l1_registers"]
    nodes = [
        _node("a_global", "A · global BF16", "global", None, 0, 0),
        _node("b_global", "B · global BF16", "global", None, 0, 2),
    ]
    if shared_tiled:
        nodes += [
            _node("a_shared", "A tile · shared", "shared", "shared", 1, 0),
            _node("b_shared", "B tile · shared", "shared", "shared", 1, 2),
        ]
    register_column = 2 if shared_tiled else 1
    nodes += [
        _node("a_registers", "A operand · thread", "register", "registers", register_column, 0),
        _node("b_registers", "B operand · thread", "register", "registers", register_column, 2),
        _node("multiply_accumulate", "FP32 multiply-add", "compute", "scalar_compute", register_column + 1, 1),
        _node("accumulator", "FP32 accumulator", "register", "registers", register_column + 2, 1),
        _node("c_global", "C · global BF16", "global", None, register_column + 3, 1),
    ]
    edges = [
        _edge("a_global_load", "a_global", first_target_a, "global_load", global_operand_bytes, "bytes", first_path,
              note="Source-level load demand; cache hits and HBM bytes are unknown."),
        _edge("b_global_load", "b_global", first_target_b, "global_load", global_operand_bytes, "bytes", first_path,
              note="Source-level load demand; cache hits and HBM bytes are unknown."),
    ]
    if shared_tiled:
        edges += [
            _edge("a_shared_read", "a_shared", "a_registers", "shared_load", operand_refs, "bytes", ["shared_registers"]),
            _edge("b_shared_read", "b_shared", "b_registers", "shared_load", operand_refs, "bytes", ["shared_registers"]),
        ]
    edges += [
        _edge("a_compute_input", "a_registers", "multiply_accumulate", "operand", macs, "elements", ["registers_scalar"]),
        _edge("b_compute_input", "b_registers", "multiply_accumulate", "operand", macs, "elements", ["registers_scalar"]),
        _edge("accumulate", "multiply_accumulate", "accumulator", "multiply_accumulate", macs, "source operations", ["registers_scalar"]),
        _edge("c_global_store", "accumulator", "c_global", "global_store", output_bytes, "bytes", ["l1_registers", "l2_l1"],
              note="Source-level stores; actual L2/HBM bytes are unknown."),
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "bounded_source_model",
        "template": template,
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "shape": {"M": m, "N": n, "K": k},
        "nodes": nodes,
        "edges": edges,
        "summary": {
            "output_elements": outputs,
            "cta_count": ctas,
            "k_tiles_per_cta": k_tiles if shared_tiled else None,
            "source_global_load_bytes": 2 * global_operand_bytes,
            "source_global_store_bytes": output_bytes,
            "source_multiply_accumulates": macs,
            "shared_reuse_per_loaded_operand": TILE if shared_tiled else None,
            "barriers_per_cta": 2 * k_tiles if shared_tiled else 0,
        },
        "synchronization": (
            [{"kind": "cta_barrier", "count_per_k_tile": 2,
              "roles": ["after shared tile fill", "before shared tile overwrite"]}]
            if shared_tiled else []
        ),
        "ownership": "one C element per active thread; each thread accumulates over K",
        "uncertainty": [
            "Source element references are not measured memory transactions.",
            "Cache residency, HBM/L2 traffic, compiler transformations, and link latency are unknown.",
            "The hardware path lists possible resource links; it does not assign bytes to them.",
        ],
    }


def cuda_node_dataflow(
    source: str,
    workload: Mapping[str, object],
    launch_config: Mapping[str, object],
    compiler_context: Mapping[str, object] | None = None,
) -> dict[str, object] | None:
    """Return an exact source model or a conservative per-node operation graph.

    SASS categories count static instruction sites. They cannot establish value
    dependencies, execution counts, or memory traffic, so the fallback graph
    deliberately leaves all edge quantities unknown.
    """
    exact = bf16_gemm_dataflow(source, workload, launch_config)
    if exact is not None:
        return exact
    if not isinstance(source, str) or not source.strip():
        return None

    context = compiler_context if isinstance(compiler_context, Mapping) else {}
    sass = context.get("sass")
    categories = sass.get("categories") if isinstance(sass, Mapping) and sass.get("status") == "parsed" else None
    categories = categories if isinstance(categories, Mapping) else {}
    counts = {
        name: value for name in (
            "global_load", "global_store", "shared_load", "shared_store", "tensor", "arithmetic", "control"
        ) if isinstance((value := categories.get(name)), int) and not isinstance(value, bool) and value >= 0
    }
    evidenced = bool(counts)
    nodes = [
        _node("inputs", "Kernel inputs · global", "global", None, 0, 0),
        _node("input_registers", "Thread values", "register", "registers", 1, 0),
        _node("compute", "Kernel compute", "compute", "tensor_compute" if counts.get("tensor", 0) else "scalar_compute", 3, 0),
        _node("output_registers", "Result values", "register", "registers", 4, 0),
        _node("outputs", "Kernel outputs · global", "global", None, 5, 0),
    ]
    edges = []
    if counts.get("global_load", 0) or not evidenced:
        edges.append(_edge("global_load", "inputs", "input_registers", "global_load", None, "bytes", ["l2_l1", "l1_registers"] if evidenced else [], note="Possible route; static instructions do not identify the input tensor or runtime bytes."))
    if counts.get("shared_store", 0) or counts.get("shared_load", 0):
        nodes.append(_node("shared", "Shared storage", "shared", "shared", 2, 2))
        if counts.get("shared_store", 0):
            edges.append(_edge("shared_store", "input_registers", "shared", "shared_store", None, "bytes", ["shared_registers"], note="Static shared-store instruction exists; its producer and execution count are unknown."))
        if counts.get("shared_load", 0):
            edges.append(_edge("shared_load", "shared", "compute", "shared_load", None, "bytes", ["shared_registers"], note="Static shared-load instruction exists; its consumer and execution count are unknown."))
    edges += [
        _edge("compute_input", "input_registers", "compute", "possible_dependency", None, "elements", [], note="Value dependence is unresolved."),
        _edge("compute_output", "compute", "output_registers", "possible_dependency", None, "elements", [], note="Value dependence is unresolved."),
    ]
    if counts.get("global_store", 0) or not evidenced:
        edges.append(_edge("global_store", "output_registers", "outputs", "global_store", None, "bytes", ["l1_registers", "l2_l1"] if evidenced else [], note="Possible route; static instructions do not establish runtime bytes."))
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "compiler_operation_skeleton",
        "template": "compiler_operation_skeleton",
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "shape": None,
        "nodes": nodes,
        "edges": edges,
        "summary": {
            "source_global_load_bytes": None,
            "source_global_store_bytes": None,
            "source_multiply_accumulates": None,
            "barriers_per_cta": None,
            "static_sass_instruction_sites": counts if evidenced else None,
        },
        "synchronization": [],
        "ownership": "unknown",
        "evidence_status": "sass_categories" if evidenced else "structure_only",
        "uncertainty": [
            "This graph shows possible stages, not traced value dependencies.",
            "Static SASS instruction sites are not execution counts or measured traffic.",
            "Unknown quantities and absent edges must not be interpreted as zero.",
        ],
    }
