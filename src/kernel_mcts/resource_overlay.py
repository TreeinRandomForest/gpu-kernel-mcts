"""Conservative PTX dataflow annotation with candidate hardware resources.

This is operation-independent diagnostic evidence. Candidate routes are not
runtime cache destinations, per-site traffic, or a performance model.
"""

from __future__ import annotations

from collections import Counter
from typing import Mapping


SCHEMA_VERSION = 1


def _present(available: set[str], *links: str) -> list[str]:
    return [link for link in links if link in available]


def annotate_operation(node: Mapping[str, object], available_links: set[str]) -> dict[str, object]:
    kind = node.get("kind")
    space = node.get("address_space")
    opcode = str(node.get("opcode") or "")
    result: dict[str, object] = {
        "operation_class": kind,
        "resource_candidates": [],
        "candidate_graph_links": [],
        "path_gap": None,
        "attribution": "static_candidate_only",
    }
    if kind == "unsupported" and opcode.startswith(("stmatrix.", "ldmatrix.")) and ".shared" in opcode:
        result["operation_class"] = (
            "shared_write" if opcode.startswith("stmatrix.") else "shared_read"
        )
        result["resource_candidates"] = ["registers", "shared"]
        result["candidate_graph_links"] = _present(available_links, "shared_registers")
        result["attribution"] = "explicit_ptx_memory_space_without_def_use"
        return result
    if kind == "load" and space == "global":
        result["operation_class"] = "global_read"
        if ".cg." in opcode or opcode.endswith(".cg"):
            result["resource_candidates"] = ["hbm", "l2", "registers"]
            result["candidate_graph_links"] = _present(
                available_links, "hbm_l2", "l2_registers"
            )
            if "l2_registers" not in available_links:
                result["path_gap"] = "l1_bypass_load_route_not_modeled"
        else:
            result["resource_candidates"] = ["hbm", "l2", "l1", "registers"]
            result["candidate_graph_links"] = _present(
                available_links, "hbm_l2", "l2_l1", "l1_registers"
            )
        return result
    if kind == "store" and space == "global":
        result["operation_class"] = "global_write"
        result["resource_candidates"] = ["registers", "l1", "l2", "hbm"]
        result["candidate_graph_links"] = _present(
            available_links, "l1_registers", "l2_l1", "hbm_l2"
        )
        return result
    if kind in ("load", "store") and space == "shared":
        result["operation_class"] = "shared_read" if kind == "load" else "shared_write"
        result["resource_candidates"] = ["shared", "registers"]
        result["candidate_graph_links"] = _present(available_links, "shared_registers")
        result["attribution"] = "explicit_ptx_memory_space"
        return result
    if kind in ("load", "store") and space == "local":
        result["operation_class"] = "local_read" if kind == "load" else "local_write"
        result["resource_candidates"] = ["registers", "l1", "l2", "hbm"]
        result["path_gap"] = "local_memory_cache_route_not_modeled"
        return result
    if kind == "async_copy":
        result["operation_class"] = "async_copy"
        if opcode.startswith("cp.async.bulk.tensor"):
            result["resource_candidates"] = ["hbm", "l2", "shared"]
            result["candidate_graph_links"] = _present(available_links, "l2_shared")
        elif opcode.startswith("cp.async.cg."):
            result["resource_candidates"] = ["hbm", "l2", "shared"]
            result["candidate_graph_links"] = _present(
                available_links, "hbm_l2", "l2_shared_cp_async"
            )
            if "l2_shared_cp_async" not in available_links:
                result["path_gap"] = "l1_bypass_async_copy_route_not_modeled"
        else:
            result["resource_candidates"] = ["l1", "shared"]
            result["candidate_graph_links"] = _present(available_links, "l1_shared")
        return result
    if kind == "tensor":
        if any(word in opcode for word in ("fence", "commit_group", "wait_group")):
            result["operation_class"] = "synchronization"
            result["resource_candidates"] = ["sm"]
        elif opcode.startswith("wgmma.mma_async"):
            result["operation_class"] = "tensor_compute"
            result["resource_candidates"] = ["shared", "tensor_compute", "registers"]
            result["candidate_graph_links"] = _present(
                available_links, "shared_tensor", "tensor_registers"
            )
        elif opcode.startswith(("mma.sync", "wmma.mma")):
            result["operation_class"] = "tensor_compute"
            result["resource_candidates"] = ["registers", "tensor_compute"]
            result["candidate_graph_links"] = _present(
                available_links, "registers_tensor", "tensor_registers"
            )
            if "registers_tensor" not in available_links:
                result["path_gap"] = "register_tensor_operand_route_not_modeled"
        else:
            result["operation_class"] = "tensor_compute"
            result["resource_candidates"] = ["tensor_compute", "registers"]
            result["path_gap"] = "tensor_operand_route_not_inferred"
        return result
    if kind == "arithmetic":
        if opcode.startswith(("shfl.", "vote.")):
            result["operation_class"] = "warp_exchange"
            result["resource_candidates"] = ["registers", "warp_exchange"]
            result["candidate_graph_links"] = _present(
                available_links, "registers_warp_exchange"
            )
            if "registers_warp_exchange" not in available_links:
                result["path_gap"] = "warp_exchange_route_not_modeled"
        else:
            result["operation_class"] = "scalar_or_vector_compute"
            result["resource_candidates"] = ["registers", "scalar_compute"]
            result["candidate_graph_links"] = _present(available_links, "registers_scalar")
        return result
    if kind in ("synchronization", "control"):
        result["resource_candidates"] = ["sm"]
    return result


def overlay_ptx_dataflow(
    ptx_graph: Mapping[str, object],
    hardware_graph: Mapping[str, object],
    *,
    source_sha256: str,
    binary_fingerprint: str | None = None,
    run_id: str | None = None,
    node_id: str | None = None,
    whole_kernel_evidence: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Retain PTX paths and annotate operations without attributing NCU per site."""
    if ptx_graph.get("status") not in ("ok", "partial"):
        raise ValueError("PTX dataflow must be available")
    if hardware_graph.get("purpose") != "diagnostic_only":
        raise ValueError("hardware graph must be diagnostic")
    if not source_sha256 or (run_id is None) != (node_id is None):
        raise ValueError("overlay requires source identity and paired node identity")
    available = {link["id"] for link in hardware_graph.get("links", [])
                 if isinstance(link, Mapping) and isinstance(link.get("id"), str)}
    nodes = []
    classes: Counter[str] = Counter()
    gaps: Counter[str] = Counter()
    for original in ptx_graph.get("nodes", []):
        if not isinstance(original, Mapping):
            raise ValueError("invalid PTX node")
        annotation = annotate_operation(original, available)
        classes[str(annotation["operation_class"])] += 1
        if annotation["path_gap"] is not None:
            gaps[str(annotation["path_gap"])] += 1
        nodes.append({**original, "hardware": annotation})
    evidence = dict(whole_kernel_evidence or {})
    if evidence and evidence.get("scope") != "whole_kernel":
        raise ValueError("measured evidence must retain whole-kernel scope")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "partial_diagnostic",
        "ptx_sha256": ptx_graph.get("ptx_sha256"),
        "entry_point": ptx_graph.get("entry_point"),
        "source_sha256": source_sha256,
        "binary_fingerprint": binary_fingerprint,
        "run_id": run_id,
        "node_id": node_id,
        "binding": "per_node" if node_id is not None else "standalone_artifact",
        "hardware_graph_manifest_id": hardware_graph.get("manifest_id"),
        "hardware_graph_topology_revision": hardware_graph.get("topology_revision", 1),
        "nodes": nodes,
        "edges": list(ptx_graph.get("edges", [])),
        "static_operation_counts": dict(sorted(classes.items())),
        "path_gaps": dict(sorted(gaps.items())),
        "ptx_uncertainty": list(ptx_graph.get("uncertainty", [])),
        "whole_kernel_evidence": evidence,
        "limitations": (
            "PTX def-use is partial across control-flow joins and memory aliases. "
            "Hardware routes are candidates, cache destinations are not inferred, "
            "and whole-kernel counters are never assigned to instruction sites."
        ),
    }
