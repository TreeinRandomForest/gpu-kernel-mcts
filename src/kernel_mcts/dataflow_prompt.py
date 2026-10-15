"""Bounded, provenance-preserving prompt views of a recorded PTX graph."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from typing import Mapping


VIEWS = ("none", "counts", "paths", "subgraph", "full")
SCHEMA_VERSION = 1
MAX_SUBGRAPH_CHARS = 16_000
MAX_FULL_CHARS = 120_000
MAX_PATHS = 8
MAX_PATH_LENGTH = 6
MAX_PATHS_CHARS = 12_000
MAX_SUBGRAPH_NODES = 64
MAX_SUBGRAPH_EDGES = 128
_PRIORITY = ("store", "tensor", "async_copy", "atomic", "load", "arithmetic",
             "synchronization", "control", "parameter_load", "parameter", "unsupported")


def _site(node: Mapping[str, object]) -> list[object]:
    return [node.get("id"), node.get("kind"), node.get("opcode"),
            node.get("address_space"), node.get("line"), node.get("block"),
            node.get("predicated")]


def _edge(edge: Mapping[str, object]) -> list[object]:
    return [edge.get("source"), edge.get("target"), edge.get("kind"), edge.get("register")]


def _chars(value: object) -> int:
    return len(json.dumps(value, sort_keys=True, separators=(",", ":")))


def _anchors(nodes: list[Mapping[str, object]]) -> list[str]:
    by_kind: dict[str, list[str]] = defaultdict(list)
    for node in nodes:
        if isinstance(node.get("id"), str) and isinstance(node.get("kind"), str):
            by_kind[node["kind"]].append(node["id"])
    return [node_id for kind in _PRIORITY for node_id in by_kind[kind][:2]]


def _paths(nodes: list[Mapping[str, object]], edges: list[Mapping[str, object]]) -> list[dict[str, object]]:
    by_id = {node["id"]: node for node in nodes if isinstance(node.get("id"), str)}
    incoming: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for edge in edges:
        if edge.get("source") in by_id and edge.get("target") in by_id:
            incoming[edge["target"]].append(edge)
    result = []
    seen: set[tuple[str, ...]] = set()
    for anchor in _anchors(nodes):
        path = [anchor]
        path_edges: list[Mapping[str, object]] = []
        while len(path) < MAX_PATH_LENGTH:
            predecessors = [edge for edge in incoming[path[0]] if edge["source"] not in path]
            if not predecessors:
                break
            # Prefer a producer with its own predecessor, then stable PTX order.
            chosen = max(predecessors, key=lambda edge: (
                bool(incoming[edge["source"]]),
                int(str(edge["source"]).split("_")[-1])
                if str(edge["source"]).split("_")[-1].isdigit() else -1,
            ))
            path.insert(0, chosen["source"])
            path_edges.insert(0, chosen)
        key = tuple(path)
        if key in seen:
            continue
        seen.add(key)
        result.append({"sites": [_site(by_id[node_id]) for node_id in path],
                       "edges": [_edge(edge) for edge in path_edges]})
        if len(result) == MAX_PATHS:
            break
    return result


def _subgraph(nodes: list[Mapping[str, object]], edges: list[Mapping[str, object]]) -> dict[str, object]:
    by_id = {node["id"]: node for node in nodes if isinstance(node.get("id"), str)}
    adjacent: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        source, target = edge.get("source"), edge.get("target")
        if source in by_id and target in by_id:
            adjacent[source].add(target)
            adjacent[target].add(source)
    selected: set[str] = set()
    frontier = [node_id for node_id in _anchors(nodes) if node_id in by_id][:8]
    for _ in range(3):
        following = []
        for node_id in frontier:
            if len(selected) >= MAX_SUBGRAPH_NODES:
                break
            selected.add(node_id)
            following.extend(sorted(adjacent[node_id] - selected))
        frontier = list(dict.fromkeys(following))
    chosen_nodes = [_site(node) for node in nodes if node.get("id") in selected]
    chosen_edges = [_edge(edge) for edge in edges
                    if edge.get("source") in selected and edge.get("target") in selected]
    chosen_edges = chosen_edges[:MAX_SUBGRAPH_EDGES]
    result: dict[str, object] = {
        "site_fields": ["id", "kind", "opcode", "address_space", "ptx_line", "block", "predicated"],
        "edge_fields": ["source", "target", "evidence", "register"],
        "sites": chosen_nodes, "edges": chosen_edges,
        "selection": "up to two sites per operation family, then local neighbors",
        "omitted_sites": len(nodes) - len(chosen_nodes),
        "omitted_edges": len(edges) - len(chosen_edges),
    }
    while _chars(result) > MAX_SUBGRAPH_CHARS and chosen_nodes:
        removed = chosen_nodes.pop()[0]
        chosen_edges[:] = [edge for edge in chosen_edges if removed not in edge[:2]]
        result["omitted_sites"] = len(nodes) - len(chosen_nodes)
        result["omitted_edges"] = len(edges) - len(chosen_edges)
    return result


def prompt_dataflow_view(graph: Mapping[str, object] | None, view: str) -> dict[str, object]:
    """Serialize one recorded graph without inventing memory or runtime edges."""
    if view not in VIEWS or view == "none":
        raise ValueError("dataflow prompt view must be counts, paths, subgraph, or full")
    result: dict[str, object] = {"schema_version": SCHEMA_VERSION, "view": view,
                                 "status": "unavailable"}
    compiled = graph.get("compiled_ptx") if isinstance(graph, Mapping) else None
    if not isinstance(compiled, Mapping):
        return result
    nodes = compiled.get("nodes")
    edges = compiled.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        return result
    nodes = [node for node in nodes if isinstance(node, Mapping) and isinstance(node.get("id"), str)]
    edges = [edge for edge in edges if isinstance(edge, Mapping)
             and isinstance(edge.get("source"), str) and isinstance(edge.get("target"), str)]
    result.update({
        "status": compiled.get("status", "unavailable"),
        "ptx_sha256": compiled.get("ptx_sha256"),
        "entry_point": compiled.get("entry_point"),
        "instruction_count": compiled.get("instruction_count"),
        "block_count": compiled.get("block_count"),
        "static_site_counts": dict(sorted(Counter(str(node.get("kind")) for node in nodes).items())),
        "static_memory_site_counts": dict(sorted(Counter(
            f"{node.get('kind')}.{node.get('address_space')}" for node in nodes
            if node.get("address_space") is not None
        ).items())),
        "proven_edge_counts": dict(sorted(Counter(str(edge.get("kind")) for edge in edges).items())),
        "uncertainty": compiled.get("uncertainty", []),
        "limits": "Static PTX sites and local value dependencies only; no dynamic counts, "
                  "measured bytes, memory alias edges, or paths across control-flow joins.",
    })
    if view == "paths":
        paths = _paths(nodes, edges)
        selected_paths = []
        for path in paths:
            if _chars(selected_paths + [path]) > MAX_PATHS_CHARS:
                break
            selected_paths.append(path)
        result["paths"] = selected_paths
        result["omitted_selected_paths"] = len(paths) - len(selected_paths)
        result["path_selection"] = "up to eight operation-anchored local predecessor paths"
    elif view == "subgraph":
        result["adjacency"] = _subgraph(nodes, edges)
    elif view == "full":
        adjacency = {
            "site_fields": ["id", "kind", "opcode", "address_space", "ptx_line", "block", "predicated"],
            "edge_fields": ["source", "target", "evidence", "register"],
            "sites": [_site(node) for node in nodes],
            "edges": [_edge(edge) for edge in edges],
        }
        if _chars(adjacency) <= MAX_FULL_CHARS:
            result["adjacency"] = adjacency
        else:
            result["status"] = "prompt_size_limit"
            result["full_adjacency_chars"] = _chars(adjacency)
            result["max_adjacency_chars"] = MAX_FULL_CHARS
    return result
