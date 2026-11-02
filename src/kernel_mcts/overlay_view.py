"""Loss-aware, operation-independent views of a PTX resource overlay."""

from __future__ import annotations

from collections import Counter, deque
from typing import Mapping


def summarize_overlay(overlay: Mapping[str, object]) -> dict[str, object]:
    """Count sites and *observed* edge pairs; do not infer paths from groups."""
    nodes = overlay["nodes"]
    edges = overlay["edges"]
    by_id = {node["id"]: node for node in nodes}
    if len(by_id) != len(nodes):
        raise ValueError("duplicate overlay node ID")
    classes = Counter(node["hardware"]["operation_class"] for node in nodes)
    links = Counter(link for node in nodes
                    for link in node["hardware"]["candidate_graph_links"])
    pairs: Counter[tuple[str, str]] = Counter()
    incident = set()
    for edge in edges:
        source, target = edge["source"], edge["target"]
        if source not in by_id or target not in by_id:
            raise ValueError("edge refers to missing node")
        incident.update((source, target))
        pairs[(by_id[source]["hardware"]["operation_class"],
               by_id[target]["hardware"]["operation_class"])] += 1
    return {
        "schema_version": 1,
        "source_sha256": overlay.get("source_sha256"),
        "ptx_sha256": overlay.get("ptx_sha256"),
        "site_count": len(nodes),
        "proven_edge_count": len(edges),
        "sites_without_proven_edges": len(nodes) - len(incident),
        "operation_classes": dict(sorted(classes.items())),
        "candidate_link_sites": dict(sorted(links.items())),
        "observed_class_edge_pairs": [
            {"source_class": source, "target_class": target, "edge_count": count}
            for (source, target), count in sorted(pairs.items())
        ],
        "path_gaps": overlay.get("path_gaps", {}),
        "ptx_uncertainty": overlay.get("ptx_uncertainty", []),
        "interpretation": (
            "Class links count original PTX edges, not traversable paths between "
            "all members of a class. Candidate hardware links are static site "
            "annotations, not measured traffic or cache outcomes."
        ),
    }


def exact_neighborhood(
    overlay: Mapping[str, object], anchor_id: str, *, hops: int = 1,
    max_nodes: int = 40,
) -> dict[str, object]:
    """Return a bounded induced subgraph around one exact PTX instruction."""
    if hops < 0 or max_nodes < 1:
        raise ValueError("invalid neighborhood bounds")
    nodes = {node["id"]: node for node in overlay["nodes"]}
    if anchor_id not in nodes:
        raise ValueError("anchor absent from overlay")
    adjacency: dict[str, list[str]] = {key: [] for key in nodes}
    for edge in overlay["edges"]:
        source, target = edge["source"], edge["target"]
        if source not in nodes or target not in nodes:
            raise ValueError("edge refers to missing node")
        adjacency[source].append(target)
        adjacency[target].append(source)
    selected = {anchor_id}
    queue = deque([(anchor_id, 0)])
    truncated = False
    while queue:
        current, depth = queue.popleft()
        if depth >= hops:
            continue
        for neighbor in sorted(set(adjacency[current])):
            if neighbor in selected:
                continue
            if len(selected) >= max_nodes:
                truncated = True
                continue
            selected.add(neighbor)
            queue.append((neighbor, depth + 1))
    return {
        "anchor_id": anchor_id,
        "hops": hops,
        "truncated": truncated,
        "nodes": [nodes[key] for key in sorted(selected)],
        "edges": [edge for edge in overlay["edges"]
                  if edge["source"] in selected and edge["target"] in selected],
        "interpretation": "Only original PTX edges are shown; omitted nodes are not implied paths.",
    }
