"""Generate compact SVG and exact PTX-neighborhood views from saved overlays."""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path

from kernel_mcts.overlay_view import exact_neighborhood, summarize_overlay


OUTPUT = Path(__file__).resolve().parent / "output"


def dot_quote(value: object) -> str:
    return json.dumps(str(value))


def overview_dot(summary: dict) -> str:
    lines = [
        "digraph overlay {",
        'graph [rankdir=LR, bgcolor="white", label="Observed PTX class edges (counts, not paths)", labelloc=t];',
        'node [shape=box, style=rounded];',
    ]
    for operation, count in summary["operation_classes"].items():
        label = f"{operation}\n{count} sites"
        lines.append(f"{dot_quote(operation)} [label={dot_quote(label)}];")
    for pair in summary["observed_class_edge_pairs"]:
        lines.append(
            f'{dot_quote(pair["source_class"])} -> {dot_quote(pair["target_class"])} '
            f'[label={dot_quote(pair["edge_count"])}];'
        )
    lines.append("}")
    return "\n".join(lines) + "\n"


def neighborhood_dot(neighborhood: dict) -> str:
    lines = [
        "digraph neighborhood {",
        'graph [rankdir=LR, bgcolor="white", label="Exact PTX neighborhood", labelloc=t];',
        'node [shape=box, style=rounded];',
    ]
    for node in neighborhood["nodes"]:
        opcode = str(node.get("opcode", node.get("kind", "")))
        if len(opcode) > 52:
            opcode = opcode[:49] + "..."
        label = f'{node["id"]}\n{opcode}'
        if node.get("line") is not None:
            label += f'\nline {node["line"]}'
        style = ', penwidth=2' if node["id"] == neighborhood["anchor_id"] else ""
        lines.append(f'{dot_quote(node["id"])} [label={dot_quote(label)}{style}];')
    edge_counts = Counter((edge["source"], edge["target"])
                          for edge in neighborhood["edges"])
    for (source, target), count in sorted(edge_counts.items()):
        lines.append(f'{dot_quote(source)} -> {dot_quote(target)} '
                     f'[label={dot_quote(count)}];')
    lines.append("}")
    return "\n".join(lines) + "\n"


def render(dot: str, destination: Path) -> None:
    subprocess.run(["dot", "-Tsvg", "-o", str(destination)], input=dot,
                   text=True, check=True)


def main() -> None:
    for path in sorted(OUTPUT.glob("*.overlay.json")):
        overlay = json.loads(path.read_text())
        summary = summarize_overlay(overlay)
        stem = path.name.removesuffix(".overlay.json")
        (OUTPUT / f"{stem}.view.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        render(overview_dot(summary), OUTPUT / f"{stem}.overview.svg")
        anchor = next((node["id"] for node in overlay["nodes"]
                       if node["hardware"]["operation_class"] == "tensor_compute"), None)
        if anchor is not None:
            neighborhood = exact_neighborhood(overlay, anchor, hops=1, max_nodes=40)
            (OUTPUT / f"{stem}.neighborhood.json").write_text(
                json.dumps(neighborhood, indent=2, sort_keys=True) + "\n"
            )
            render(neighborhood_dot(neighborhood), OUTPUT / f"{stem}.neighborhood.svg")
        print(stem, summary["site_count"], summary["proven_edge_count"],
              summary["sites_without_proven_edges"])


if __name__ == "__main__":
    main()
