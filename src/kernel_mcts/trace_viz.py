from __future__ import annotations

import argparse
import json
import math
import shutil
import sqlite3
import subprocess
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


class TraceVisualizationError(RuntimeError):
    """A trace cannot be rendered safely or completely."""


@dataclass(frozen=True, slots=True)
class TraceGraphOptions:
    max_depth: int | None = None
    min_edge_visits: int = 0

    def __post_init__(self) -> None:
        if self.max_depth is not None and self.max_depth < 0:
            raise ValueError("maximum depth cannot be negative")
        if self.min_edge_visits < 0:
            raise ValueError("minimum edge visits cannot be negative")


def render_trace_dot(
    trace_path: str | Path,
    *,
    run_id: str | None = None,
    options: TraceGraphOptions = TraceGraphOptions(),
) -> tuple[str, str]:
    """Render one persisted MCTS run as DOT and return DOT plus selected run ID."""
    path = Path(trace_path).resolve()
    if not path.is_file():
        raise TraceVisualizationError(f"trace database does not exist: {path}")
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        selected_run = _select_run(connection, run_id)
        return _render(connection, selected_run, options), selected_run
    except sqlite3.Error as error:
        raise TraceVisualizationError("failed to read trace database") from error
    finally:
        connection.close()


def write_trace_visualization(
    trace_path: str | Path,
    dot_path: str | Path,
    *,
    run_id: str | None = None,
    png_path: str | Path | None = None,
    options: TraceGraphOptions = TraceGraphOptions(),
    force: bool = False,
) -> str:
    """Write DOT and optionally render it to PNG with Graphviz."""
    dot_output = Path(dot_path)
    png_output = Path(png_path) if png_path is not None else None
    outputs = (dot_output,) if png_output is None else (dot_output, png_output)
    existing = [path for path in outputs if path.exists()]
    if existing and not force:
        raise TraceVisualizationError(
            f"refusing to overwrite existing output: {existing[0]}"
        )
    dot, selected_run = render_trace_dot(
        trace_path,
        run_id=run_id,
        options=options,
    )
    dot_output.write_text(dot, encoding="utf-8")
    if png_output is not None:
        executable = shutil.which("dot")
        if executable is None:
            raise TraceVisualizationError(
                "Graphviz 'dot' is required for PNG output; the DOT file was written"
            )
        result = subprocess.run(
            [executable, "-Tpng", str(dot_output), "-o", str(png_output)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise TraceVisualizationError("Graphviz failed to render PNG output")
    return selected_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a persisted MCTS run as a Graphviz DAG."
    )
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--dot", type=Path, required=True)
    parser.add_argument("--png", type=Path)
    parser.add_argument("--max-depth", type=int)
    parser.add_argument("--min-edge-visits", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        options = TraceGraphOptions(
            max_depth=arguments.max_depth,
            min_edge_visits=arguments.min_edge_visits,
        )
        selected_run = write_trace_visualization(
            arguments.trace,
            arguments.dot,
            run_id=arguments.run_id,
            png_path=arguments.png,
            options=options,
            force=arguments.force,
        )
    except (TraceVisualizationError, ValueError) as error:
        parser.error(str(error))
    print(f"Rendered run {selected_run} to {arguments.dot.resolve()}")
    if arguments.png is not None:
        print(f"PNG: {arguments.png.resolve()}")
    return 0


def _select_run(connection: sqlite3.Connection, requested: str | None) -> str:
    if requested is not None:
        row = connection.execute(
            "SELECT run_id FROM search_runs WHERE run_id = ?", (requested,)
        ).fetchone()
        if row is None:
            raise TraceVisualizationError(f"run ID not found: {requested}")
        return requested
    row = connection.execute(
        "SELECT run_id FROM search_runs ORDER BY started_at DESC, run_id DESC LIMIT 1"
    ).fetchone()
    if row is None:
        raise TraceVisualizationError("trace database contains no search runs")
    return str(row["run_id"])


def _render(
    connection: sqlite3.Connection,
    run_id: str,
    options: TraceGraphOptions,
) -> str:
    run = connection.execute(
        "SELECT * FROM search_runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    if run is None:
        raise TraceVisualizationError(f"run ID not found: {run_id}")
    nodes = connection.execute(
        "SELECT * FROM nodes WHERE run_id = ? ORDER BY node_id", (run_id,)
    ).fetchall()
    strategies = connection.execute(
        """SELECT * FROM strategy_edges WHERE run_id = ?
           ORDER BY parent_node_id, strategy_id""",
        (run_id,),
    ).fetchall()
    realizations = connection.execute(
        """SELECT * FROM realization_edges WHERE run_id = ?
           ORDER BY parent_node_id, strategy_id, child_node_id""",
        (run_id,),
    ).fetchall()
    failures = connection.execute(
        """SELECT generation_id, b_gen, repair_attempt, parent_node_id,
                  strategy_id, proposal_status, invalid_reason, compile_status,
                  correctness_status
           FROM generations
           WHERE run_id = ? AND proposal_status != 'VALID'
           ORDER BY b_gen, generation_id""",
        (run_id,),
    ).fetchall()
    root_id = _root_node_id(connection, run_id, nodes, realizations)
    depths = _shortest_depths(root_id, realizations)
    visible_nodes = {
        str(node["node_id"])
        for node in nodes
        if options.max_depth is None
        or depths.get(str(node["node_id"]), math.inf) <= options.max_depth
    }
    visible_realizations = [
        edge
        for edge in realizations
        if str(edge["parent_node_id"]) in visible_nodes
        and str(edge["child_node_id"]) in visible_nodes
        and int(edge["descents"]) >= options.min_edge_visits
    ]
    visible_strategy_keys = {
        (str(edge["parent_node_id"]), str(edge["strategy_id"]))
        for edge in visible_realizations
    }
    visible_strategy_keys.update(
        (str(item["parent_node_id"]), str(item["strategy_id"]))
        for item in failures
        if str(item["parent_node_id"]) in visible_nodes
        and options.min_edge_visits == 0
    )
    strategy_by_key = {
        (str(item["parent_node_id"]), str(item["strategy_id"])): item
        for item in strategies
    }
    action_visits: dict[str, int] = {}
    for strategy in strategies:
        parent = str(strategy["parent_node_id"])
        action_visits[parent] = action_visits.get(parent, 0) + int(strategy["visits"])

    terminal_event = connection.execute(
        """SELECT event_type FROM search_events
           WHERE run_id = ? AND event_type IN ('run_completed', 'run_failed')
           ORDER BY id DESC LIMIT 1""",
        (run_id,),
    ).fetchone()
    status = (
        "RUNNING/INTERRUPTED"
        if terminal_event is None
        else str(terminal_event["event_type"]).removeprefix("run_").upper()
    )
    observed_b_gen = connection.execute(
        "SELECT count(*) FROM generations WHERE run_id = ?", (run_id,)
    ).fetchone()[0]
    observed_iterations = connection.execute(
        "SELECT count(*) FROM iterations WHERE run_id = ?", (run_id,)
    ).fetchone()[0]
    lines = [
        "digraph mcts_search {",
        "  graph [rankdir=LR, labelloc=t, fontsize=12, fontname=Helvetica];",
        "  node [fontname=Helvetica, fontsize=10];",
        "  edge [fontname=Helvetica, fontsize=9];",
        f'  label="{_escape(_run_label(run, status, len(nodes), observed_b_gen, observed_iterations))}";',
    ]
    best_node_id = str(run["best_node_id"]) if run["best_node_id"] else None
    if best_node_id is None and nodes:
        best_node_id = str(max(nodes, key=lambda item: float(item["reward"]))["node_id"])
    for node in nodes:
        node_id = str(node["node_id"])
        if node_id not in visible_nodes:
            continue
        attributes = [f'label="{_escape(_node_label(node, action_visits.get(node_id, 0)))}"']
        attributes.extend(("shape=box", "style=filled"))
        if node_id == root_id and node_id == best_node_id:
            attributes.extend(("fillcolor=gold", "penwidth=3"))
        elif node_id == root_id:
            attributes.extend(("fillcolor=lightblue", "penwidth=2"))
        elif node_id == best_node_id:
            attributes.extend(("fillcolor=gold", "penwidth=3"))
        else:
            attributes.append("fillcolor=white")
        lines.append(f'  "{_escape_id(node_id)}" [{", ".join(attributes)}];')
    for parent_id, strategy_id in sorted(visible_strategy_keys):
        strategy = strategy_by_key.get((parent_id, strategy_id))
        strategy_dot_id = _strategy_dot_id(parent_id, strategy_id)
        lines.append(
            f'  "{strategy_dot_id}" [shape=diamond, style=filled, fillcolor=lavender, '
            f'label="{_escape(_strategy_label(strategy_id, strategy))}"];'
        )
        lines.append(
            f'  "{_escape_id(parent_id)}" -> "{strategy_dot_id}" '
            '[color=royalblue, label="PUCT"];'
        )
    for edge in visible_realizations:
        strategy_dot_id = _strategy_dot_id(
            str(edge["parent_node_id"]), str(edge["strategy_id"])
        )
        lines.append(
            f'  "{strategy_dot_id}" -> "{_escape_id(str(edge["child_node_id"]))}" '
            f'[color=darkgreen, label="{_escape(_realization_label(edge))}"];'
        )
    for failure in failures:
        if options.min_edge_visits > 0:
            continue
        key = (str(failure["parent_node_id"]), str(failure["strategy_id"]))
        if key not in visible_strategy_keys:
            continue
        failure_id = f"failure:{failure['b_gen']}:{failure['generation_id']}"
        lines.append(
            f'  "{_escape_id(failure_id)}" [shape=note, style=filled, fillcolor=mistyrose, '
            f'label="{_escape(_failure_label(failure))}"];'
        )
        lines.append(
            f'  "{_strategy_dot_id(*key)}" -> "{_escape_id(failure_id)}" '
            '[style=dashed, color=firebrick, label="proposal"];'
        )
    lines.append("}")
    return "\n".join(lines) + "\n"


def _root_node_id(
    connection: sqlite3.Connection,
    run_id: str,
    nodes: Sequence[sqlite3.Row],
    realizations: Sequence[sqlite3.Row],
) -> str | None:
    event = connection.execute(
        """SELECT payload_json FROM search_events
           WHERE run_id = ? AND event_type = 'run_started'
           ORDER BY id LIMIT 1""",
        (run_id,),
    ).fetchone()
    if event is not None:
        payload = _json_mapping(event["payload_json"])
        if isinstance(payload.get("root_node_id"), str):
            return str(payload["root_node_id"])
    children = {str(edge["child_node_id"]) for edge in realizations}
    roots = sorted(str(node["node_id"]) for node in nodes if str(node["node_id"]) not in children)
    return roots[0] if roots else (str(nodes[0]["node_id"]) if nodes else None)


def _shortest_depths(
    root_id: str | None,
    realizations: Sequence[sqlite3.Row],
) -> dict[str, int]:
    if root_id is None:
        return {}
    children: dict[str, list[str]] = {}
    for edge in realizations:
        children.setdefault(str(edge["parent_node_id"]), []).append(
            str(edge["child_node_id"])
        )
    depths = {root_id: 0}
    queue = deque([root_id])
    while queue:
        parent = queue.popleft()
        for child in children.get(parent, []):
            candidate = depths[parent] + 1
            if child not in depths or candidate < depths[child]:
                depths[child] = candidate
                queue.append(child)
    return depths


def _run_label(
    run: sqlite3.Row,
    status: str,
    node_count: int,
    observed_b_gen: int,
    observed_iterations: int,
) -> str:
    b_gen = run["final_b_gen"] if run["final_b_gen"] is not None else observed_b_gen
    iterations = (
        run["final_iterations"]
        if run["final_iterations"] is not None
        else observed_iterations
    )
    return (
        f"MCTS run {run['run_id']} | {status} | benchmark={run['benchmark_id']}\n"
        f"model={run['model_name'] or '-'} | B_gen={b_gen}/{run['generation_budget'] or '?'} "
        f"| iterations={iterations} | nodes={node_count}"
    )


def _node_label(node: sqlite3.Row, visits: int) -> str:
    reward = float(node["reward"])
    benchmark = _json_mapping(node["benchmark_json"])
    median = benchmark.get("median_us")
    latency = f"{float(median):.3f} us" if isinstance(median, (int, float)) else "-"
    return (
        f"node {_short_id(str(node['node_id']))}\n"
        f"reward={reward:.4f} | speedup={math.exp(reward):.3f}x\n"
        f"median={latency} | action visits={visits}"
    )


def _strategy_label(strategy_id: str, strategy: sqlite3.Row | None) -> str:
    if strategy is None:
        return strategy_id
    q_max = "-" if strategy["q_max"] is None else f"{float(strategy['q_max']):.4f}"
    return (
        f"{strategy_id}\nprior={float(strategy['prior']):.3f} | "
        f"visits={strategy['visits']}\nQ_mean={float(strategy['q_mean']):.4f} | "
        f"Q_max={q_max}\nproposals={strategy['proposal_count']} "
        f"(valid={strategy['valid_proposal_count']}, invalid={strategy['invalid_proposal_count']})"
    )


def _realization_label(edge: sqlite3.Row) -> str:
    return f"UCB | descents={edge['descents']} | Q={float(edge['q_mean']):.4f}"


def _failure_label(generation: sqlite3.Row) -> str:
    reason = generation["invalid_reason"] or "-"
    return (
        f"B_gen={generation['b_gen']} | {generation['proposal_status']}\n"
        f"repair={generation['repair_attempt']} | reason={reason}\n"
        f"compile={generation['compile_status']} | "
        f"correctness={generation['correctness_status']}"
    )


def _strategy_dot_id(parent_id: str, strategy_id: str) -> str:
    return _escape_id(f"strategy:{parent_id}:{strategy_id}")


def _short_id(value: str) -> str:
    return value if len(value) <= 12 else value[:8]


def _json_mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, str):
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, Mapping) else {}


def _escape(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _escape_id(value: str) -> str:
    return _escape(value)


if __name__ == "__main__":
    raise SystemExit(main())
