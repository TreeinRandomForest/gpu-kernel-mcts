from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import sqlite3
import webbrowser
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import parse_qs, quote, unquote, urlparse


class TraceBrowserError(RuntimeError):
    """A trace-browser request cannot be satisfied safely."""


REQUIRED_TABLES = {
    "search_runs",
    "search_events",
    "nodes",
    "strategy_edges",
    "realization_edges",
    "generations",
    "iterations",
    "iteration_steps",
}


def _json_value(value: object, fallback: Any = None) -> Any:
    if not isinstance(value, str):
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return fallback


def _connection(path: Path) -> sqlite3.Connection:
    uri = f"file:{quote(str(path.resolve()))}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    return connection


class TraceCatalog:
    """Discovers and queries trace databases beneath one allowed directory."""

    def __init__(self, trace_dir: str | Path) -> None:
        self.trace_dir = Path(trace_dir).resolve()
        if not self.trace_dir.is_dir():
            raise TraceBrowserError(f"trace directory does not exist: {self.trace_dir}")

    def list_traces(self) -> list[dict[str, object]]:
        entries: list[dict[str, object]] = []
        candidates = sorted(
            {*(self.trace_dir.rglob("*.sqlite")), *(self.trace_dir.rglob("*.sqlite3"))}
        )
        for path in candidates:
            relative = path.relative_to(self.trace_dir).as_posix()
            trace_id = self._trace_id(relative)
            try:
                with _connection(path) as connection:
                    self._validate(connection)
                    runs = self._run_summaries(connection)
                entries.append(
                    {
                        "trace_id": trace_id,
                        "name": relative,
                        "size_bytes": path.stat().st_size,
                        "runs": runs,
                    }
                )
            except (sqlite3.Error, TraceBrowserError) as error:
                entries.append(
                    {
                        "trace_id": trace_id,
                        "name": relative,
                        "size_bytes": path.stat().st_size,
                        "runs": [],
                        "error": str(error),
                    }
                )
        return entries

    def graph(self, trace_id: str, run_id: str) -> dict[str, object]:
        path = self._resolve(trace_id)
        with _connection(path) as connection:
            self._validate(connection)
            run = self._run(connection, run_id)
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
                """SELECT generation_id, iteration, b_gen, repair_attempt, parent_node_id,
                          strategy_id, proposal_status, invalid_reason, compile_status,
                          correctness_status
                   FROM generations
                   WHERE run_id = ? AND proposal_status != 'VALID'
                   ORDER BY b_gen, generation_id""",
                (run_id,),
            ).fetchall()
            root_id = self._root_id(connection, run_id, nodes, realizations)
            depths = self._depths(root_id, realizations)
            creation_iterations = {
                str(row["created_node_id"]): int(row["created_iteration"])
                for row in connection.execute(
                    """SELECT created_node_id, min(iteration) AS created_iteration
                       FROM generations
                       WHERE run_id = ? AND created_node_id IS NOT NULL
                             AND iteration IS NOT NULL
                       GROUP BY created_node_id""",
                    (run_id,),
                ).fetchall()
            }
            realization_iterations = {
                (str(row["parent_node_id"]), str(row["strategy_id"]), str(row["created_node_id"])):
                int(row["created_iteration"])
                for row in connection.execute(
                    """SELECT parent_node_id, strategy_id, created_node_id,
                              min(iteration) AS created_iteration
                       FROM generations
                       WHERE run_id = ? AND created_node_id IS NOT NULL
                             AND iteration IS NOT NULL
                       GROUP BY parent_node_id, strategy_id, created_node_id""",
                    (run_id,),
                ).fetchall()
            }
            action_visits: dict[str, int] = {}
            for strategy in strategies:
                parent = str(strategy["parent_node_id"])
                action_visits[parent] = action_visits.get(parent, 0) + int(
                    strategy["visits"]
                )
            best_node_id = run["best_node_id"]
            if best_node_id is None and nodes:
                best_node_id = max(nodes, key=lambda row: float(row["reward"]))[
                    "node_id"
                ]
            decisions = self._selection_decisions(connection, run_id)
            return {
                "run": self._run_summary(connection, run),
                "root_node_id": root_id,
                "best_node_id": best_node_id,
                "historical_scores_available": bool(decisions),
                "selection_decisions": decisions,
                "nodes": [
                    {
                        "node_id": row["node_id"],
                        "reward": row["reward"],
                        "speedup": math.exp(float(row["reward"])),
                        "median_us": (_json_value(row["benchmark_json"], {}) or {}).get(
                            "median_us"
                        ),
                        "profiled": row["profile_json"] is not None,
                        "depth": depths.get(str(row["node_id"])),
                        "created_iteration": (
                            0
                            if str(row["node_id"]) == root_id
                            else creation_iterations.get(str(row["node_id"]))
                        ),
                        "action_visits": action_visits.get(str(row["node_id"]), 0),
                    }
                    for row in nodes
                ],
                "strategies": [self._strategy(row) for row in strategies],
                "realizations": [
                    {
                        **dict(row),
                        "created_iteration": realization_iterations.get(
                            (
                                str(row["parent_node_id"]),
                                str(row["strategy_id"]),
                                str(row["child_node_id"]),
                            )
                        ),
                    }
                    for row in realizations
                ],
                "failures": [dict(row) for row in failures],
                "analysis": self._analysis(connection, run_id, root_id, nodes),
            }

    def node(self, trace_id: str, run_id: str, node_id: str) -> dict[str, object]:
        path = self._resolve(trace_id)
        with _connection(path) as connection:
            self._validate(connection)
            self._run(connection, run_id)
            row = connection.execute(
                "SELECT * FROM nodes WHERE run_id = ? AND node_id = ?",
                (run_id, node_id),
            ).fetchone()
            if row is None:
                raise TraceBrowserError(f"node not found: {node_id}")
            generations = connection.execute(
                """SELECT generation_id, iteration, b_gen, parent_node_id, strategy_id,
                          repair_attempt, reused_node, prompt_text, api_instructions,
                          raw_output, input_tokens, output_tokens, llm_latency_seconds
                   FROM generations
                   WHERE run_id = ? AND created_node_id = ?
                   ORDER BY b_gen, generation_id""",
                (run_id, node_id),
            ).fetchall()
            return {
                "node_id": row["node_id"],
                "state_key": row["state_key"],
                "program_text": row["program_text"],
                "backend_type": row["backend_type"],
                "reward": row["reward"],
                "speedup": math.exp(float(row["reward"])),
                "benchmark": _json_value(row["benchmark_json"], None),
                "profile": _json_value(row["profile_json"], None),
                "workload": _json_value(row["workload_json"], {}),
                "hardware": _json_value(row["hardware_json"], {}),
                "metadata": _json_value(row["metadata_json"], {}),
                "launch_config": _json_value(row["launch_config_json"], {}),
                "source_hash": row["source_hash"],
                "binary_hash": row["binary_hash"],
                "worker_id": row["worker_id"],
                "environment_manifest_id": row["environment_manifest_id"],
                "generations": [dict(item) for item in generations],
            }

    def compare(
        self, trace_id: str, run_id: str, node_a: str, node_b: str
    ) -> dict[str, object]:
        first = self.node(trace_id, run_id, node_a)
        second = self.node(trace_id, run_id, node_b)
        path = self._resolve(trace_id)
        with _connection(path) as connection:
            edges = [
                dict(row)
                for row in connection.execute(
                    """SELECT parent_node_id, strategy_id, child_node_id, descents,
                              q_mean
                       FROM realization_edges WHERE run_id = ?""",
                    (run_id,),
                ).fetchall()
            ]
        diff = "".join(
            difflib.unified_diff(
                str(first["program_text"]).splitlines(keepends=True),
                str(second["program_text"]).splitlines(keepends=True),
                fromfile=f"node-{node_a}",
                tofile=f"node-{node_b}",
            )
        )
        return {
            "node_a": node_a,
            "node_b": node_b,
            "relationship": self._relationship(node_a, node_b, edges),
            "source_diff": diff,
            "profile_comparison": self._profile_comparison(
                first.get("profile"), second.get("profile")
            ),
        }

    def _resolve(self, trace_id: str) -> Path:
        for entry in self.list_traces():
            if entry["trace_id"] == trace_id:
                path = (self.trace_dir / str(entry["name"])).resolve()
                if not path.is_relative_to(self.trace_dir):
                    break
                return path
        raise TraceBrowserError("trace file not found")

    @staticmethod
    def _trace_id(relative: str) -> str:
        return hashlib.sha256(relative.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _validate(connection: sqlite3.Connection) -> None:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        missing = sorted(REQUIRED_TABLES - tables)
        if missing:
            raise TraceBrowserError(
                "not a supported trace database; missing tables: " + ", ".join(missing)
            )

    @staticmethod
    def _has_table(connection: sqlite3.Connection, name: str) -> bool:
        return (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
            ).fetchone()
            is not None
        )

    def _selection_decisions(
        self, connection: sqlite3.Connection, run_id: str
    ) -> list[dict[str, object]]:
        required = ("selection_decisions", "puct_candidates", "ucb_candidates")
        if not all(self._has_table(connection, name) for name in required):
            return []
        decisions: list[dict[str, object]] = []
        for row in connection.execute(
            """SELECT * FROM selection_decisions WHERE run_id = ?
               ORDER BY iteration, step""",
            (run_id,),
        ).fetchall():
            decision = dict(row)
            decision["puct_candidates"] = [
                dict(candidate)
                for candidate in connection.execute(
                    """SELECT strategy_id, prior, visits, q_mean, q_max,
                              exploit_term, explore_term, total_score, selected
                       FROM puct_candidates
                       WHERE run_id = ? AND iteration = ? AND step = ?
                       ORDER BY total_score DESC, strategy_id""",
                    (run_id, row["iteration"], row["step"]),
                ).fetchall()
            ]
            decision["ucb_candidates"] = [
                dict(candidate)
                for candidate in connection.execute(
                    """SELECT child_node_id, descents, q_mean, exploit_term,
                              explore_term, total_score, selected
                       FROM ucb_candidates
                       WHERE run_id = ? AND iteration = ? AND step = ?
                       ORDER BY total_score DESC, child_node_id""",
                    (run_id, row["iteration"], row["step"]),
                ).fetchall()
            ]
            decisions.append(decision)
        return decisions

    def _analysis(
        self,
        connection: sqlite3.Connection,
        run_id: str,
        root_id: str | None,
        nodes: Sequence[sqlite3.Row],
    ) -> dict[str, object]:
        node_by_id = {str(row["node_id"]): row for row in nodes}
        root_reward = (
            float(node_by_id[root_id]["reward"])
            if root_id is not None and root_id in node_by_id
            else 0.0
        )
        best_reward = root_reward
        best_node_id = root_id
        timeline: list[dict[str, object]] = []
        for row in connection.execute(
            """SELECT iteration, status, selected_strategy_id, leaf_node_id,
                      backed_up_reward, b_gen, b_prior
               FROM iterations WHERE run_id = ? ORDER BY iteration""",
            (run_id,),
        ).fetchall():
            leaf_id = str(row["leaf_node_id"]) if row["leaf_node_id"] is not None else None
            reward = row["backed_up_reward"]
            if reward is not None and float(reward) > best_reward:
                best_reward = float(reward)
                best_node_id = leaf_id
            best_node = node_by_id.get(best_node_id) if best_node_id is not None else None
            benchmark = _json_value(best_node["benchmark_json"], {}) if best_node else {}
            timeline.append(
                {
                    **dict(row),
                    "cumulative_best_reward": best_reward,
                    "cumulative_best_speedup": math.exp(best_reward),
                    "cumulative_best_node_id": best_node_id,
                    "cumulative_best_median_us": (
                        benchmark.get("median_us") if isinstance(benchmark, Mapping) else None
                    ),
                }
            )

        edge_rows = connection.execute(
            """SELECT strategy_id, sum(visits) AS visits,
                      sum(proposal_count) AS proposals,
                      sum(generation_attempt_count) AS generation_attempts,
                      sum(repair_generation_count) AS repair_generations,
                      sum(valid_proposal_count) AS valid_proposals,
                      sum(invalid_proposal_count) AS invalid_proposals,
                      count(*) AS parent_edges
               FROM strategy_edges WHERE run_id = ? GROUP BY strategy_id""",
            (run_id,),
        ).fetchall()
        generation_rows = {
            str(row["strategy_id"]): row
            for row in connection.execute(
                """SELECT strategy_id, count(*) AS calls,
                          sum(CASE WHEN repair_attempt > 0 THEN 1 ELSE 0 END) AS repair_calls,
                          sum(CASE WHEN proposal_status = 'VALID' THEN 1 ELSE 0 END) AS valid_calls,
                          sum(CASE WHEN proposal_status = 'INVALID' THEN 1 ELSE 0 END) AS invalid_calls,
                          sum(CASE WHEN proposal_status = 'INFRASTRUCTURE_FAILURE' THEN 1 ELSE 0 END) AS infrastructure_calls,
                          avg(CASE WHEN proposal_status = 'VALID' THEN reward END) AS mean_valid_reward,
                          max(CASE WHEN proposal_status = 'VALID' THEN reward END) AS max_valid_reward
                   FROM generations WHERE run_id = ? GROUP BY strategy_id""",
                (run_id,),
            ).fetchall()
        }
        iteration_rows = {
            str(row["selected_strategy_id"]): row
            for row in connection.execute(
                """SELECT selected_strategy_id, count(*) AS completed_proposals,
                          sum(CASE WHEN status = 'VALID' THEN 1 ELSE 0 END) AS valid_outcomes,
                          sum(CASE WHEN status = 'INVALID' THEN 1 ELSE 0 END) AS invalid_outcomes,
                          sum(CASE WHEN status = 'INFRASTRUCTURE_FAILURE' THEN 1 ELSE 0 END) AS infrastructure_outcomes
                   FROM iterations
                   WHERE run_id = ? AND selected_strategy_id IS NOT NULL
                   GROUP BY selected_strategy_id""",
                (run_id,),
            ).fetchall()
        }
        strategy_ids = sorted(
            {str(row["strategy_id"]) for row in edge_rows}
            | set(generation_rows)
            | set(iteration_rows)
        )
        edges = {str(row["strategy_id"]): row for row in edge_rows}
        analytics = []
        for strategy_id in strategy_ids:
            edge = edges.get(strategy_id)
            calls = generation_rows.get(strategy_id)
            outcomes = iteration_rows.get(strategy_id)
            analytics.append(
                {
                    "strategy_id": strategy_id,
                    **self._row_with_defaults(
                        edge,
                        ("visits", "proposals", "generation_attempts", "repair_generations", "valid_proposals", "invalid_proposals", "parent_edges"),
                    ),
                    **self._row_with_defaults(
                        calls,
                        ("calls", "repair_calls", "valid_calls", "invalid_calls", "infrastructure_calls", "mean_valid_reward", "max_valid_reward"),
                    ),
                    **self._row_with_defaults(
                        outcomes,
                        ("completed_proposals", "valid_outcomes", "invalid_outcomes", "infrastructure_outcomes"),
                    ),
                }
            )
        return {"timeline": timeline, "strategies": analytics}

    @staticmethod
    def _row_with_defaults(
        row: sqlite3.Row | None, fields: Sequence[str]
    ) -> dict[str, object]:
        return {field: (row[field] if row is not None else None) for field in fields}

    def _run_summaries(self, connection: sqlite3.Connection) -> list[dict[str, object]]:
        return [
            self._run_summary(connection, row)
            for row in connection.execute(
                "SELECT * FROM search_runs ORDER BY started_at DESC, run_id DESC"
            ).fetchall()
        ]

    def _run_summary(
        self, connection: sqlite3.Connection, run: sqlite3.Row
    ) -> dict[str, object]:
        run_id = str(run["run_id"])
        node_stats = connection.execute(
            "SELECT count(*), max(reward) FROM nodes WHERE run_id = ?", (run_id,)
        ).fetchone()
        observed_b_gen = connection.execute(
            "SELECT count(*) FROM generations WHERE run_id = ?", (run_id,)
        ).fetchone()[0]
        observed_iterations = connection.execute(
            "SELECT count(*) FROM iterations WHERE run_id = ?", (run_id,)
        ).fetchone()[0]
        terminal = connection.execute(
            """SELECT event_type FROM search_events
               WHERE run_id = ? AND event_type IN ('run_completed', 'run_failed')
               ORDER BY id DESC LIMIT 1""",
            (run_id,),
        ).fetchone()
        status = (
            "RUNNING/INTERRUPTED"
            if terminal is None
            else str(terminal["event_type"]).removeprefix("run_").upper()
        )
        best_reward = node_stats[1]
        return {
            "run_id": run_id,
            "benchmark_id": run["benchmark_id"],
            "algorithm": run["algorithm"],
            "started_at": run["started_at"],
            "ended_at": run["ended_at"],
            "status": status,
            "model_name": run["model_name"],
            "seed": run["seed"],
            "generation_budget": run["generation_budget"],
            "b_gen": run["final_b_gen"] if run["final_b_gen"] is not None else observed_b_gen,
            "b_prior": run["final_b_prior"],
            "iterations": (
                run["final_iterations"]
                if run["final_iterations"] is not None
                else observed_iterations
            ),
            "node_count": node_stats[0],
            "best_node_id": run["best_node_id"],
            "best_reward": best_reward,
            "best_speedup": math.exp(float(best_reward)) if best_reward is not None else None,
            "config": _json_value(run["config_json"], {}),
            "hardware": _json_value(run["hardware_json"], {}),
        }

    @staticmethod
    def _run(connection: sqlite3.Connection, run_id: str) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM search_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise TraceBrowserError(f"run ID not found: {run_id}")
        return row

    @staticmethod
    def _strategy(row: sqlite3.Row) -> dict[str, object]:
        result = dict(row)
        result["exploit"] = float(row["q_mean"])
        result["final_statistics_only"] = True
        return result

    @staticmethod
    def _root_id(
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
        payload = _json_value(event["payload_json"], {}) if event else {}
        if isinstance(payload, Mapping) and isinstance(payload.get("root_node_id"), str):
            return str(payload["root_node_id"])
        children = {str(edge["child_node_id"]) for edge in realizations}
        roots = sorted(
            str(node["node_id"])
            for node in nodes
            if str(node["node_id"]) not in children
        )
        return roots[0] if roots else (str(nodes[0]["node_id"]) if nodes else None)

    @staticmethod
    def _depths(
        root_id: str | None, realizations: Sequence[sqlite3.Row]
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

    @staticmethod
    def _shortest_path(
        start: str, target: str, edges: Sequence[Mapping[str, object]]
    ) -> list[dict[str, object]] | None:
        outgoing: dict[str, list[Mapping[str, object]]] = {}
        for edge in edges:
            outgoing.setdefault(str(edge["parent_node_id"]), []).append(edge)
        queue = deque([(start, [])])
        seen = {start}
        while queue:
            node, path = queue.popleft()
            if node == target:
                return path
            for edge in outgoing.get(node, []):
                child = str(edge["child_node_id"])
                if child not in seen:
                    seen.add(child)
                    queue.append((child, [*path, dict(edge)]))
        return None

    def _relationship(
        self, node_a: str, node_b: str, edges: Sequence[Mapping[str, object]]
    ) -> dict[str, object]:
        direct = self._shortest_path(node_a, node_b, edges)
        if direct is not None:
            return {"kind": "A_ANCESTOR_OF_B", "path": direct}
        reverse = self._shortest_path(node_b, node_a, edges)
        if reverse is not None:
            return {"kind": "B_ANCESTOR_OF_A", "path": reverse}

        all_nodes = {node_a, node_b}
        for edge in edges:
            all_nodes.add(str(edge["parent_node_id"]))
            all_nodes.add(str(edge["child_node_id"]))
        candidates: list[tuple[int, str, list[dict[str, object]], list[dict[str, object]]]] = []
        for candidate in all_nodes:
            to_a = self._shortest_path(candidate, node_a, edges)
            to_b = self._shortest_path(candidate, node_b, edges)
            if to_a is not None and to_b is not None:
                candidates.append((len(to_a) + len(to_b), candidate, to_a, to_b))
        if not candidates:
            return {"kind": "UNRELATED"}
        _, ancestor, to_a, to_b = min(candidates, key=lambda item: (item[0], item[1]))
        return {
            "kind": "COMMON_ANCESTOR",
            "common_ancestor": ancestor,
            "path_to_a": to_a,
            "path_to_b": to_b,
        }

    @staticmethod
    def _profile_comparison(first: object, second: object) -> list[dict[str, object]]:
        def flatten(value: object, prefix: str = "") -> dict[str, float]:
            result: dict[str, float] = {}
            if isinstance(value, Mapping):
                for key, child in value.items():
                    name = f"{prefix}.{key}" if prefix else str(key)
                    result.update(flatten(child, name))
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                result[prefix] = float(value)
            return result

        left, right = flatten(first), flatten(second)
        return [
            {
                "metric": metric,
                "a": left.get(metric),
                "b": right.get(metric),
                "delta": (
                    right[metric] - left[metric]
                    if metric in left and metric in right
                    else None
                ),
                "percent_change": (
                    100.0 * (right[metric] - left[metric]) / abs(left[metric])
                    if metric in left and metric in right and left[metric] != 0
                    else None
                ),
            }
            for metric in sorted(left.keys() | right.keys())
        ]


class TraceBrowserHandler(BaseHTTPRequestHandler):
    catalog: TraceCatalog

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/traces":
                self._json({"traces": self.catalog.list_traces()})
                return
            query = {key: values[-1] for key, values in parse_qs(parsed.query).items()}
            if parsed.path == "/api/graph":
                self._json(self.catalog.graph(query["trace"], query["run"]))
                return
            if parsed.path == "/api/node":
                self._json(
                    self.catalog.node(query["trace"], query["run"], query["node"])
                )
                return
            if parsed.path == "/api/compare":
                self._json(
                    self.catalog.compare(
                        query["trace"], query["run"], query["a"], query["b"]
                    )
                )
                return
            self._static(parsed.path)
        except KeyError as error:
            self._error(HTTPStatus.BAD_REQUEST, f"missing query parameter: {error.args[0]}")
        except TraceBrowserError as error:
            self._error(HTTPStatus.NOT_FOUND, str(error))
        except sqlite3.Error:
            self._error(HTTPStatus.BAD_REQUEST, "failed to read trace database")

    def log_message(self, format: str, *args: object) -> None:
        return

    def _json(self, value: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        payload = json.dumps(value, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._json({"error": message}, status)

    def _static(self, request_path: str) -> None:
        names = {
            "/": ("index.html", "text/html; charset=utf-8"),
            "/index.html": ("index.html", "text/html; charset=utf-8"),
            "/app.js": ("app.js", "text/javascript; charset=utf-8"),
            "/style.css": ("style.css", "text/css; charset=utf-8"),
        }
        selected = names.get(unquote(request_path))
        if selected is None:
            self._error(HTTPStatus.NOT_FOUND, "not found")
            return
        name, content_type = selected
        payload = files("kernel_mcts.trace_browser_static").joinpath(name).read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Browse local MCTS SQLite traces.")
    parser.add_argument("--trace-dir", type=Path, default=Path.cwd())
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", dest="open_browser")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    catalog = TraceCatalog(arguments.trace_dir)
    handler = type("ConfiguredTraceBrowserHandler", (TraceBrowserHandler,), {"catalog": catalog})
    server = ThreadingHTTPServer((arguments.host, arguments.port), handler)
    url = f"http://{arguments.host}:{server.server_port}/"
    print(f"Trace browser: {url}")
    print(f"Trace directory: {catalog.trace_dir}")
    if arguments.open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nTrace browser stopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
