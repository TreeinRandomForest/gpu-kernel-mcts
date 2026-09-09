from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping


SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS search_runs (
    run_id TEXT PRIMARY KEY,
    benchmark_id TEXT NOT NULL,
    algorithm TEXT NOT NULL,
    config_json TEXT NOT NULL,
    started_at TEXT NOT NULL,
    seed INTEGER,
    generation_budget INTEGER,
    environment_manifest_id TEXT,
    ended_at TEXT,
    best_node_id TEXT,
    final_b_gen INTEGER,
    final_b_prior INTEGER,
    final_iterations INTEGER,
    workload_json TEXT,
    hardware_json TEXT,
    toolchain_json TEXT,
    git_commit TEXT,
    dirty_tree INTEGER,
    model_name TEXT
);
CREATE TABLE IF NOT EXISTS search_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES search_runs(run_id),
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS search_events_run_id ON search_events(run_id, id);
CREATE TABLE IF NOT EXISTS generations (
    generation_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES search_runs(run_id),
    iteration INTEGER,
    b_gen INTEGER NOT NULL,
    parent_node_id TEXT NOT NULL,
    strategy_id TEXT NOT NULL,
    repair_attempt INTEGER NOT NULL,
    proposal_status TEXT NOT NULL,
    strategy_prior REAL,
    parent_visit_count INTEGER,
    parent_action_q_mean REAL,
    parent_action_q_max REAL,
    invalid_reason TEXT,
    compile_status TEXT NOT NULL,
    correctness_status TEXT NOT NULL,
    prompt_hash TEXT,
    prompt_text TEXT,
    raw_output TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    llm_latency_seconds REAL,
    candidate_program TEXT,
    state_key TEXT,
    reward REAL,
    benchmark_json TEXT,
    metadata_json TEXT NOT NULL,
    worker_id TEXT,
    environment_manifest_id TEXT,
    created_node_id TEXT,
    reused_node INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS generations_run_b_gen ON generations(run_id, b_gen);
CREATE TABLE IF NOT EXISTS nodes (
    run_id TEXT NOT NULL REFERENCES search_runs(run_id),
    node_id TEXT NOT NULL,
    state_key TEXT NOT NULL,
    program_text TEXT NOT NULL,
    backend_type TEXT NOT NULL,
    reward REAL NOT NULL,
    workload_json TEXT NOT NULL,
    hardware_json TEXT NOT NULL,
    benchmark_json TEXT,
    profile_json TEXT,
    metadata_json TEXT NOT NULL,
    source_hash TEXT,
    binary_hash TEXT,
    launch_config_json TEXT NOT NULL,
    worker_id TEXT,
    environment_manifest_id TEXT,
    PRIMARY KEY (run_id, node_id),
    UNIQUE (run_id, state_key)
);
CREATE TABLE IF NOT EXISTS strategy_edges (
    run_id TEXT NOT NULL,
    parent_node_id TEXT NOT NULL,
    strategy_id TEXT NOT NULL,
    prior REAL NOT NULL,
    visits INTEGER NOT NULL,
    value_sum REAL NOT NULL,
    q_mean REAL NOT NULL,
    q_max REAL,
    proposal_count INTEGER NOT NULL,
    generation_attempt_count INTEGER NOT NULL,
    repair_generation_count INTEGER NOT NULL,
    valid_proposal_count INTEGER NOT NULL,
    invalid_proposal_count INTEGER NOT NULL,
    PRIMARY KEY (run_id, parent_node_id, strategy_id),
    FOREIGN KEY (run_id, parent_node_id) REFERENCES nodes(run_id, node_id)
);
CREATE TABLE IF NOT EXISTS realization_edges (
    run_id TEXT NOT NULL,
    parent_node_id TEXT NOT NULL,
    strategy_id TEXT NOT NULL,
    child_node_id TEXT NOT NULL,
    descents INTEGER NOT NULL,
    value_sum REAL NOT NULL,
    q_mean REAL NOT NULL,
    PRIMARY KEY (run_id, parent_node_id, strategy_id, child_node_id),
    FOREIGN KEY (run_id, parent_node_id, strategy_id)
        REFERENCES strategy_edges(run_id, parent_node_id, strategy_id),
    FOREIGN KEY (run_id, child_node_id) REFERENCES nodes(run_id, node_id)
);
CREATE TABLE IF NOT EXISTS iterations (
    run_id TEXT NOT NULL REFERENCES search_runs(run_id),
    iteration INTEGER NOT NULL,
    status TEXT NOT NULL,
    expanded_parent_node_id TEXT,
    selected_strategy_id TEXT,
    leaf_node_id TEXT,
    backed_up_reward REAL,
    b_gen INTEGER NOT NULL,
    b_prior INTEGER NOT NULL,
    PRIMARY KEY (run_id, iteration)
);
CREATE TABLE IF NOT EXISTS iteration_steps (
    run_id TEXT NOT NULL,
    iteration INTEGER NOT NULL,
    step INTEGER NOT NULL,
    node_id TEXT NOT NULL,
    strategy_id TEXT NOT NULL,
    selection_mode TEXT NOT NULL,
    child_node_id TEXT,
    PRIMARY KEY (run_id, iteration, step),
    FOREIGN KEY (run_id, iteration) REFERENCES iterations(run_id, iteration)
);
"""

SCHEMA_VERSION = 1

SEARCH_RUN_ADDITIONAL_COLUMNS = {
    "seed": "INTEGER",
    "generation_budget": "INTEGER",
    "environment_manifest_id": "TEXT",
    "ended_at": "TEXT",
    "best_node_id": "TEXT",
    "final_b_gen": "INTEGER",
    "final_b_prior": "INTEGER",
    "final_iterations": "INTEGER",
    "workload_json": "TEXT",
    "hardware_json": "TEXT",
    "toolchain_json": "TEXT",
    "git_commit": "TEXT",
    "dirty_tree": "INTEGER",
    "model_name": "TEXT",
}


class SQLiteTraceStore:
    def __init__(self, path: str | Path) -> None:
        self.connection = sqlite3.connect(path)
        current_version = self.connection.execute("PRAGMA user_version").fetchone()[0]
        if current_version > SCHEMA_VERSION:
            self.connection.close()
            raise RuntimeError(
                f"trace database schema version {current_version} is newer than supported "
                f"version {SCHEMA_VERSION}"
            )
        self.connection.executescript(SCHEMA)
        if current_version < SCHEMA_VERSION:
            self._migrate_search_runs()
            self.connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self.connection.commit()
        self.run_id: str | None = None

    def start_run(
        self, run_id: str, benchmark_id: str, algorithm: str, config: Mapping[str, object]
    ) -> None:
        self.connection.execute(
            """INSERT INTO search_runs(
                run_id, benchmark_id, algorithm, config_json, started_at
            ) VALUES (?, ?, ?, ?, ?)""",
            (run_id, benchmark_id, algorithm, json.dumps(config, sort_keys=True), _now()),
        )
        self.connection.commit()
        self.run_id = run_id

    def emit(self, event_type: str, payload: Mapping[str, object]) -> None:
        if self.run_id is None:
            raise RuntimeError("start_run must be called before emitting events")
        created_at = _now()
        with self.connection:
            self.connection.execute(
                """INSERT INTO search_events(
                    run_id, event_type, payload_json, created_at
                ) VALUES (?, ?, ?, ?)""",
                (self.run_id, event_type, _json(payload), created_at),
            )
            self._materialize_event(event_type, payload, created_at)

    def close(self) -> None:
        self.connection.close()

    def _migrate_search_runs(self) -> None:
        existing = {
            row[1] for row in self.connection.execute("PRAGMA table_info(search_runs)").fetchall()
        }
        for name, sql_type in SEARCH_RUN_ADDITIONAL_COLUMNS.items():
            if name not in existing:
                self.connection.execute(f"ALTER TABLE search_runs ADD COLUMN {name} {sql_type}")

    def _materialize_event(
        self,
        event_type: str,
        payload: Mapping[str, object],
        created_at: str,
    ) -> None:
        handlers = {
            "run_started": self._materialize_run_started,
            "run_completed": self._materialize_run_completed,
            "run_failed": self._materialize_run_failed,
            "node_created": self._materialize_node,
            "strategy_priors": self._materialize_strategy_priors,
            "generation": self._materialize_generation,
            "backup": self._materialize_backup,
            "iteration_completed": self._materialize_iteration,
        }
        handler = handlers.get(event_type)
        if handler is not None:
            handler(payload, created_at)

    def _materialize_run_started(
        self, payload: Mapping[str, object], created_at: str
    ) -> None:
        self.connection.execute(
            """UPDATE search_runs SET
                seed = ?, generation_budget = ?, workload_json = ?, hardware_json = ?
            WHERE run_id = ?""",
            (
                payload.get("seed"),
                payload.get("generation_budget"),
                _json(payload.get("workload", {})),
                _json(payload.get("hardware", {})),
                self.run_id,
            ),
        )

    def _materialize_run_completed(
        self, payload: Mapping[str, object], created_at: str
    ) -> None:
        self.connection.execute(
            """UPDATE search_runs SET
                ended_at = ?, best_node_id = ?, final_b_gen = ?,
                final_b_prior = ?, final_iterations = ?
            WHERE run_id = ?""",
            (
                created_at,
                payload.get("best_node_id"),
                payload.get("b_gen"),
                payload.get("b_prior"),
                payload.get("iterations"),
                self.run_id,
            ),
        )

    def _materialize_run_failed(
        self, payload: Mapping[str, object], created_at: str
    ) -> None:
        self.connection.execute(
            """UPDATE search_runs SET
                ended_at = ?, final_b_gen = ?, final_b_prior = ?, final_iterations = ?
            WHERE run_id = ?""",
            (
                created_at,
                payload.get("b_gen"),
                payload.get("b_prior"),
                payload.get("iterations"),
                self.run_id,
            ),
        )

    def _materialize_node(
        self, payload: Mapping[str, object], created_at: str
    ) -> None:
        evaluation = _mapping(payload.get("evaluation"), "node evaluation")
        program = _mapping(evaluation.get("program"), "node program")
        workload_json, hardware_json = self.connection.execute(
            "SELECT workload_json, hardware_json FROM search_runs WHERE run_id = ?",
            (self.run_id,),
        ).fetchone()
        self.connection.execute(
            """INSERT INTO nodes(
                run_id, node_id, state_key, program_text, backend_type, reward,
                workload_json, hardware_json, benchmark_json, profile_json,
                metadata_json, source_hash, binary_hash, launch_config_json,
                worker_id, environment_manifest_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, node_id) DO UPDATE SET
                profile_json = excluded.profile_json
            """,
            (
                self.run_id,
                payload["node_id"],
                payload["state_key"],
                payload.get("program_text", program["source"]),
                payload.get("backend_type", program["backend"]),
                payload["reward"],
                workload_json or "{}",
                hardware_json or "{}",
                _optional_json(evaluation.get("benchmark")),
                _optional_json(payload.get("profile")),
                _json(evaluation.get("metadata", {})),
                evaluation.get("source_hash"),
                evaluation.get("binary_hash"),
                _json(evaluation.get("launch_config", {})),
                evaluation.get("worker_id"),
                evaluation.get("environment_manifest_id"),
            ),
        )

    def _materialize_strategy_priors(
        self, payload: Mapping[str, object], created_at: str
    ) -> None:
        priors = _mapping(payload.get("priors"), "strategy priors")
        for strategy_id, prior in priors.items():
            self.connection.execute(
                """INSERT INTO strategy_edges(
                    run_id, parent_node_id, strategy_id, prior, visits,
                    value_sum, q_mean, q_max, proposal_count,
                    generation_attempt_count, repair_generation_count,
                    valid_proposal_count, invalid_proposal_count
                ) VALUES (?, ?, ?, ?, 0, 0, 0, NULL, 0, 0, 0, 0, 0)
                ON CONFLICT(run_id, parent_node_id, strategy_id)
                DO UPDATE SET prior = excluded.prior""",
                (self.run_id, payload["node_id"], strategy_id, prior),
            )

    def _materialize_generation(
        self, payload: Mapping[str, object], created_at: str
    ) -> None:
        generation = _mapping(payload.get("generation"), "generation")
        evaluation = _mapping(payload.get("evaluation"), "evaluation")
        self.connection.execute(
            """INSERT INTO generations(
                generation_id, run_id, iteration, b_gen, parent_node_id,
                strategy_id, repair_attempt, proposal_status, strategy_prior,
                parent_visit_count, parent_action_q_mean, parent_action_q_max,
                invalid_reason, compile_status, correctness_status, prompt_hash,
                prompt_text, raw_output, input_tokens, output_tokens,
                llm_latency_seconds, candidate_program, state_key, reward,
                benchmark_json, metadata_json, worker_id,
                environment_manifest_id, created_node_id, reused_node
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                      ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                payload["generation_id"],
                self.run_id,
                payload.get("iteration"),
                payload["b_gen"],
                payload["parent_node_id"],
                payload["strategy_id"],
                payload["repair_attempt"],
                payload["proposal_status"],
                payload.get("strategy_prior"),
                payload.get("parent_visit_count"),
                payload.get("parent_action_q_mean"),
                payload.get("parent_action_q_max"),
                payload.get("invalid_reason"),
                payload["compile_status"],
                payload["correctness_status"],
                payload.get("prompt_hash"),
                payload.get("prompt_text"),
                payload.get("raw_output"),
                payload.get("input_tokens"),
                payload.get("output_tokens"),
                payload.get("llm_latency_seconds"),
                payload.get("candidate_program"),
                payload.get("state_key"),
                payload.get("reward"),
                _optional_json(payload.get("benchmark")),
                _json(
                    {
                        "generation": generation.get("metadata", {}),
                        "evaluation": evaluation.get("metadata", {}),
                    }
                ),
                payload.get("worker_id"),
                payload.get("environment_manifest_id"),
                payload.get("created_node_id"),
                int(bool(payload.get("reused_node", False))),
            ),
        )

    def _materialize_backup(
        self, payload: Mapping[str, object], created_at: str
    ) -> None:
        strategy = _mapping(payload.get("strategy"), "backup strategy")
        realization = _mapping(payload.get("realization"), "backup realization")
        self.connection.execute(
            """INSERT INTO strategy_edges(
                run_id, parent_node_id, strategy_id, prior, visits,
                value_sum, q_mean, q_max, proposal_count,
                generation_attempt_count, repair_generation_count,
                valid_proposal_count, invalid_proposal_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, parent_node_id, strategy_id) DO UPDATE SET
                prior = excluded.prior,
                visits = excluded.visits,
                value_sum = excluded.value_sum,
                q_mean = excluded.q_mean,
                q_max = excluded.q_max,
                proposal_count = excluded.proposal_count,
                generation_attempt_count = excluded.generation_attempt_count,
                repair_generation_count = excluded.repair_generation_count,
                valid_proposal_count = excluded.valid_proposal_count,
                invalid_proposal_count = excluded.invalid_proposal_count""",
            (
                self.run_id,
                payload["parent_node_id"],
                payload["strategy_id"],
                strategy["prior"],
                strategy["visits"],
                strategy["value_sum"],
                strategy["q_mean"],
                strategy.get("q_max"),
                strategy["proposal_count"],
                strategy["generation_attempt_count"],
                strategy["repair_generation_count"],
                strategy["valid_proposal_count"],
                strategy["invalid_proposal_count"],
            ),
        )
        self.connection.execute(
            """INSERT INTO realization_edges(
                run_id, parent_node_id, strategy_id, child_node_id,
                descents, value_sum, q_mean
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, parent_node_id, strategy_id, child_node_id)
            DO UPDATE SET
                descents = excluded.descents,
                value_sum = excluded.value_sum,
                q_mean = excluded.q_mean""",
            (
                self.run_id,
                payload["parent_node_id"],
                payload["strategy_id"],
                payload["child_node_id"],
                realization["descents"],
                realization["value_sum"],
                realization["q_mean"],
            ),
        )

    def _materialize_iteration(
        self, payload: Mapping[str, object], created_at: str
    ) -> None:
        self.connection.execute(
            """INSERT INTO iterations(
                run_id, iteration, status, expanded_parent_node_id,
                selected_strategy_id, leaf_node_id, backed_up_reward,
                b_gen, b_prior
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                self.run_id,
                payload["iteration"],
                payload["status"],
                payload.get("expanded_parent_node_id"),
                payload.get("selected_strategy_id"),
                payload.get("leaf_node_id"),
                payload.get("backed_up_reward"),
                payload["b_gen"],
                payload["b_prior"],
            ),
        )
        for step in payload.get("steps", []):
            step = _mapping(step, "iteration step")
            self.connection.execute(
                """INSERT INTO iteration_steps(
                    run_id, iteration, step, node_id, strategy_id,
                    selection_mode, child_node_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    self.run_id,
                    payload["iteration"],
                    step["step"],
                    step["node_id"],
                    step["strategy_id"],
                    step["selection_mode"],
                    step.get("child_node_id"),
                ),
            )

    def __enter__(self) -> "SQLiteTraceStore":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True)


def _optional_json(value: object) -> str | None:
    return None if value is None else _json(value)


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value
