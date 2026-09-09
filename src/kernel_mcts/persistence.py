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
        self.connection.execute(
            "INSERT INTO search_events(run_id, event_type, payload_json, created_at) VALUES (?, ?, ?, ?)",
            (self.run_id, event_type, json.dumps(payload, sort_keys=True), _now()),
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def _migrate_search_runs(self) -> None:
        existing = {
            row[1] for row in self.connection.execute("PRAGMA table_info(search_runs)").fetchall()
        }
        for name, sql_type in SEARCH_RUN_ADDITIONAL_COLUMNS.items():
            if name not in existing:
                self.connection.execute(f"ALTER TABLE search_runs ADD COLUMN {name} {sql_type}")

    def __enter__(self) -> "SQLiteTraceStore":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
