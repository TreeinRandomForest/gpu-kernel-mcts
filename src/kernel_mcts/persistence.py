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
    started_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS search_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES search_runs(run_id),
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS search_events_run_id ON search_events(run_id, id);
"""


class SQLiteTraceStore:
    def __init__(self, path: str | Path) -> None:
        self.connection = sqlite3.connect(path)
        self.connection.executescript(SCHEMA)
        self.connection.commit()
        self.run_id: str | None = None

    def start_run(
        self, run_id: str, benchmark_id: str, algorithm: str, config: Mapping[str, object]
    ) -> None:
        self.connection.execute(
            "INSERT INTO search_runs VALUES (?, ?, ?, ?, ?)",
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

    def __enter__(self) -> "SQLiteTraceStore":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

