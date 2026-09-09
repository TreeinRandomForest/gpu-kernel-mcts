import sqlite3

from kernel_mcts.persistence import SQLiteTraceStore


def test_trace_store_records_run_and_event(tmp_path) -> None:
    path = tmp_path / "trace.sqlite"
    with SQLiteTraceStore(path) as store:
        store.start_run("run", "toy", "mcts", {"budget": 1})
        store.emit("proposal", {"status": "VALID"})
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT count(*) FROM search_events").fetchone()[0] == 1


def test_trace_store_creates_versioned_structured_schema(tmp_path) -> None:
    path = tmp_path / "trace.sqlite"
    with SQLiteTraceStore(path) as store:
        assert store.connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1

    with sqlite3.connect(path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        assert {
            "search_runs",
            "search_events",
            "generations",
            "nodes",
            "strategy_edges",
            "realization_edges",
            "iterations",
            "iteration_steps",
        } <= tables
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1


def test_trace_store_additively_migrates_existing_search_runs(tmp_path) -> None:
    path = tmp_path / "trace.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """CREATE TABLE search_runs (
                run_id TEXT PRIMARY KEY,
                benchmark_id TEXT NOT NULL,
                algorithm TEXT NOT NULL,
                config_json TEXT NOT NULL,
                started_at TEXT NOT NULL
            )"""
        )
        connection.execute(
            "INSERT INTO search_runs VALUES ('old', 'toy', 'mcts', '{}', 'timestamp')"
        )

    with SQLiteTraceStore(path):
        pass

    with sqlite3.connect(path) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(search_runs)").fetchall()
        }
        assert {
            "seed",
            "generation_budget",
            "final_b_gen",
            "final_b_prior",
            "workload_json",
            "hardware_json",
            "toolchain_json",
        } <= columns
        assert connection.execute(
            "SELECT benchmark_id FROM search_runs WHERE run_id = 'old'"
        ).fetchone() == ("toy",)


def test_trace_store_rejects_newer_schema(tmp_path) -> None:
    path = tmp_path / "trace.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA user_version = 999")

    try:
        SQLiteTraceStore(path)
    except RuntimeError as error:
        assert "newer than supported" in str(error)
    else:
        raise AssertionError("expected newer schema to be rejected")
