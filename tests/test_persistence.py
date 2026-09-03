import sqlite3

from kernel_mcts.persistence import SQLiteTraceStore


def test_trace_store_records_run_and_event(tmp_path) -> None:
    path = tmp_path / "trace.sqlite"
    with SQLiteTraceStore(path) as store:
        store.start_run("run", "toy", "mcts", {"budget": 1})
        store.emit("proposal", {"status": "VALID"})
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT count(*) FROM search_events").fetchone()[0] == 1

