"""Extract exact seed-0 B/D winners and hash-linked PTX from read-only traces."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path


HERE = Path(__file__).resolve().parent
FACTORIAL = HERE.parent / "triton-h100-context-factorial" / "output"
ARMS = ("B", "D")


def write_exact(path: Path, content: bytes) -> None:
    if path.exists() and path.read_bytes() != content:
        raise FileExistsError(f"refusing to replace different artifact: {path}")
    path.write_bytes(content)


def extract_arm(arm: str, trace: Path) -> dict:
    if arm not in ARMS:
        raise ValueError("unexpected arm")
    with sqlite3.connect(f"file:{trace}?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        runs = db.execute("SELECT run_id, best_node_id, ended_at FROM search_runs").fetchall()
        if len(runs) != 1 or runs[0]["ended_at"] is None or runs[0]["best_node_id"] is None:
            raise ValueError(f"trace not closed with a best node: {trace}")
        run = runs[0]
        node = db.execute(
            "SELECT * FROM nodes WHERE run_id=? AND node_id=?",
            (run["run_id"], run["best_node_id"]),
        ).fetchone()
        if node is None or node["backend_type"] != "triton_h100" or node["profile_json"] is None:
            raise ValueError(f"missing valid profiled Triton winner: {trace}")
        source = node["program_text"].encode()
        source_hash = hashlib.sha256(source).hexdigest()
        if source_hash != node["source_hash"]:
            raise ValueError(f"source hash mismatch: {trace}")
        context = json.loads(node["metadata_json"])["compiler_context"]
        artifacts = context["ptx"]["artifacts"]
        if len(artifacts) != 1 or artifacts[0]["storage_status"] != "stored":
            raise ValueError(f"PTX missing from node: {trace}")
        expected_ptx_hash = artifacts[0]["sha256"]
        row = db.execute(
            """SELECT a.content FROM node_compiler_artifacts n
               JOIN compiler_artifacts a ON a.sha256=n.sha256
               WHERE n.run_id=? AND n.node_id=? AND n.sha256=? AND a.kind='ptx'""",
            (run["run_id"], run["best_node_id"], expected_ptx_hash),
        ).fetchone()
        if row is None or hashlib.sha256(row[0]).hexdigest() != expected_ptx_hash:
            raise ValueError(f"PTX hash mismatch: {trace}")
        write_exact(HERE / f"trace_{arm}.py", source)
        write_exact(HERE / f"trace_{arm}.ptx", row[0])
        benchmark = json.loads(node["benchmark_json"])
        profile = json.loads(node["profile_json"])
        return {
            "arm": arm, "trace": str(trace.relative_to(HERE.parents[2])),
            "run_id": run["run_id"], "node_id": run["best_node_id"],
            "source_sha256": source_hash, "ptx_sha256": expected_ptx_hash,
            "binary_hash": node["binary_hash"],
            "median_us": benchmark["median_us"],
            "timing_samples_us": benchmark["timings_us"],
            "compiler_resources": context["resources"],
            "profile_summary": profile["summary"],
            "environment_manifest_id": node["environment_manifest_id"],
        }


def main() -> None:
    if (HERE / "trace_manifest.json").exists():
        raise FileExistsError(HERE / "trace_manifest.json")
    items = [extract_arm(arm, FACTORIAL / f"corrected-b10-s0-{arm}.sqlite")
             for arm in ARMS]
    (HERE / "trace_manifest.json").write_text(json.dumps({
        "schema_version": 1, "selection": "closed corrected seed-0 best nodes",
        "arms": items,
    }, indent=2, sort_keys=True) + "\n")
    for item in items:
        print(item["arm"], item["median_us"], item["source_sha256"], item["ptx_sha256"])


if __name__ == "__main__":
    main()
