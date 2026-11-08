"""Audit verified best-node source and retained PTX across the matched pilot."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"


def audit(path: Path) -> dict:
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        run = db.execute("SELECT * FROM search_runs").fetchone()
        if run is None or run["ended_at"] is None:
            raise ValueError(f"incomplete trace: {path}")
        node = db.execute("SELECT * FROM nodes WHERE node_id=?",
                          (run["best_node_id"],)).fetchone()
        if node is None:
            raise ValueError("best node missing")
        source = node["program_text"]
        if hashlib.sha256(source.encode()).hexdigest() != node["source_hash"]:
            raise ValueError("source hash mismatch")
        context = json.loads(node["metadata_json"])["compiler_context"]
        artifacts = db.execute(
            "SELECT a.sha256, a.kind, a.size_bytes, a.content FROM compiler_artifacts a "
            "JOIN node_compiler_artifacts n ON n.sha256=a.sha256 "
            "WHERE n.run_id=? AND n.node_id=?",
            (run["run_id"], node["node_id"]),
        ).fetchall()
        ptx = [item for item in artifacts if item["kind"] == "ptx"]
        if len(ptx) != 1:
            raise ValueError("expected one retained PTX artifact")
        artifact = ptx[0]
        content = bytes(artifact["content"])
        if hashlib.sha256(content).hexdigest() != artifact["sha256"] or len(content) != artifact["size_bytes"]:
            raise ValueError("retained PTX integrity failed")
        if artifact["sha256"] != context["ptx"]["artifacts"][0]["sha256"]:
            raise ValueError("PTX context hash mismatch")
        profile = json.loads(node["profile_json"])["summary"] if node["profile_json"] else None
        generation = db.execute("SELECT strategy_id FROM generations WHERE created_node_id=?",
                                (node["node_id"],)).fetchone()
        return {"trace": path.name, "seed": run["seed"],
                "catalog": json.loads(run["config_json"])["arm"],
                "best_median_us": json.loads(node["benchmark_json"])["median_us"],
                "source_sha256": node["source_hash"], "strategy_id": generation[0] if generation else None,
                "source_uses_pointer_load": "tl.load(" in source,
                "source_uses_descriptor_load": "_experimental_descriptor_load" in source,
                "ptx_sha256": artifact["sha256"],
                "ptx_size_bytes": artifact["size_bytes"],
                "ptx_has_cp_async": b"cp.async" in content,
                "ptx_has_tma_load": b"cp.async.bulk.tensor" in content,
                "ptx_sites": context["ptx"]["static_instruction_sites"],
                "resources": context["resources"],
                "sass_status": context["sass"]["status"],
                "profile_summary": {key: profile.get(key) for key in (
                    "dram_bytes_read", "instructions_executed", "tensor_pipe_utilization_pct")}
                if profile else None}


def main() -> None:
    records = [audit(OUTPUT / f"b10-s{seed}-{arm}.sqlite")
               for seed in (0, 1) for arm in ("old", "new")]
    (HERE / "best_audit.json").write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    for item in records:
        print(item["trace"], item["best_median_us"],
              "TMA", item["source_uses_descriptor_load"],
              "registers", item["resources"]["registers_per_thread"])


if __name__ == "__main__":
    main()
