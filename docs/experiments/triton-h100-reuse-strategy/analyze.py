"""Audit the completed matched MCTS pilot without exposing prompt/source text."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"
BUDGET = 10
SEEDS = (0, 1)


def audit(path: Path) -> dict:
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        run = db.execute("SELECT * FROM search_runs").fetchall()
        if len(run) != 1 or run[0]["ended_at"] is None or run[0]["final_b_gen"] != BUDGET:
            raise ValueError(f"incomplete trace: {path}")
        generations = db.execute("SELECT strategy_id, proposal_status, repair_attempt, "
                                 "json_extract(benchmark_json, '$.median_us') AS median_us, "
                                 "prompt_text FROM generations ORDER BY b_gen").fetchall()
        if len(generations) != BUDGET:
            raise ValueError(f"wrong generation count: {path}")
        if any(json.loads(row["prompt_text"])["parent_profile"]["metric_set"] != "reuse_v1"
               for row in generations):
            raise ValueError(f"prompt lacks reuse_v1: {path}")
        strategies: dict[str, dict] = {}
        for row in generations:
            item = strategies.setdefault(row["strategy_id"],
                                         {"calls": 0, "valid": 0, "best_median_us": None})
            item["calls"] += 1
            if row["proposal_status"] == "VALID":
                item["valid"] += 1
                value = row["median_us"]
                if value is not None:
                    item["best_median_us"] = min(item["best_median_us"] or value, value)
        root = db.execute("SELECT json_extract(benchmark_json, '$.median_us') FROM nodes "
                          "WHERE reward=0 ORDER BY rowid LIMIT 1").fetchone()
        best = db.execute("SELECT json_extract(benchmark_json, '$.median_us') FROM nodes "
                          "WHERE node_id=?", (run[0]["best_node_id"],)).fetchone()
        config = json.loads(run[0]["config_json"])
        return {"trace": path.name, "model": run[0]["model_name"],
                "seed": run[0]["seed"], "budget": BUDGET,
                "catalog": config["arm"], "root_median_us": root[0],
                "best_median_us": best[0], "valid_generations": sum(
                    row["proposal_status"] == "VALID" for row in generations),
                "profile_calls": run[0]["final_profile_calls"],
                "strategies": strategies}


def main() -> None:
    arms = [audit(OUTPUT / f"b{BUDGET}-s{seed}-{arm}.sqlite")
            for seed in SEEDS for arm in ("old", "new")]
    by_key = {(item["seed"], item["catalog"]): item for item in arms}
    deltas = {str(seed): by_key[(seed, "new")]["best_median_us"]
              - by_key[(seed, "old")]["best_median_us"] for seed in SEEDS}
    report = {"schema_version": 1, "arms": arms,
              "new_minus_old_best_us": deltas,
              "interpretation": "Two B_gen=10 seeds are a pilot, not an efficacy estimate; the new action was sampled only once."}
    (HERE / "analysis.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"new_minus_old_best_us": deltas,
                      "reuse_action_calls": sum(
                          arm["strategies"].get("improve_cross_cta_reuse", {}).get("calls", 0)
                          for arm in arms)}, sort_keys=True))


if __name__ == "__main__":
    main()
