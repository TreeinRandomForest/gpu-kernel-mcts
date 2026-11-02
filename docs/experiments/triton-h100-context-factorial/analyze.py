"""Reconcile the twelve completed factorial traces, excluding pilot and failed traces."""

from __future__ import annotations

import json
import sqlite3
import statistics
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"
EXPECTED = {"A": (False, False), "B": (True, False),
            "C": (False, True), "D": (True, True)}


def trace_path(seed: int, arm: str) -> Path:
    return OUTPUT / f"corrected-b10-s{seed}-{arm}.sqlite"


def inspect(seed: int, arm: str) -> dict[str, object]:
    path = trace_path(seed, arm)
    with sqlite3.connect(path) as connection:
        run = connection.execute(
            "SELECT run_id, config_json, ended_at, best_node_id, final_b_gen, "
            "final_profile_calls FROM search_runs"
        ).fetchone()
        if run is None or run[2] is None or run[4] != 10:
            raise ValueError(f"incomplete trace: {path}")
        config = json.loads(run[1])
        expected_profile, expected_compiler = EXPECTED[arm]
        if config["ablation_arm"] != arm or config["seed"] != seed \
                or config["mcts"]["profile_metric_set"] != "spill_v1" \
                or config["mcts"]["profile_prompt_view"] != "ncu_only" \
                or config["mcts"]["include_parent_profile"] != expected_profile \
                or config["mcts"]["include_compiler_summary"] != expected_compiler:
            raise ValueError(f"configuration mismatch: {path}")
        generations = connection.execute(
            "SELECT proposal_status, strategy_id, prompt_text, metadata_json, "
            "llm_latency_seconds, invalid_reason FROM generations ORDER BY b_gen"
        ).fetchall()
        if len(generations) != 10:
            raise ValueError(f"generation mismatch: {path}")
        statuses: Counter[str] = Counter()
        strategies: Counter[str] = Counter()
        invalid_reasons: Counter[str] = Counter()
        gateway_errors: Counter[str] = Counter()
        max_llm_latency = 0.0
        for status, strategy, prompt_text, metadata_text, llm_latency, invalid_reason in generations:
            statuses[status] += 1
            strategies[strategy] += 1
            if invalid_reason:
                invalid_reasons[invalid_reason] += 1
            if llm_latency is not None:
                max_llm_latency = max(max_llm_latency, llm_latency)
            metadata = json.loads(metadata_text)
            generation = metadata.get("generation", {})
            if generation.get("gateway_error"):
                gateway_errors[generation["gateway_error"]] += 1
            if generation.get("model") != "rits/zai-org/glm-5-3" \
                    or generation.get("thinking_mode") != "disabled":
                raise ValueError(f"model mismatch: {path}")
            if not prompt_text:
                raise ValueError(f"missing prompt: {path}")
            prompt = json.loads(prompt_text)
            profile = prompt.get("parent_profile")
            compiler = prompt.get("parent_compiler_summary")
            if (profile is not None) != expected_profile \
                    or (compiler is not None) != expected_compiler:
                raise ValueError(f"prompt factor mismatch: {path}")
            if expected_profile:
                if profile.get("metric_set") != "spill_v1" \
                        or any(key.startswith("compiler_") or key == "registers_per_thread"
                               for key in profile.get("summary", {})) \
                        or "launch__registers_per_thread" in profile.get("metrics", {}) \
                        or "launch__registers_per_thread" in profile.get("resolved_metrics", []):
                    raise ValueError(f"profile prompt mask mismatch: {path}")
            if expected_compiler and (compiler.get("status") != "available" \
                    or not compiler.get("ptx_static_instruction_sites")):
                raise ValueError(f"compiler prompt view missing: {path}")
        best = connection.execute(
            "SELECT benchmark_json, metadata_json, profile_json, source_hash "
            "FROM nodes WHERE run_id=? AND node_id=?",
            (run[0], run[3]),
        ).fetchone()
        if best is None:
            raise ValueError(f"missing best node: {path}")
        best_metadata = json.loads(best[1])
        best_profile = json.loads(best[2]) if best[2] else {}
        first_parent = connection.execute(
            "SELECT parent_node_id FROM generations ORDER BY b_gen LIMIT 1"
        ).fetchone()[0]
        root_benchmark = connection.execute(
            "SELECT benchmark_json FROM nodes WHERE run_id=? AND node_id=?",
            (run[0], first_parent),
        ).fetchone()
        if root_benchmark is None:
            raise ValueError(f"missing root node: {path}")
        profiles = [json.loads(row[0]) for row in connection.execute(
            "SELECT profile_json FROM nodes WHERE profile_json IS NOT NULL"
        )]
        if not profiles or any(profile.get("metric_set") != "spill_v1" for profile in profiles):
            raise ValueError(f"missing stored spill_v1 profiles: {path}")
        return {
            "seed": seed, "arm": arm, "trace": path.name, "run_id": run[0],
            "best_median_us": json.loads(best[0])["median_us"],
            "root_median_us": json.loads(root_benchmark[0])["median_us"],
            "best_source_hash": best[3],
            "best_compiler_resources": best_metadata.get("compiler_context", {}).get("resources"),
            "best_profile_summary": {
                key: best_profile.get("summary", {}).get(key)
                for key in ("tensor_pipe_utilization_pct", "local_load_sectors",
                            "local_store_sectors", "dram_bytes_read", "dram_bytes_written")
            },
            "generations": len(generations), "valid_generations": statuses["VALID"],
            "invalid_generations": len(generations) - statuses["VALID"],
            "invalid_reasons": dict(invalid_reasons),
            "gateway_errors": dict(gateway_errors),
            "strategies": dict(strategies), "stored_profiles": len(profiles),
            "profile_calls": run[5], "max_llm_latency_seconds": max_llm_latency,
            "worker_image": config["worker_image"],
            "worker_code_overlay_sha256": config["worker_code_overlay_sha256"],
            "strategy_catalog_sha256": config["strategy_catalog_sha256"],
            "root_source_sha256": config["root_source_sha256"],
        }


def main() -> None:
    results = [inspect(seed, arm) for seed in range(3) for arm in EXPECTED]
    for field in ("worker_image", "strategy_catalog_sha256", "root_source_sha256"):
        if len({result[field] for result in results}) != 1:
            raise ValueError(f"inconsistent {field}")
    summary = {
        "schema_version": 1,
        "runs": results,
        "arms": {
            arm: {
                "best_latency_median_us": statistics.median(
                    result["best_median_us"] for result in results if result["arm"] == arm
                ),
                "valid_generations": sum(result["valid_generations"] for result in results
                                         if result["arm"] == arm),
                "total_generations": 30,
            }
            for arm in EXPECTED
        },
        "excluded_exploratory_traces": ["factorial-b10-*.sqlite",
                                         "factorial-retry-b10-*.sqlite"],
    }
    (OUTPUT / "analysis.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    for result in results:
        print(result["seed"], result["arm"], round(result["best_median_us"], 3),
              f'{result["valid_generations"]}/10 valid',
              f'{result["stored_profiles"]} profiles')
    print("median best by arm:", {arm: round(value["best_latency_median_us"], 3)
                                  for arm, value in summary["arms"].items()})


if __name__ == "__main__":
    main()
