"""Check that a completed Triton search trace uses one Graph timing contract."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


CONTRACT = "triton_h100_cuda_graph_v1"


def audit(path: Path) -> dict[str, object]:
    with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as connection:
        runs = connection.execute(
            "SELECT run_id, algorithm, config_json, ended_at, final_b_gen FROM search_runs"
        ).fetchall()
        if len(runs) != 1:
            raise ValueError(f"{path}: expected exactly one search run")
        run_id, algorithm, config_text, ended_at, generations = runs[0]
        config = json.loads(config_text)
        if ended_at is None or config.get("timing_contract") != CONTRACT:
            raise ValueError(f"{path}: run is incomplete or uses another timing contract")
        calibrations = connection.execute(
            "SELECT payload_json FROM search_events "
            "WHERE run_id = ? AND event_type = 'worker_calibration'", (run_id,),
        ).fetchall()
        if len(calibrations) != 1:
            raise ValueError(f"{path}: expected one worker calibration")
        calibration = json.loads(calibrations[0][0])
        if calibration.get("timing_contract") != CONTRACT:
            raise ValueError(f"{path}: worker calibration uses another timing contract")
        benchmarks = {"root": calibration.get("benchmark")}
        for name in ("cublas", "cutlass"):
            entry = calibration.get("vendor_baselines", {}).get(name, {})
            if entry.get("correctness", {}).get("success") is not True:
                raise ValueError(f"{path}: {name} calibration failed correctness")
            benchmarks[name] = entry.get("benchmark")
        nodes = connection.execute(
            "SELECT node_id, benchmark_json FROM nodes WHERE run_id = ?", (run_id,),
        ).fetchall()
        for node_id, benchmark_text in nodes:
            benchmarks[f"node:{node_id}"] = json.loads(benchmark_text) if benchmark_text else None
        candidate_rows = connection.execute(
            "SELECT generation_id, benchmark_json FROM generations "
            "WHERE run_id = ? AND benchmark_json IS NOT NULL", (run_id,),
        ).fetchall()
        for generation_id, benchmark_text in candidate_rows:
            benchmarks[f"generation:{generation_id}"] = json.loads(benchmark_text)
        for name, benchmark in benchmarks.items():
            if not isinstance(benchmark, dict) or benchmark.get("timing_contract") != CONTRACT:
                raise ValueError(f"{path}: {name} benchmark uses another timing contract")
        recorded_generations = connection.execute(
            "SELECT COUNT(*) FROM generations WHERE run_id = ?", (run_id,),
        ).fetchone()[0]
        if recorded_generations != generations:
            raise ValueError(f"{path}: generation count mismatch")
        best = min(float(json.loads(text)["median_us"]) for _, text in nodes)
        return {
            "trace": str(path), "run_id": run_id, "algorithm": algorithm,
            "timing_contract": CONTRACT, "generations": generations,
            "nodes": len(nodes), "benchmarked_generations": len(candidate_rows),
            "best_median_us": best,
            "root_median_us": benchmarks["root"]["median_us"],
            "cublas_median_us": benchmarks["cublas"]["median_us"],
            "cutlass_median_us": benchmarks["cutlass"]["median_us"],
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+", type=Path)
    arguments = parser.parse_args()
    for path in arguments.traces:
        print(json.dumps(audit(path), sort_keys=True))


if __name__ == "__main__":
    main()
