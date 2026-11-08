"""Summarize matched CUDA Graph timing and NCU evidence."""

from __future__ import annotations

import importlib.util
import json
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
ARMS = ("gluon", "cublas")


def profile_parser():
    path = HERE.parent / "triton-h100-gluon-vs-triton-profile" / "analyze.py"
    spec = importlib.util.spec_from_file_location("matched_profile_parser", path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load profile parser")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def analyze() -> dict:
    run = json.loads((HERE / "results.json").read_text())
    paired = json.loads((HERE / "output/paired.json").read_text())
    if not run["complete"] or not paired["complete"] or len(paired["observations"]) != 16:
        raise ValueError("incomplete matched comparison")
    if (run["source_sha256"] != paired["source_sha256"] or
            run["helper_sha256"] != paired["helper_sha256"] or
            not all(paired["correctness"].values())):
        raise ValueError("paired source or correctness mismatch")
    rounds = {}
    for observation in paired["observations"]:
        rounds.setdefault(observation["round"], {})[observation["name"]] = (
            observation["benchmark"]["median_us"])
    if set(rounds) != set(range(8)) or any(set(row) != set(ARMS) for row in rounds.values()):
        raise ValueError("missing balanced timing round")
    parser = profile_parser()
    profiles = {}
    for name in ARMS:
        record = run["profiles"][name]
        if record["status"] != "ok" or not record["correctness_exact"]:
            raise ValueError(f"failed profile: {name}")
        profiles[name] = {
            "raw": parser.parse_raw(HERE / "output" / f"{name}.raw.csv"),
            "sass": parser.parse_sass(HERE / "output" / f"{name}.sass.csv"),
        }
    differences = [row["gluon"] - row["cublas"] for row in rounds.values()]
    return {"schema_version": 1, "run_id": run["run_id"],
            "rounds_us": rounds,
            "median_us": {name: statistics.median(row[name] for row in rounds.values())
                          for name in ARMS},
            "gluon_minus_cublas_us": {"round_differences": differences,
                                      "median": statistics.median(differences),
                                      "gluon_faster_rounds": sum(value < 0 for value in differences)},
            "profiles": profiles}


if __name__ == "__main__":
    (HERE / "analysis.json").write_text(json.dumps(analyze(), indent=2, sort_keys=True) + "\n")
