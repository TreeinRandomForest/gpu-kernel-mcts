"""Summarize paired timing effects and NCU evidence for Gluon factors."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def contrast(rounds: dict, a: str, b: str) -> dict:
    differences = [row[a] - row[b] for row in rounds.values()]
    return {"a_minus_b_median_us": statistics.median(differences),
            "a_faster_rounds": sum(value < 0 for value in differences),
            "round_differences_us": differences}


def analyze():
    run = json.loads((HERE / "results.json").read_text())
    paired = json.loads((HERE / "output/paired.json").read_text())
    if (not run["complete"] or not paired["complete"] or
            run["sha256"]["paired"] != hashlib.sha256((HERE / "paired.py").read_bytes()).hexdigest()):
        raise ValueError("incomplete or changed experiment")
    names = tuple(paired["names"])
    if names != tuple(run["expected_arms"]) or not all(paired["correctness"].values()):
        raise ValueError("arm or exactness mismatch")
    rounds = {}
    for observation in paired["observations"]:
        rounds.setdefault(observation["round"], {})[observation["name"]] = (
            observation["benchmark"]["median_us"])
    if (set(rounds) != set(range(len(names))) or
            any(set(row) != set(names) for row in rounds.values())):
        raise ValueError("incomplete balanced timing")
    parser = load(HERE.parent / "triton-h100-gluon-vs-triton-profile/analyze.py",
                  "factor_profile_parser")
    profiles = {}
    for name in names:
        if run["profiles"][name]["status"] != "ok":
            raise ValueError(f"failed profile: {name}")
        profiles[name] = {
            "raw": parser.parse_raw(HERE / "output" / f"{name}.raw.csv"),
            "sass": parser.parse_sass(HERE / "output" / f"{name}.sass.csv"),
        }
    effects = {}
    for bn in (256, 128):
        for order in ("mfast", "nfast"):
            regular = f"n{bn}_{order}_regular"
            persistent = f"n{bn}_{order}_persistent"
            effects[f"persistent_at_n{bn}_{order}"] = contrast(rounds, persistent, regular)
        for persistence in ("regular", "persistent"):
            mfast = f"n{bn}_mfast_{persistence}"
            nfast = f"n{bn}_nfast_{persistence}"
            effects[f"nfast_at_n{bn}_{persistence}"] = contrast(rounds, nfast, mfast)
    for order in ("mfast", "nfast"):
        for persistence in ("regular", "persistent"):
            narrow = f"n128_{order}_{persistence}"
            wide = f"n256_{order}_{persistence}"
            effects[f"n128_at_{order}_{persistence}"] = contrast(rounds, narrow, wide)
    return {"schema_version": 1, "run_id": run["run_id"],
            "median_us": {name: statistics.median(row[name] for row in rounds.values())
                          for name in names},
            "rounds_us": rounds, "effects": effects,
            "profiles": profiles}


if __name__ == "__main__":
    (HERE / "analysis.json").write_text(json.dumps(analyze(), indent=2, sort_keys=True) + "\n")
