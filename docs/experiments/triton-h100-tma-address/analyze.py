"""Validate paired TMA source edits and summarize their measured effects."""

from __future__ import annotations

import importlib.util
import json
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"
NAMES = ("baseline", "unmasked_store", "constant_group", "both")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_analysis(report: dict, raw: dict[str, str], sass: dict[str, str]) -> dict:
    if not report.get("complete") or report.get("benchmark", {}).get("status") != "ok":
        raise ValueError("experiment incomplete")
    benchmark = report["benchmark"]
    if benchmark["source_sha256"] != report["variant_sha256"]:
        raise ValueError("variant source mismatch")
    if set(report["profiles"]) != set(NAMES) or set(raw) != set(NAMES) or set(sass) != set(NAMES):
        raise ValueError("profile set incomplete")
    rounds = {name: {} for name in (*NAMES, "cublas")}
    for item in benchmark["observations"]:
        name, index = item["implementation"], item["round"]
        if name not in rounds or index in rounds[name]:
            raise ValueError("unexpected or duplicate timing observation")
        if len(item["timings_us"]) != 30 or statistics.median(item["timings_us"]) != item["median_us"]:
            raise ValueError("invalid timing samples")
        rounds[name][index] = item["median_us"]
    if any(set(values) != {0, 1, 2} for values in rounds.values()):
        raise ValueError("missing timing round")
    latency = load_module("tile_latency_analysis", HERE.parent / "triton-h100-latency-tile/analyze.py")
    sass_parser = load_module("matched_sass_analysis", HERE.parent / "triton-h100-cublas-sass/analyze.py")
    profiles = {name: {
        "raw": latency.parse_raw(raw[name]),
        "sass": sass_parser.parse_sass_csv(sass[name]),
        "ncu_report_sha256": report["profiles"][name]["ncu_report_sha256"],
    } for name in NAMES}
    deltas = {name: [rounds[name][i] - rounds["baseline"][i] for i in range(3)]
              for name in NAMES if name != "baseline"}
    return {"schema_version": 1, "source_sha256": report["variant_sha256"],
            "correctness": benchmark["correctness"],
            "round_medians_us": {name: [rounds[name][i] for i in range(3)]
                                 for name in rounds},
            "paired_delta_from_baseline_us": deltas,
            "median_paired_delta_us": {name: statistics.median(values)
                                       for name, values in deltas.items()},
            "profiles": profiles,
            "limitations": "Full NCU metrics may be replayed separately and are not a latency decomposition. The specialization is tested only for the aligned 4096^3 workload."}


def main() -> None:
    report = json.loads((HERE / "results.json").read_text())
    analysis = build_analysis(
        report,
        {name: (OUTPUT / f"{name}.raw.csv").read_text() for name in NAMES},
        {name: (OUTPUT / f"{name}.sass.csv").read_text() for name in NAMES},
    )
    (HERE / "analysis.json").write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    print(json.dumps(analysis["median_paired_delta_us"], sort_keys=True))


if __name__ == "__main__":
    main()
