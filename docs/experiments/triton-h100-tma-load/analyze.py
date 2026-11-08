"""Check paired timing and confirm the descriptor path in executed SASS."""

from __future__ import annotations

import importlib.util
import json
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"
NAMES = ("baseline", "tma", "cublas")


def _load_sass_parser():
    path = HERE.parent / "triton-h100-cublas-sass/analyze.py"
    spec = importlib.util.spec_from_file_location("matched_sass_analysis", path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot import SASS parser")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_analysis(report: dict, sass: dict[str, str], raw: dict[str, str]) -> dict:
    if not report.get("complete") or report.get("benchmark", {}).get("status") != "ok":
        raise ValueError("comparison incomplete")
    benchmark = report["benchmark"]
    if benchmark["source_sha256"] != report["source_sha256"]:
        raise ValueError("benchmark source mismatch")
    by_name = {name: {} for name in NAMES}
    for item in benchmark["observations"]:
        name, round_index = item["implementation"], item["round"]
        if name not in by_name or round_index in by_name[name]:
            raise ValueError("unexpected or duplicate timing observation")
        if len(item["timings_us"]) != 30 or statistics.median(item["timings_us"]) != item["median_us"]:
            raise ValueError("timing samples do not match median")
        by_name[name][round_index] = item["median_us"]
    if any(set(rounds) != {0, 1, 2} for rounds in by_name.values()):
        raise ValueError("missing timing round")
    parser = _load_sass_parser()
    profiles = {}
    for name in ("baseline", "tma"):
        profiles[name] = {
            "sass": parser.parse_sass_csv(sass[name]),
            "launch": parser.parse_raw_csv(raw[name]),
            "report_sha256": report["profiles"][name]["ncu_report_sha256"],
        }
    if profiles["tma"]["sass"]["opcode_executions"].get("UTMALDG", 0) == 0:
        raise ValueError("TMA variant has no executed TMA loads")
    if profiles["baseline"]["sass"]["opcode_executions"].get("UTMALDG", 0):
        raise ValueError("baseline unexpectedly has TMA loads")
    medians = {name: [by_name[name][index] for index in range(3)] for name in NAMES}
    deltas = [a - b for a, b in zip(medians["tma"], medians["baseline"])]
    return {"schema_version": 1, "source_sha256": report["source_sha256"],
            "correctness": benchmark["correctness"],
            "round_medians_us": medians,
            "tma_minus_baseline_us": deltas,
            "median_tma_minus_baseline_us": statistics.median(deltas),
            "profiles": profiles,
            "limitations": "NCU replay profiles are separate from event timing; this changes loads only, not CTA clustering or persistence."}


def main() -> None:
    report = json.loads((HERE / "results.json").read_text())
    analysis = build_analysis(
        report,
        {name: (OUTPUT / f"{name}.sass.csv").read_text() for name in ("baseline", "tma")},
        {name: (OUTPUT / f"{name}.raw.csv").read_text() for name in ("baseline", "tma")},
    )
    (HERE / "analysis.json").write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    print("TMA minus baseline (median paired µs):", analysis["median_tma_minus_baseline_us"])
    for name in ("baseline", "tma"):
        print(name, analysis["profiles"][name]["launch"]["kernel_name"])
        print(name, "TMA loads", analysis["profiles"][name]["sass"]["opcode_executions"].get("UTMALDG", 0))


if __name__ == "__main__":
    main()
