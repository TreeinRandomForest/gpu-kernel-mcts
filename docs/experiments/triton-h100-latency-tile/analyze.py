"""Compare matched TMA tile shapes with NCU launch metrics and executed SASS."""

from __future__ import annotations

import csv
import importlib.util
import json
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"
NAMES = ("tile_128x256_g8", "tile_256x128_g8")
RAW_METRICS = (
    "launch__registers_per_thread", "launch__shared_mem_per_block",
    "smsp__inst_executed.sum",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__throughput.avg.pct_of_peak_sustained_active",
    "dram__bytes_read.sum", "lts__t_sector_op_read_hit_rate.pct",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum",
)


def _load_sass_parser():
    path = HERE.parent / "triton-h100-cublas-sass/analyze.py"
    spec = importlib.util.spec_from_file_location("matched_sass_analysis", path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot import SASS parser")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.parse_sass_csv


def parse_raw(csv_text: str) -> dict:
    rows = list(csv.reader(csv_text.splitlines()))
    header = next((row for row in rows if "Kernel Name" in row and
                   "smsp__inst_executed.sum" in row), None)
    if header is None:
        raise ValueError("NCU raw export lacks selected metrics")
    values = [row for row in rows[rows.index(header) + 2:]
              if len(row) == len(header) and row[0].isdigit()]
    if len(values) != 1:
        raise ValueError(f"expected one NCU launch, found {len(values)}")
    row = dict(zip(header, values[0]))
    missing = [metric for metric in RAW_METRICS if row.get(metric) in (None, "")]
    if missing:
        raise ValueError(f"NCU raw export missing metrics: {missing}")
    return {"kernel_name": row["Kernel Name"],
            "metrics": {metric: float(row[metric]) for metric in RAW_METRICS}}


def build_analysis(report: dict, raw: dict[str, str], sass: dict[str, str]) -> dict:
    if not report.get("complete") or report.get("benchmark", {}).get("status") != "ok":
        raise ValueError("latency comparison incomplete")
    if report["source_sha256"] != report["benchmark"]["source_sha256"]:
        raise ValueError("source identity mismatch")
    rounds = {name: {} for name in NAMES}
    for item in report["benchmark"]["observations"]:
        name = item["implementation"]
        if name not in rounds:
            continue
        index = item["round"]
        if index in rounds[name]:
            raise ValueError("duplicate timing observation")
        if len(item["timings_us"]) != 30 or statistics.median(item["timings_us"]) != item["median_us"]:
            raise ValueError("timing median mismatch")
        rounds[name][index] = item["median_us"]
    if any(set(value) != {0, 1, 2} for value in rounds.values()):
        raise ValueError("missing timing round")
    if set(raw) != set(NAMES) or set(sass) != set(NAMES) or set(report["profiles"]) != set(NAMES):
        raise ValueError("missing or unexpected NCU profile")
    parse_sass = _load_sass_parser()
    profiles = {}
    for name in NAMES:
        profiles[name] = {"raw": parse_raw(raw[name]), "sass": parse_sass(sass[name]),
                          "report_sha256": report["profiles"][name]["ncu_report_sha256"]}
    slow = [rounds[NAMES[1]][i] - rounds[NAMES[0]][i] for i in range(3)]
    return {"schema_version": 1, "source_sha256": report["source_sha256"],
            "correctness": report["benchmark"]["correctness"],
            "round_medians_us": {name: [rounds[name][i] for i in range(3)] for name in NAMES},
            "tile_256x128_minus_128x256_us": slow,
            "median_paired_delta_us": statistics.median(slow),
            "profiles": profiles,
            "limitations": "NCU raw and Source counters may use different replay passes; stalls and utilization identify associations, not isolated latency causes."}


def main() -> None:
    report = json.loads((HERE / "results.json").read_text())
    analysis = build_analysis(
        report,
        {name: (OUTPUT / f"{name}.raw.csv").read_text() for name in NAMES},
        {name: (OUTPUT / f"{name}.sass.csv").read_text() for name in NAMES},
    )
    (HERE / "analysis.json").write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    print("paired 256x128 minus 128x256 µs", analysis["median_paired_delta_us"])
    for name in NAMES:
        profile = analysis["profiles"][name]
        print(name, "registers", profile["raw"]["metrics"]["launch__registers_per_thread"],
              "tensor active", profile["raw"]["metrics"][
                  "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"],
              "raw instructions", profile["raw"]["metrics"]["smsp__inst_executed.sum"])


if __name__ == "__main__":
    main()
