"""Validate and summarize read-sector hierarchy across CTA order/cache arms."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"
NAMES = ("group_1", "group_8", "group_32", "cublas")
CACHE_MODES = ("all", "none")
METRICS = (
    "dram__bytes_read.sum",
    "dram__sectors_read.sum",
    "lts__t_sectors_srcunit_tex_op_read.sum",
    "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum",
    "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum",
    "lts__t_sector_op_read_hit_rate.pct",
)


def parse_raw(csv_text: str) -> dict:
    rows = list(csv.reader(csv_text.splitlines()))
    header = next((row for row in rows if "Kernel Name" in row and
                   "dram__bytes_read.sum" in row), None)
    if header is None:
        raise ValueError("NCU raw export lacks read metrics")
    values = [row for row in rows[rows.index(header) + 2:]
              if len(row) == len(header) and row[0].isdigit()]
    if len(values) != 1:
        raise ValueError(f"expected one NCU launch, found {len(values)}")
    row = dict(zip(header, values[0]))
    missing = [metric for metric in METRICS if row.get(metric) in (None, "")]
    if missing:
        raise ValueError(f"NCU raw export missing metrics: {missing}")
    return {"kernel_name": row["Kernel Name"],
            "metrics": {metric: float(row[metric]) for metric in METRICS}}


def build_analysis(report: dict, raw: dict[str, str]) -> dict:
    if not report.get("complete") or report.get("benchmark", {}).get("status") != "ok":
        raise ValueError("comparison incomplete")
    if report["source_sha256"] != report["benchmark"]["source_sha256"]:
        raise ValueError("source identity mismatch")
    rounds = {name: {} for name in NAMES}
    for item in report["benchmark"]["observations"]:
        name, index = item["implementation"], item["round"]
        if name not in rounds or index in rounds[name]:
            raise ValueError("unexpected or duplicate timing observation")
        if len(item["timings_us"]) != 30 or statistics.median(item["timings_us"]) != item["median_us"]:
            raise ValueError("timing median mismatch")
        rounds[name][index] = item["median_us"]
    if any(set(value) != {0, 1, 2} for value in rounds.values()):
        raise ValueError("missing timing round")
    expected = {f"{cache_mode}-{name}" for cache_mode in CACHE_MODES for name in NAMES}
    if set(raw) != expected or set(report["profiles"]) != expected:
        raise ValueError("missing or unexpected NCU profile")
    profiles = {key: parse_raw(raw[key]) for key in sorted(expected)}
    for key, profile in profiles.items():
        profile["report_sha256"] = report["profiles"][key]["ncu_report_sha256"]
    return {
        "schema_version": 1,
        "source_sha256": report["source_sha256"],
        "correctness": report["benchmark"]["correctness"],
        "round_medians_us": {name: [rounds[name][index] for index in range(3)]
                             for name in NAMES},
        "profiles": profiles,
        "limitations": (
            "L2 TEX read sectors are whole-kernel counters, not A/B-address-specific. "
            "L2 misses are not equal to DRAM sectors, and application replay with "
            "cache-control none preserves workload-dependent warm cache state."
        ),
    }


def main() -> None:
    report = json.loads((HERE / "results.json").read_text())
    raw = {f"{cache_mode}-{name}":
           (OUTPUT / f"{cache_mode}-{name}.raw.csv").read_text()
           for cache_mode in CACHE_MODES for name in NAMES}
    analysis = build_analysis(report, raw)
    (HERE / "analysis.json").write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    for cache_mode in CACHE_MODES:
        for name in NAMES:
            metrics = analysis["profiles"][f"{cache_mode}-{name}"]["metrics"]
            print(cache_mode, name, "DRAM MB", metrics["dram__bytes_read.sum"],
                  "L2 read sectors", metrics["lts__t_sectors_srcunit_tex_op_read.sum"],
                  "L2 read misses", metrics["lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum"])


if __name__ == "__main__":
    main()
