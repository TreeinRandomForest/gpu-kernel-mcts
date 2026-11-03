"""Summarize matched-input timing and dynamically executed SASS opcodes."""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"
NAMES = ("triton", "cublas")
RAW_METRICS = (
    "smsp__inst_executed.sum", "sm__inst_executed_pipe_tensor.sum",
    "dram__bytes_read.sum", "dram__bytes_write.sum",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    "launch__registers_per_thread", "launch__shared_mem_per_block",
    "launch__cluster_size", "launch__grid_size", "launch__block_size",
)


def parse_sass_csv(csv_text: str) -> dict:
    rows = list(csv.reader(csv_text.splitlines()))
    header = next((row for row in rows if "Address" in row and "Source" in row
                   and "Instructions Executed" in row), None)
    if header is None:
        raise ValueError("NCU SASS export lacks instruction columns")
    address_index = header.index("Address")
    source_index = header.index("Source")
    count_index = header.index("Instructions Executed")
    sites = Counter()
    executions = Counter()
    for row in rows[rows.index(header) + 1:]:
        if len(row) <= max(address_index, source_index, count_index):
            continue
        if not row[address_index].startswith("0x"):
            continue
        words = row[source_index].strip().split()
        if words and words[0].startswith("@"):
            words = words[1:]
        if not words:
            continue
        opcode = words[0].split(".", 1)[0].upper()
        try:
            count = int(row[count_index].replace(",", ""))
        except ValueError:
            continue
        sites[opcode] += 1
        executions[opcode] += count
    if not sites:
        raise ValueError("NCU SASS export has no parsed sites")
    return {
        "site_count": sum(sites.values()),
        "executed_warp_instructions": sum(executions.values()),
        "opcode_sites": dict(sorted(sites.items())),
        "opcode_executions": dict(sorted(executions.items())),
        "top_executed_opcodes": [
            {"opcode": opcode, "warp_instructions": count,
             "static_sites": sites[opcode]}
            for opcode, count in executions.most_common(25)
        ],
    }


def parse_raw_csv(csv_text: str) -> dict:
    rows = list(csv.reader(csv_text.splitlines()))
    header = next((row for row in rows if "Kernel Name" in row and
                   "smsp__inst_executed.sum" in row), None)
    if header is None:
        raise ValueError("NCU raw export lacks selected launch metrics")
    units = rows[rows.index(header) + 1]
    data = [row for row in rows[rows.index(header) + 2:]
            if len(row) == len(header) and row[0].isdigit()]
    if len(data) != 1:
        raise ValueError(f"expected one NCU launch, found {len(data)}")
    values = dict(zip(header, data[0]))
    unit_values = dict(zip(header, units))
    return {"kernel_name": values["Kernel Name"],
            "block_size": values.get("Block Size"),
            "grid_size": values.get("Grid Size"),
            "metrics": {key: {"value": float(values[key]), "unit": unit_values.get(key, "")}
                        for key in RAW_METRICS if values.get(key) not in (None, "")}}


def build_analysis(report: dict, sass: dict[str, str], raw: dict[str, str]) -> dict:
    if not report.get("complete") or report.get("benchmark", {}).get("status") != "ok":
        raise ValueError("matched-input comparison incomplete")
    benchmark = report["benchmark"]
    if benchmark["source_sha256"] != report["source_sha256"]:
        raise ValueError("benchmark source differs from recorded source")
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
    profiles = {}
    for name in NAMES:
        profiles[name] = {
            "sass": parse_sass_csv(sass[name]),
            "launch": parse_raw_csv(raw[name]),
            "report_sha256": report["profiles"][name]["ncu_report_sha256"],
        }
        ncu_total = profiles[name]["launch"]["metrics"]["smsp__inst_executed.sum"]["value"]
        profiles[name]["sass_minus_raw_instructions"] = (
            profiles[name]["sass"]["executed_warp_instructions"] - ncu_total
        )
        profiles[name]["sass_matches_ncu_total"] = (
            profiles[name]["sass_minus_raw_instructions"] == 0
        )
    triton_medians = [by_name["triton"][index] for index in range(3)]
    cublas_medians = [by_name["cublas"][index] for index in range(3)]
    deltas = [a - b for a, b in zip(triton_medians, cublas_medians)]
    return {
        "schema_version": 1, "status": "matched_input_sass_comparison",
        "source_sha256": report["source_sha256"],
        "maximum_error": benchmark["maximum_error"],
        "round_medians_us": {"triton": triton_medians, "cublas": cublas_medians},
        "paired_deltas_us": deltas, "median_paired_delta_us": statistics.median(deltas),
        "profiles": profiles,
        "limitations": (
            "NCU reports are separate application replays and are not timing samples. "
            "Source-page instruction totals can differ from raw counters collected "
            "on another replay pass. "
            "SASS opcode counts are warp-level executions; instruction shapes and work "
            "assigned to each CTA may differ. The PyTorch/cuBLAS dispatch identity "
            "must be compared with the earlier C++ cublasGemmEx kernel."
        ),
    }


def main() -> None:
    report = json.loads((HERE / "results.json").read_text())
    analysis = build_analysis(
        report,
        {name: (OUTPUT / f"{name}.sass.csv").read_text() for name in NAMES},
        {name: (OUTPUT / f"{name}.raw.csv").read_text() for name in NAMES},
    )
    (HERE / "analysis.json").write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    print("paired gap", analysis["median_paired_delta_us"])
    for name in NAMES:
        print(name, analysis["profiles"][name]["launch"]["kernel_name"])
        print(analysis["profiles"][name]["sass"]["top_executed_opcodes"][:12])


if __name__ == "__main__":
    main()
