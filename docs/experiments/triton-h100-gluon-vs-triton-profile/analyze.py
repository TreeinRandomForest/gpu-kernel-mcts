"""Audit matched NCU counters and executed SASS for three GEMM kernels."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
ARMS = ("source_ordinary", "triton_b", "cublas")
METRICS = (
    "gpu__time_duration.sum",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "lts__t_sectors_srcunit_tex_op_read.sum",
    "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum",
    "smsp__inst_executed.sum",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "smsp__pcsamp_warps_issue_stalled_barrier",
    "smsp__pcsamp_warps_issue_stalled_long_scoreboard",
    "smsp__pcsamp_warps_issue_stalled_wait",
    "smsp__pcsamp_warps_issue_stalled_warpgroup_arrive",
    "launch__waves_per_multiprocessor",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block",
    "launch__block_size",
    "launch__grid_size",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_raw(path: Path) -> dict:
    rows = list(csv.reader(path.open(newline="")))
    if len(rows) != 3:
        raise ValueError(f"expected one kernel and unit row: {path}")
    header, units, values = rows
    raw = dict(zip(header, values, strict=True))
    unit_map = dict(zip(header, units, strict=True))
    metrics = {}
    for name in METRICS:
        value = raw.get(name, "")
        metrics[name] = ({"value": float(value), "unit": unit_map[name]}
                         if value and value != "N/A" else None)
    return {"kernel_name": raw["Kernel Name"], "metrics": metrics}


def parse_sass(path: Path) -> dict:
    rows = list(csv.reader(path.open(newline="")))
    if rows[0][0] != "Kernel Name":
        raise ValueError(f"unexpected SASS header in {path}")
    header = rows[1]
    source = header.index("Source")
    executed = header.index("Instructions Executed")
    samples = header.index("Warp Stall Sampling (All Samples)")
    dynamic = Counter()
    sites = Counter()
    sampled = Counter()
    shapes = Counter()
    hotspots = []
    address = header.index("Address")
    for row in rows[2:]:
        instruction = re.sub(r"^@!?[A-Z]+\d+\s+", "", row[source].strip())
        if not instruction:
            continue
        opcode = instruction.split(None, 1)[0]
        key = ("UTMALDG.MULTICAST" if opcode.startswith("UTMALDG") and
               ".MULTICAST" in opcode else opcode.split(".")[0])
        dynamic[key] += int(float(row[executed] or 0))
        sites[key] += 1
        sampled[key] += int(float(row[samples] or 0))
        if opcode.startswith("HGMMA"):
            shapes[opcode] += int(float(row[executed] or 0))
        hotspots.append({"address": row[address], "source": row[source].strip(),
                         "samples": int(float(row[samples] or 0)),
                         "instructions_executed": int(float(row[executed] or 0))})
    names = ("UTMALDG", "UTMALDG.MULTICAST", "UTMASTG", "LDGSTS",
             "HGMMA", "HMMA", "WARPGROUP", "BAR", "UCGABAR_WAIT",
             "IMAD", "ISETP", "LDG", "STG", "BRA")
    return {"kernel_name": rows[0][1],
            "dynamic_executions": {name: dynamic[name] for name in names},
            "static_sites": {name: sites[name] for name in names},
            "stall_samples_at_sites": {name: sampled[name] for name in names},
            "source_page_instruction_total": sum(dynamic.values()),
            "top_dynamic_opcodes": dynamic.most_common(20),
            "hgmma_shapes": dict(shapes),
            "total_pc_samples": sum(item["samples"] for item in hotspots),
            "top_pc_samples": sorted(hotspots, key=lambda item: item["samples"],
                                     reverse=True)[:8],
            "warpgroup_wait_sites": sum(1 for row in rows[2:] if
                                        "WARPGROUP.DEPBAR.LE" in row[source]),
            "warpgroup_wait_executions": sum(int(float(row[executed] or 0))
                                              for row in rows[2:] if
                                              "WARPGROUP.DEPBAR.LE" in row[source]),
            "warpgroup_wait_stall_samples": sum(int(float(row[samples] or 0))
                                               for row in rows[2:] if
                                               "WARPGROUP.DEPBAR.LE" in row[source])}


def analyze() -> dict:
    run = json.loads((HERE / "results.json").read_text())
    if not run["complete"] or set(run["profiles"]) != set(ARMS):
        raise ValueError("incomplete three-arm profiling run")
    profiles = {}
    for name in ARMS:
        record = run["profiles"][name]
        if record["status"] != "ok" or not record["correctness"]["exact"]:
            raise ValueError(f"failed or incorrect profile: {name}")
        if (record["source_sha256"] != run["source_sha256"] or
                record["fast_b_sha256"] != run["fast_b_sha256"] or
                record["paired_sha256"] != run["paired_sha256"]):
            raise ValueError(f"source mismatch: {name}")
        raw = HERE / "output" / f"{name}.raw.csv"
        sass = HERE / "output" / f"{name}.sass.csv"
        if sha256(raw) != record["raw_csv_sha256"] or sha256(sass) != record["sass_csv_sha256"]:
            raise ValueError(f"CSV hash mismatch: {name}")
        raw_result = parse_raw(raw)
        sass_result = parse_sass(sass)
        profiles[name] = {"kernel_name": raw_result["kernel_name"],
                          "metrics": raw_result["metrics"], "sass": sass_result,
                          "cubin_sha256": record["cubin_sha256"]}
    return {"schema_version": 1, "run_id": run["run_id"],
            "source_sha256": run["source_sha256"],
            "fast_b_sha256": run["fast_b_sha256"], "profiles": profiles}


if __name__ == "__main__":
    (HERE / "analysis.json").write_text(json.dumps(analyze(), indent=2, sort_keys=True) + "\n")
