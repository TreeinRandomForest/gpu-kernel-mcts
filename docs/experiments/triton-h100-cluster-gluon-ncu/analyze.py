"""Extract a small, auditable comparison from the three NCU reports."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
ARMS = ("single_pipeline", "cluster_pipeline_ordinary", "cluster_pipeline_multicast")
METRICS = (
    "gpu__time_duration.sum",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "lts__t_sectors_srcunit_tex_op_read.sum",
    "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum",
    "smsp__inst_executed.sum",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
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
    if raw["Kernel Name"] != "gemm":
        raise ValueError(f"unexpected kernel: {raw['Kernel Name']}")
    return {name: {"value": float(raw[name]), "unit": unit_map[name]}
            for name in METRICS}


def parse_sass(path: Path) -> dict:
    rows = list(csv.reader(path.open(newline="")))
    if rows[0][1] != "gemm":
        raise ValueError(f"unexpected SASS kernel in {path}")
    header = rows[1]
    source = header.index("Source")
    executed = header.index("Instructions Executed")
    samples = header.index("Warp Stall Sampling (All Samples)")
    dynamic = Counter()
    sites = Counter()
    sampled = Counter()
    for row in rows[2:]:
        instruction = re.sub(r"^@!?[A-Z]+\d+\s+", "", row[source].strip())
        if not instruction:
            continue
        opcode = instruction.split(None, 1)[0]
        key = opcode if opcode == "UTMALDG.2D.MULTICAST" else opcode.split(".")[0]
        dynamic[key] += int(float(row[executed] or 0))
        sites[key] += 1
        sampled[key] += int(float(row[samples] or 0))
    names = ("UTMALDG", "UTMALDG.2D.MULTICAST", "HGMMA", "WARPGROUP", "BAR", "UTMASTG")
    return {"dynamic_executions": {name: dynamic[name] for name in names},
            "static_sites": {name: sites[name] for name in names},
            "stall_samples_at_sites": {name: sampled[name] for name in names},
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
        if record["source_sha256"] != run["source_sha256"]:
            raise ValueError(f"source mismatch: {name}")
        raw = HERE / "output" / f"{name}.raw.csv"
        sass = HERE / "output" / f"{name}.sass.csv"
        if sha256(raw) != record["raw_csv_sha256"] or sha256(sass) != record["sass_csv_sha256"]:
            raise ValueError(f"CSV hash mismatch: {name}")
        profiles[name] = {"metrics": parse_raw(raw), "sass": parse_sass(sass),
                          "cubin_sha256": record["cubin_sha256"]}
    return {"schema_version": 1, "run_id": run["run_id"],
            "source_sha256": run["source_sha256"], "profiles": profiles}


if __name__ == "__main__":
    (HERE / "analysis.json").write_text(json.dumps(analyze(), indent=2, sort_keys=True) + "\n")
