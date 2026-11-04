"""Summarize paired timing and NCU Source-page PC samples for this experiment."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
ARMS = ("candidate", "control")
RAW_METRICS = (
    "gpu__time_duration.sum",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes_read.sum",
    "smsp__inst_executed.sum",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block",
)


def sass_samples(path: Path) -> dict:
    rows = list(csv.reader(path.open(newline="")))
    header = rows[1]
    source = header.index("Source")
    samples = header.index("Warp Stall Sampling (All Samples)")
    executed = header.index("Instructions Executed")
    sites = []
    for row in rows[2:]:
        instruction = row[source].strip()
        if not instruction:
            continue
        sites.append({"instruction": instruction,
                      "samples": int(float(row[samples] or 0)),
                      "executed": int(float(row[executed] or 0))})
    poll_branches = [sites[index + 1]["samples"]
                     for index, site in enumerate(sites[:-1])
                     if "SYNCS.PHASECHK.TRANS64.TRYWAIT" in site["instruction"] and
                     "BRA" in sites[index + 1]["instruction"]]
    return {"total_source_page_pc_samples": sum(site["samples"] for site in sites),
            "top_sites": sorted(sites, key=lambda site: site["samples"], reverse=True)[:10],
            "barrier_poll_branch_samples": sum(poll_branches),
            "warpgroup_wait_samples": sum(site["samples"] for site in sites
                                          if "WARPGROUP.DEPBAR.LE" in site["instruction"])}


def raw_metrics(path: Path) -> dict:
    rows = list(csv.reader(path.open(newline="")))
    if len(rows) != 3:
        raise ValueError("expected one profiled kernel")
    metrics = dict(zip(rows[0], rows[2], strict=True))
    units = dict(zip(rows[0], rows[1], strict=True))
    return {key: {"value": float(metrics[key]), "unit": units[key]}
            for key in RAW_METRICS}


def analyze() -> dict:
    run = json.loads((HERE / "results.json").read_text())
    paired = json.loads((HERE / "output/paired.json").read_text())
    if not run["complete"] or not paired["complete"]:
        raise ValueError("incomplete experiment")
    output = {"schema_version": 1, "run_id": run["run_id"],
              "source_sha256": paired["source_sha256"], "timing": {}, "profiles": {}}
    for name in ("candidate", "control", "triton_b", "cublas"):
        rounds = [item["benchmark"]["median_us"] for item in paired["observations"]
                  if item["name"] == name]
        if len(rounds) != 3:
            raise ValueError(f"missing paired timing: {name}")
        output["timing"][name] = {"round_medians_us": rounds,
                                  "median_of_round_medians_us": statistics.median(rounds)}
    for name in ARMS:
        if (run["profiles"][name]["status"] != "ok" or
                run["profiles"][name]["cubin_sha256"] != paired["compiler"][name]["cubin_sha256"]):
            raise ValueError(f"incomplete or changed profile: {name}")
        output["profiles"][name] = {
            "cubin_sha256": paired["compiler"][name]["cubin_sha256"],
            "compiler": paired["compiler"][name],
            "metrics": raw_metrics(HERE / "output" / f"{name}.raw.csv"),
            "sass": sass_samples(HERE / "output" / f"{name}.sass.csv"),
        }
    output["paired_candidate_speedup"] = [
        control / candidate for control, candidate in zip(
            output["timing"]["control"]["round_medians_us"],
            output["timing"]["candidate"]["round_medians_us"], strict=True)]
    return output


if __name__ == "__main__":
    (HERE / "analysis.json").write_text(json.dumps(analyze(), indent=2, sort_keys=True) + "\n")
