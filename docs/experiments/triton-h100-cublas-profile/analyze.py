"""Validate and compare paired timing rounds and matched NCU counters."""

from __future__ import annotations

import json
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results.json"
ANALYSIS = HERE / "analysis.json"
KEYS = (
    "achieved_occupancy_pct", "sm_throughput_pct", "tensor_pipe_utilization_pct",
    "dram_throughput_pct", "l2_throughput_pct", "l1tex_throughput_pct",
    "registers_per_thread", "instructions_executed", "tensor_instructions_executed",
    "dram_bytes_read", "dram_bytes_written", "l2_sector_hit_rate_pct",
    "eligible_warps_per_cycle", "issue_active_pct", "long_scoreboard_warps_per_issue",
    "short_scoreboard_warps_per_issue", "barrier_stalled_warps_per_issue",
    "math_pipe_throttle_warps_per_issue", "shared_bank_conflicts",
    "local_load_sectors", "local_store_sectors",
)


def build_analysis(run: dict) -> dict:
    if not run.get("complete"):
        raise ValueError("profiling run is incomplete")
    by_name = {"triton": {}, "cublas": {}}
    for item in run["observations"]:
        name, round_index = item["implementation"], item["round"]
        if name not in by_name or round_index in by_name[name]:
            raise ValueError("unexpected or duplicate timing observation")
        samples = (item["benchmark"]["timings_us"] if name == "triton"
                   else item["timings_us"])
        median = (item["benchmark"]["median_us"] if name == "triton"
                  else item["median_us"])
        if len(samples) != 30 or statistics.median(samples) != median:
            raise ValueError("timing median does not match 30 samples")
        by_name[name][round_index] = median
    if any(set(rounds) != {0, 1, 2} for rounds in by_name.values()):
        raise ValueError("three paired timing rounds are required")
    triton_profile = run["triton_profile"]
    cublas_profile = run["cublas_profile"]
    if (triton_profile["metric_set"] != "diagnostic_v2" or
            cublas_profile["metric_set"] != "diagnostic_v2" or
            triton_profile["resolved_metrics"] != cublas_profile["resolved_metrics"]):
        raise ValueError("NCU metric sets differ")
    counter_comparison = {}
    for key in KEYS:
        triton = triton_profile["summary"][key]
        cublas = cublas_profile["summary"][key]
        counter_comparison[key] = {
            "triton": triton, "cublas": cublas,
            "triton_minus_cublas": triton - cublas,
            "triton_over_cublas": triton / cublas if cublas else None,
        }
    triton_medians = [by_name["triton"][index] for index in range(3)]
    cublas_medians = [by_name["cublas"][index] for index in range(3)]
    deltas = [a - b for a, b in zip(triton_medians, cublas_medians)]
    return {
        "schema_version": 1, "status": "same_worker_matched_metric_comparison",
        "triton_source_sha256": run["triton_source_sha256"],
        "cublas_kernel_name": cublas_profile["kernel_name"],
        "cublas_launch": cublas_profile.get("launch", {}),
        "round_medians_us": {"triton": triton_medians, "cublas": cublas_medians},
        "median_of_round_medians_us": {
            "triton": statistics.median(triton_medians),
            "cublas": statistics.median(cublas_medians),
        },
        "paired_delta_us": deltas,
        "median_paired_delta_us": statistics.median(deltas),
        "counters": counter_comparison,
        "limitations": (
            "One diagnostic NCU application replay per implementation, separate from "
            "three CUDA-event timing rounds. Whole-kernel counters can be compared, "
            "but they do not identify the exact instruction-level cause of the latency gap. "
            "The Triton Python and cuBLAS C++ harnesses use different random-input "
            "distributions and setup paths. NCU replay duration is not reward timing."
        ),
    }


def main() -> None:
    analysis = build_analysis(json.loads(RESULTS.read_text()))
    ANALYSIS.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    print("Triton", analysis["round_medians_us"]["triton"])
    print("cuBLAS", analysis["round_medians_us"]["cublas"])
    print("paired gap", analysis["median_paired_delta_us"])
    for key in KEYS:
        values = analysis["counters"][key]
        print(key, values["triton"], values["cublas"])


if __name__ == "__main__":
    main()
