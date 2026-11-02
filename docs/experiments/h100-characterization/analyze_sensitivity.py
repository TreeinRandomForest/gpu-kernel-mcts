"""Validate read-path sensitivity sweeps and retain cross-sweep spread."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

from kernel_mcts.cuda_backend import _parse_ncu_csv


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"
CONFIGS = {(size, grid, pattern) for size in (8 << 20, 256 << 20)
           for grid in (132, 528, 2112)
           for pattern in ("contiguous", "stride_32_words")}
PROFILES = ("sensitivity_l2_contiguous_dram", "sensitivity_l2_strided_dram",
            "sensitivity_hbm_contiguous_dram", "sensitivity_hbm_strided_dram")
METRIC = "dram__bytes_read.sum"


def build_analysis(timings: dict, output_dir: Path) -> dict:
    if timings.get("schema_version") != 1 or timings.get("compute_capability") != "9.0":
        raise ValueError("unsupported sensitivity schema or GPU")
    sweeps = timings.get("sweeps")
    if not isinstance(sweeps, list) or len(sweeps) != 3:
        raise ValueError("expected three sensitivity sweeps")
    by_config = {key: [] for key in CONFIGS}
    for expected_index, sweep in enumerate(sweeps):
        if sweep["index"] != expected_index or len(sweep["cases"]) != len(CONFIGS):
            raise ValueError("invalid sensitivity sweep")
        seen = set()
        for item in sweep["cases"]:
            key = (item["payload_bytes"], item["grid_programs"], item["access_pattern"])
            if key not in CONFIGS or key in seen:
                raise ValueError("unexpected or duplicate sensitivity case")
            seen.add(key)
            samples = item["raw_us_per_launch"]
            if len(samples) != item["rounds"] or any(
                not isinstance(value, (float, int)) or not math.isfinite(value)
                or value <= 0 for value in samples
            ) or not math.isclose(item["median_us_per_launch"], statistics.median(samples),
                                  rel_tol=1e-9):
                raise ValueError("invalid sensitivity timing samples")
            if item["working_set_bytes"] != key[0] \
                    or item["requested_read_bytes_per_launch"] != key[0] \
                    or item["physical_link_attribution"] != "unverified":
                raise ValueError("sensitivity case overclaims bytes or physical link")
            if not any("ld.global.cg" in line for line in item["ptx_global_load_lines"]):
                raise ValueError("PTX does not verify .cg load")
            by_config[key].append(item)
    summary = []
    for key in sorted(CONFIGS):
        medians = [item["median_us_per_launch"] for item in by_config[key]]
        summary.append({
            "payload_bytes": key[0], "grid_programs": key[1],
            "access_pattern": key[2], "sweep_medians_us": medians,
            "median_of_sweep_medians_us": statistics.median(medians),
            "min_sweep_median_us": min(medians),
            "max_sweep_median_us": max(medians),
        })
    profiles = {}
    for label in PROFILES:
        path = output_dir / f"{label}.ncu.csv"
        if path.is_file():
            profiles[label] = {"status": "measured",
                               "metric": _parse_ncu_csv(path.read_text(), (METRIC,))[METRIC],
                               "scope": "separate single-metric NCU application replay"}
        else:
            profiles[label] = {"status": "unavailable"}
    return {
        "schema_version": 1,
        "device_name": timings["device_name"],
        "case_summary": summary,
        "gpu_state_by_sweep": [{"before": sweep["gpu_state_before"],
                                "after": sweep["gpu_state_after"]} for sweep in sweeps],
        "profile_checks": profiles,
        "physical_link_calibrations": {},
        "limitations": (
            "Grid count changes work per program and occupancy; access pattern changes "
            "coalescing and memory traffic. Timings include reduction and launch. "
            "These are sensitivity curves for end-to-end read paths, not physical "
            "link rates. NCU profiles are separate from timing."
        ),
    }


def main() -> None:
    timings = json.loads((OUTPUT / "sensitivity_timings.json").read_text())
    result = build_analysis(timings, OUTPUT)
    (OUTPUT / "sensitivity_analysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    for item in result["case_summary"]:
        print(item["payload_bytes"], item["grid_programs"], item["access_pattern"],
              round(item["median_of_sweep_medians_us"], 3),
              round(item["min_sweep_median_us"], 3),
              round(item["max_sweep_median_us"], 3))
    for label, profile in result["profile_checks"].items():
        print(label, profile["status"], profile.get("metric"))


if __name__ == "__main__":
    main()
