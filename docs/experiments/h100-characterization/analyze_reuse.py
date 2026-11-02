"""Validate in-kernel repeated-read measurements without overclaiming links."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

from kernel_mcts.cuda_backend import _parse_ncu_csv


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"
CONFIGS = {(8 << 20, n) for n in (1, 4, 16, 64)}
CONFIGS.update({(256 << 20, n) for n in (1, 4)})
PROFILES = {"reuse_cg_8m_x64": (8 << 20, 64),
            "reuse_cg_256m_x4": (256 << 20, 4)}
METRICS = ("dram__bytes_read.sum", "dram__bytes_write.sum",
           "lts__t_sector_hit_rate.pct")


def counter_warnings(metrics: dict) -> list[str]:
    hit_pct = metrics["lts__t_sector_hit_rate.pct"]["value"]
    return ["l2_hit_rate_outside_0_100"] if not 0 <= hit_pct <= 100 else []


def build_analysis(timings: dict, output_dir: Path) -> dict:
    if timings.get("schema_version") != 1 or timings.get("compute_capability") != "9.0":
        raise ValueError("unsupported measurement schema or GPU")
    observations = timings.get("observations")
    if not isinstance(observations, list) or len(observations) != len(CONFIGS):
        raise ValueError("expected six in-kernel reuse cases")
    cases = {}
    for item in observations:
        key = (item["payload_bytes"], item["repeats"])
        if key not in CONFIGS or key in cases:
            raise ValueError("unexpected or duplicate reuse case")
        samples = item["raw_us_per_launch"]
        if len(samples) != item["rounds"] or any(
            not isinstance(sample, (float, int)) or not math.isfinite(sample)
            or sample <= 0 for sample in samples
        ) or not math.isclose(item["median_us_per_launch"], statistics.median(samples),
                              rel_tol=1e-9):
            raise ValueError("invalid reuse timing samples")
        if item["measurement_scope"] != "read_path_plus_address_arithmetic_reduction_and_output" \
                or item["physical_link_attribution"] != "unverified":
            raise ValueError("reuse result overclaims physical-link attribution")
        if item["requested_read_bytes_per_launch"] != key[0] * key[1]:
            raise ValueError("incorrect requested byte count")
        if not any("ld.global.cg" in line for line in item["ptx_global_load_lines"]):
            raise ValueError("PTX does not verify .cg global load")
        cases[key] = item
    profiles = {}
    for label, (size, repeats) in PROFILES.items():
        path = output_dir / f"{label}.ncu.csv"
        if not path.is_file():
            profiles[label] = {"status": "unavailable"}
            continue
        metrics = _parse_ncu_csv(path.read_text(), METRICS)
        profiles[label] = {
            "status": "measured",
            "metrics": metrics,
            "dram_read_fraction_of_requested_bytes": (
                metrics["dram__bytes_read.sum"]["value"] / (size * repeats)
            ),
            "counter_warnings": counter_warnings(metrics),
            "scope": "separate application-replay NCU run",
            "cache_control": "none",
        }
    return {
        "schema_version": 1,
        "device_name": timings["device_name"],
        "timing_method": timings["timing_method"],
        "cases": [cases[key] for key in sorted(cases)],
        "profile_checks": profiles,
        "physical_link_calibrations": {},
        "limitations": (
            "Changing repeat count changes both L2/HBM loads and loop/address "
            "arithmetic. Timing includes final reduction, output, and launch. "
            "Counter runs are separate application replays; an out-of-range "
            "L2 hit percentage flags inconsistent replay passes. No single "
            "physical link bandwidth or latency is inferred."
        ),
    }


def main() -> None:
    timings = json.loads((OUTPUT / "reuse_timings.json").read_text())
    result = build_analysis(timings, OUTPUT)
    (OUTPUT / "reuse_analysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    for item in result["cases"]:
        print(item["payload_bytes"], item["repeats"],
              round(item["median_us_per_launch"], 3),
              round(item["effective_requested_read_gb_per_s"], 1))
    for label, profile in result["profile_checks"].items():
        print(label, profile["status"],
              profile.get("dram_read_fraction_of_requested_bytes"))


if __name__ == "__main__":
    main()
