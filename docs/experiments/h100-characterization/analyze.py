"""Validate the standalone H100 measurements and retain path-level curves."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

from kernel_mcts.cuda_backend import _parse_ncu_csv


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"
METRICS = ("dram__bytes_read.sum", "dram__bytes_write.sum",
           "lts__t_sector_hit_rate.pct")


def build_analysis(timings: dict, output_dir: Path) -> dict:
    if timings.get("schema_version") != 1 or timings.get("compute_capability") != "9.0":
        raise ValueError("unsupported measurement schema or GPU")
    observations = timings.get("observations")
    if not isinstance(observations, list) or len(observations) != 11:
        raise ValueError("expected eleven global-copy cases")
    by_size: dict[str, list[dict[str, object]]] = {}
    seen: set[tuple[int, str]] = set()
    for item in observations:
        size = item["payload_bytes"]
        policy = item["working_set_policy"]
        if type(size) is not int or size <= 0 or (size, policy) in seen:
            raise ValueError("invalid or duplicate copy case")
        seen.add((size, policy))
        samples = item["raw_us_per_launch"]
        if len(samples) != item["rounds"] or any(
            not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0
            for value in samples
        ):
            raise ValueError("invalid timing samples")
        if not math.isclose(item["median_us_per_launch"], statistics.median(samples),
                            rel_tol=1e-9):
            raise ValueError("incorrect median")
        if item["measurement_scope"] != "end_to_end_path" \
                or item["cache_level_attribution"] != "unverified":
            raise ValueError("copy-path result overclaims physical-link attribution")
        by_size.setdefault(str(size), []).append(item)
    profiles = {}
    for label in ("repeat_8m", "rotate_128m"):
        csv_path = output_dir / f"{label}.ncu.csv"
        if not csv_path.is_file():
            profiles[label] = {"status": "unavailable"}
            continue
        metrics = _parse_ncu_csv(csv_path.read_text(), METRICS)
        profiles[label] = {"status": "measured", "metrics": metrics,
                           "cache_control": "none", "clock_control": "none",
                           "scope": "separate single-kernel NCU replay"}
    return {
        "schema_version": 1,
        "device_name": timings["device_name"],
        "timing_method": timings["timing_method"],
        "launch_floor": timings["launch_floor"],
        "size_response_bytes": by_size,
        "profile_checks": profiles,
        "physical_link_calibrations": {},
        "limitations": (
            "Copy timings measure an end-to-end global-memory path, including launch, "
            "caches, and stores. NCU runs separately from timing and can perturb caches. "
            "No single HBM, L2, or L1 link bandwidth or latency is inferred here."
        ),
    }


def main() -> None:
    timings = json.loads((OUTPUT / "timings.json").read_text())
    result = build_analysis(timings, OUTPUT)
    (OUTPUT / "analysis.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    by_size = result["size_response_bytes"]
    for size in sorted(map(int, by_size)):
        for item in by_size[str(size)]:
            print(size, item["working_set_policy"],
                  round(item["median_us_per_launch"], 3),
                  round(item["effective_gb_per_s"], 1))
    for label, profile in result["profile_checks"].items():
        print(label, profile["status"])


if __name__ == "__main__":
    main()
