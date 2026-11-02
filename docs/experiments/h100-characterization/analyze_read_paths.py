"""Validate read-only path probes and label cache regimes from NCU evidence."""

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
PROFILES = {"read_cg_8m": (8 << 20, ".cg", "kernel"),
            "read_ca_8m": (8 << 20, ".ca", "kernel"),
            "read_cg_256m": (256 << 20, ".cg", "kernel"),
            "read_cg_1m_app": (1 << 20, ".cg", "application"),
            "read_cg_8m_app": (8 << 20, ".cg", "application"),
            "read_ca_8m_app": (8 << 20, ".ca", "application"),
            "read_cg_256m_app": (256 << 20, ".cg", "application")}
EXPECTED_CASES = {(size, ".cg") for size in (1 << 20, 8 << 20,
                    32 << 20, 64 << 20, 256 << 20)}
EXPECTED_CASES.update({(8 << 20, ".ca"), (256 << 20, ".ca")})


def classify_cache_regime(payload_bytes: int, modifier: str,
                          metrics: dict) -> str:
    """Classify observed read source; never assign a physical-link bandwidth."""
    if modifier != ".cg":
        return "not_classified_l1_allowed"
    read_bytes = metrics["dram__bytes_read.sum"]["value"]
    hit_pct = metrics["lts__t_sector_hit_rate.pct"]["value"]
    if read_bytes <= 0.05 * payload_bytes and hit_pct >= 80:
        return "l2_resident_read_evidence"
    if read_bytes >= 0.8 * payload_bytes and payload_bytes >= 64 << 20:
        return "hbm_sourced_read_evidence"
    return "mixed_or_unresolved"


def build_analysis(timings: dict, output_dir: Path) -> dict:
    if timings.get("schema_version") != 1 or timings.get("compute_capability") != "9.0":
        raise ValueError("unsupported measurement schema or GPU")
    observations = timings.get("observations")
    if not isinstance(observations, list) or len(observations) != len(EXPECTED_CASES):
        raise ValueError("expected seven read-path cases")
    cases = {}
    for item in observations:
        key = (item["payload_bytes"], item["cache_modifier"])
        if key not in EXPECTED_CASES or key in cases:
            raise ValueError("unexpected or duplicate read-path case")
        samples = item["raw_us_per_launch"]
        if len(samples) != item["rounds"] or any(
            not isinstance(sample, (float, int)) or not math.isfinite(sample)
            or sample <= 0 for sample in samples
        ) or not math.isclose(item["median_us_per_launch"], statistics.median(samples),
                              rel_tol=1e-9):
            raise ValueError("invalid read-path timing samples")
        if item["measurement_scope"] != "read_path_plus_reduction_and_output" \
                or item["physical_link_attribution"] != "unverified":
            raise ValueError("read-path result overclaims physical-link attribution")
        if not any("ld.global" in line and key[1] in line
                   for line in item["ptx_global_load_lines"]):
            raise ValueError("PTX does not verify requested cache modifier")
        cases[key] = item
    profiles = {}
    for label, (size, modifier, replay_mode) in PROFILES.items():
        csv_path = output_dir / f"{label}.ncu.csv"
        if not csv_path.is_file():
            profiles[label] = {"status": "unavailable"}
            continue
        metrics = _parse_ncu_csv(csv_path.read_text(), METRICS)
        profiles[label] = {
            "status": "measured",
            "metrics": metrics,
            "dram_read_fraction_of_requested_bytes": (
                metrics["dram__bytes_read.sum"]["value"] / size
            ),
            "cache_regime_evidence": classify_cache_regime(size, modifier, metrics),
            "scope": "separate single-kernel NCU replay",
            "cache_control": "none",
            "clock_control": "none",
            "replay_mode": replay_mode,
        }
    return {
        "schema_version": 1,
        "device_name": timings["device_name"],
        "timing_method": timings["timing_method"],
        "cases": [cases[key] for key in sorted(cases)],
        "profile_checks": profiles,
        "physical_link_calibrations": {},
        "limitations": (
            "Explicit .cg requests L1 bypass, and NCU can support an L2-resident or "
            "HBM-sourced read regime. Timings also include reduction, output writes, "
            "and launch; no individual hardware link bandwidth is inferred. "
            "NCU replay is separate from CUDA Graph timing."
        ),
    }


def main() -> None:
    timings = json.loads((OUTPUT / "read_paths_timings.json").read_text())
    result = build_analysis(timings, OUTPUT)
    (OUTPUT / "read_paths_analysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    for item in result["cases"]:
        print(item["payload_bytes"], item["cache_modifier"],
              round(item["median_us_per_launch"], 3),
              round(item["effective_requested_read_gb_per_s"], 1))
    for label, profile in result["profile_checks"].items():
        print(label, profile["status"], profile.get("cache_regime_evidence"))


if __name__ == "__main__":
    main()
