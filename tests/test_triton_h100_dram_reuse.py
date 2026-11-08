"""GPU-free validation of the CTA-order traffic experiment's analysis."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


PATH = (Path(__file__).resolve().parents[1]
        / "docs/experiments/triton-h100-dram-reuse/analyze.py")
SPEC = importlib.util.spec_from_file_location("dram_reuse_analysis", PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def raw_csv(misses: int) -> str:
    metrics = module.METRICS
    header = ["ID", "Kernel Name", *metrics]
    values = ["1", "gemm", "150", "5", "20", "10", str(misses), "70"]
    return ",".join(header) + "\n" + ",".join(["unit"] * len(header)) + "\n" + ",".join(values) + "\n"


def report() -> dict:
    observations = []
    for index in range(3):
        for name in module.NAMES:
            observations.append({"round": index, "implementation": name,
                                 "timings_us": [200.0] * 30, "median_us": 200.0})
    return {"complete": True, "source_sha256": "abc",
            "benchmark": {"status": "ok", "source_sha256": "abc",
                          "correctness": {}, "observations": observations},
            "profiles": {f"{mode}-{name}": {"ncu_report_sha256": "hash"}
                         for mode in module.CACHE_MODES for name in module.NAMES}}


def raw() -> dict[str, str]:
    return {f"{mode}-{name}": raw_csv(3)
            for mode in module.CACHE_MODES for name in module.NAMES}


def test_complete_traffic_comparison() -> None:
    analysis = module.build_analysis(report(), raw())
    assert analysis["profiles"]["all-group_8"]["metrics"][
        "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum"] == 3
    assert len(analysis["profiles"]) == 8


def test_missing_profile_is_rejected() -> None:
    profiles = raw()
    profiles.pop("none-cublas")
    with pytest.raises(ValueError, match="missing or unexpected"):
        module.build_analysis(report(), profiles)


def test_duplicate_timing_is_rejected() -> None:
    data = report()
    data["benchmark"]["observations"].append(
        data["benchmark"]["observations"][0].copy())
    with pytest.raises(ValueError, match="duplicate"):
        module.build_analysis(data, raw())
