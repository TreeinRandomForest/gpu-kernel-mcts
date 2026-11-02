"""Keep the repeated-read probe's byte accounting and link scope honest."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ANALYZER_PATH = (Path(__file__).resolve().parents[1]
                 / "docs/experiments/h100-characterization/analyze_reuse.py")


def _analyzer():
    spec = importlib.util.spec_from_file_location("h100_reuse_analysis", ANALYZER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _timings() -> dict:
    configs = [(8 << 20, n) for n in (1, 4, 16, 64)]
    configs += [(256 << 20, n) for n in (1, 4)]
    return {
        "schema_version": 1,
        "compute_capability": "9.0",
        "device_name": "NVIDIA H100",
        "timing_method": "test event timing",
        "observations": [{
            "payload_bytes": size,
            "repeats": repeats,
            "requested_read_bytes_per_launch": size * repeats,
            "raw_us_per_launch": [1.0, 1.2, 1.1],
            "rounds": 3,
            "median_us_per_launch": 1.1,
            "measurement_scope": "read_path_plus_address_arithmetic_reduction_and_output",
            "physical_link_attribution": "unverified",
            "ptx_global_load_lines": ["ld.global.cg.v4.b32"],
        } for size, repeats in configs],
    }


def test_reuse_analysis_preserves_unknown_links(tmp_path):
    result = _analyzer().build_analysis(_timings(), tmp_path)
    assert len(result["cases"]) == 6
    assert result["physical_link_calibrations"] == {}


def test_reuse_analysis_rejects_incorrect_byte_count(tmp_path):
    timings = _timings()
    timings["observations"][0]["requested_read_bytes_per_launch"] += 1
    with pytest.raises(ValueError, match="incorrect requested byte count"):
        _analyzer().build_analysis(timings, tmp_path)


def test_out_of_range_profiler_hit_rate_is_flagged():
    warning = _analyzer().counter_warnings
    assert warning({"lts__t_sector_hit_rate.pct": {"value": 101.51}}) == [
        "l2_hit_rate_outside_0_100"
    ]
