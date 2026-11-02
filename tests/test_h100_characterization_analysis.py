"""Guard the standalone H100 analysis against fabricated link calibration."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ANALYZER_PATH = (Path(__file__).resolve().parents[1]
                 / "docs/experiments/h100-characterization/analyze.py")


def _analyzer():
    spec = importlib.util.spec_from_file_location("h100_characterization_analysis", ANALYZER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _timings() -> dict:
    observations = []
    for size in range(1, 12):
        observations.append({
            "payload_bytes": size * 4096,
            "working_set_policy": "repeat_same_buffer",
            "raw_us_per_launch": [1.0, 1.2, 1.1],
            "rounds": 3,
            "median_us_per_launch": 1.1,
            "measurement_scope": "end_to_end_path",
            "cache_level_attribution": "unverified",
        })
    return {
        "schema_version": 1,
        "compute_capability": "9.0",
        "device_name": "NVIDIA H100",
        "timing_method": "test event timing",
        "launch_floor": {},
        "observations": observations,
    }


def test_analysis_preserves_path_scope_without_ncu(tmp_path):
    result = _analyzer().build_analysis(_timings(), tmp_path)
    assert result["physical_link_calibrations"] == {}
    assert len(result["size_response_bytes"]) == 11
    assert result["profile_checks"]["repeat_8m"]["status"] == "unavailable"


def test_analysis_rejects_link_attribution_without_isolation(tmp_path):
    timings = _timings()
    timings["observations"][0]["cache_level_attribution"] = "l2_only"
    with pytest.raises(ValueError, match="overclaims physical-link attribution"):
        _analyzer().build_analysis(timings, tmp_path)
