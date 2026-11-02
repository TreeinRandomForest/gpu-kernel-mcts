"""Ensure sensitivity sweeps preserve configuration and uncertainty."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ANALYZER_PATH = (Path(__file__).resolve().parents[1]
                 / "docs/experiments/h100-characterization/analyze_sensitivity.py")


def _analyzer():
    spec = importlib.util.spec_from_file_location("h100_sensitivity_analysis", ANALYZER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _timings() -> dict:
    analyzer = _analyzer()
    sweeps = []
    for index in range(3):
        sweeps.append({"index": index,
                       "gpu_state_before": {}, "gpu_state_after": {},
                       "cases": [{
                           "payload_bytes": size, "grid_programs": grid,
                           "access_pattern": pattern,
                           "working_set_bytes": size,
                           "requested_read_bytes_per_launch": size,
                           "raw_us_per_launch": [1.0, 1.1 + index, 1.2 + index],
                           "rounds": 3,
                           "median_us_per_launch": 1.1 + index,
                           "physical_link_attribution": "unverified",
                           "ptx_global_load_lines": ["ld.global.cg.v4.b32"],
                       } for size, grid, pattern in sorted(analyzer.CONFIGS)]})
    return {"schema_version": 1, "compute_capability": "9.0",
            "device_name": "NVIDIA H100", "sweeps": sweeps}


def test_sensitivity_analysis_keeps_spread_and_unknown_links(tmp_path):
    result = _analyzer().build_analysis(_timings(), tmp_path)
    assert len(result["case_summary"]) == 12
    assert result["case_summary"][0]["sweep_medians_us"] == [1.1, 2.1, 3.1]
    assert result["physical_link_calibrations"] == {}


def test_sensitivity_analysis_rejects_duplicate_case(tmp_path):
    timings = _timings()
    timings["sweeps"][0]["cases"][1] = timings["sweeps"][0]["cases"][0]
    with pytest.raises(ValueError, match="duplicate"):
        _analyzer().build_analysis(timings, tmp_path)
