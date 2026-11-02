"""Ensure dependent-load timing stays labeled as path latency."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ANALYZER_PATH = (Path(__file__).resolve().parents[1]
                 / "docs/experiments/h100-characterization/analyze_latency.py")


def _analyzer():
    spec = importlib.util.spec_from_file_location("h100_latency_analysis", ANALYZER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _timings() -> dict:
    analyzer = _analyzer()
    return {
        "schema_version": 1, "compute_capability": "9.0",
        "sector_stride_bytes": 128, "device_name": "NVIDIA H100",
        "timer": "clock64 and globaltimer", "rounds": 3,
        "cases": [{
            "label": label, "kind": kind, "working_set_bytes": size, "steps": steps,
            "raw_cycles_per_hop": [20.0, 21.0, 22.0],
            "raw_nanoseconds_per_hop": [10.0, 11.0, 12.0],
            "median_cycles_per_hop": 21.0, "median_nanoseconds_per_hop": 11.0,
            "measurement_scope": "dependent_read_chain",
            "physical_link_attribution": "unverified",
        } for label, (kind, size, steps) in analyzer.CASES.items()],
    }


def test_latency_analysis_keeps_links_unknown(tmp_path):
    result = _analyzer().build_analysis(_timings(), tmp_path)
    assert len(result["cases"]) == 9
    assert result["physical_link_calibrations"] == {}


def test_latency_analysis_rejects_wrong_scope(tmp_path):
    timings = _timings()
    timings["cases"][0]["physical_link_attribution"] = "l1_registers"
    with pytest.raises(ValueError, match="overclaims"):
        _analyzer().build_analysis(timings, tmp_path)


def test_sass_checks_distinguish_global_cache_policy_and_shared_load():
    dump = (
        "Function : global_chase_ca\nLDG.E.STRONG.SM\n"
        "Function : global_chase_cg\nLDG.E.STRONG.GPU\n"
        "Function : shared_chase_probe\nLDS\n"
        "Function : register_chase_control\nIADD3\n"
    )
    summarize = _analyzer().summarize_sass
    result = summarize(dump)
    assert result["ca"]["LDG_STRONG_SM"] == 1
    assert result["cg"]["LDG_STRONG_GPU"] == 1
    with pytest.raises(ValueError, match="does not support"):
        summarize(dump.replace("LDG.E.STRONG.GPU", "LDG.E.STRONG.SM"))
