"""Check that read-path evidence is attributed only after PTX and NCU checks."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ANALYZER_PATH = (Path(__file__).resolve().parents[1]
                 / "docs/experiments/h100-characterization/analyze_read_paths.py")


def _analyzer():
    spec = importlib.util.spec_from_file_location("h100_read_paths_analysis", ANALYZER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _timings() -> dict:
    sizes = (1 << 20, 8 << 20, 32 << 20, 64 << 20, 256 << 20)
    configs = [(size, ".cg") for size in sizes]
    configs += [(8 << 20, ".ca"), (256 << 20, ".ca")]
    return {
        "schema_version": 1,
        "compute_capability": "9.0",
        "device_name": "NVIDIA H100",
        "timing_method": "test event timing",
        "observations": [{
            "payload_bytes": size,
            "cache_modifier": modifier,
            "raw_us_per_launch": [1.0, 1.2, 1.1],
            "rounds": 3,
            "median_us_per_launch": 1.1,
            "measurement_scope": "read_path_plus_reduction_and_output",
            "physical_link_attribution": "unverified",
            "ptx_global_load_lines": [f"ld.global{modifier}.b32 %r1, [%rd1];"],
        } for size, modifier in configs],
    }


def test_read_path_analysis_requires_ptx_and_retains_unknown_links(tmp_path):
    analyzer = _analyzer()
    result = analyzer.build_analysis(_timings(), tmp_path)
    assert result["physical_link_calibrations"] == {}
    assert len(result["cases"]) == 7
    assert result["profile_checks"]["read_cg_8m"]["status"] == "unavailable"
    timings = _timings()
    timings["observations"][0]["ptx_global_load_lines"] = ["ld.global.ca.b32"]
    with pytest.raises(ValueError, match="PTX does not verify"):
        analyzer.build_analysis(timings, tmp_path)


def test_read_source_classification_requires_matching_traffic():
    classify = _analyzer().classify_cache_regime
    small = 8 << 20
    large = 256 << 20
    def metrics(dram: int, hit: float) -> dict:
        return {"dram__bytes_read.sum": {"value": dram},
                "lts__t_sector_hit_rate.pct": {"value": hit}}
    assert classify(small, ".cg", metrics(0, 99)) == "l2_resident_read_evidence"
    assert classify(small, ".cg", metrics(0, 87.5)) == "l2_resident_read_evidence"
    assert classify(small, ".cg", metrics(0, 60)) == "mixed_or_unresolved"
    assert classify(large, ".cg", metrics(large, 5)) == "hbm_sourced_read_evidence"
    assert classify(small, ".ca", metrics(0, 99)) == "not_classified_l1_allowed"
    assert classify(small, ".cg", metrics(small // 2, 70)) == "mixed_or_unresolved"
