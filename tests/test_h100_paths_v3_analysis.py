"""Protect matched-cache and on-chip probe attribution/byte accounting."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ANALYZER_PATH = (Path(__file__).resolve().parents[1]
                 / "docs/experiments/h100-characterization/analyze_paths_v3.py")


def _analyzer():
    spec = importlib.util.spec_from_file_location("h100_paths_v3_analysis", ANALYZER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inputs() -> tuple[dict, dict]:
    read_config = [(8 << 20, n, "cyclic") for n in (1, 4, 16, 64)]
    read_config += [(256 << 20, n, "disjoint") for n in (1, 4)]
    read = {
        "schema_version": 1, "compute_capability": "9.0",
        "device_name": "NVIDIA H100",
        "observations": [{
            "payload_bytes": size, "repeats": repeats, "working_set_policy": policy,
            "working_set_bytes": size * (repeats if policy == "disjoint" else 1),
            "requested_read_bytes_per_launch": size * repeats,
            "rounds": 3, "raw_us_per_launch": [1.0, 1.2, 1.1],
            "median_us_per_launch": 1.1,
            "ptx_global_load_lines": ["ld.global.cg.v4.b32"],
            "physical_link_attribution": "unverified",
        } for size, repeats, policy in read_config],
    }
    blocks, threads = 528, 256
    onchip = {
        "schema_version": 1, "compute_capability": "9.0",
        "device_name": "NVIDIA H100", "blocks": blocks,
        "threads_per_block": threads, "rounds": 3,
        "cases": [{
            "kind": kind, "repeats": repeats,
            "requested_shared_read_bytes_per_launch": (
                blocks * threads * 4 * repeats if kind == "shared" else 0
            ),
            "raw_us_per_launch": [1.0, 1.2, 1.1],
            "median_us_per_launch": 1.1,
            "physical_link_attribution": "unverified",
        } for kind in ("shared", "register") for repeats in (1, 16, 64, 256, 1024)],
    }
    return read, onchip


def test_analysis_keeps_physical_links_unknown(tmp_path):
    result = _analyzer().build_analysis(*_inputs(), tmp_path)
    assert result["physical_link_calibrations"] == {}
    assert len(result["matched_read_cases"]) == 6
    assert len(result["onchip_cases"]) == 10
    shared = next(item for item in result["onchip_cases"]
                  if item["kind"] == "shared" and item["repeats"] == 1024)
    assert shared["effective_requested_shared_gb_per_s"] > 0


def test_analysis_rejects_fake_disjoint_working_set(tmp_path):
    read, onchip = _inputs()
    read["observations"][-1]["working_set_bytes"] = 256 << 20
    with pytest.raises(ValueError, match="incorrect working-set"):
        _analyzer().build_analysis(read, onchip, tmp_path)


def test_sass_check_requires_shared_loads_only_in_shared_probe():
    summarize = _analyzer().summarize_sass
    dump = "Function : _Z22register_control_probePji\nSTG.E\n" \
           "Function : _Z17shared_load_probePji\nSTS\nLDS\nSTG.E\n"
    result = summarize(dump)
    assert result["shared"]["LDS"] == 1
    assert result["register"]["LDS"] == 0
    with pytest.raises(ValueError, match="does not support"):
        summarize(dump.replace("LDS\n", ""))
