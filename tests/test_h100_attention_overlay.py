"""The attention validator preserves provenance and static/measurement scope."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parents[1]
EXPERIMENT = HERE / "docs/experiments/h100-resource-overlay"


def _analyzer():
    spec = importlib.util.spec_from_file_location(
        "h100_attention_analysis", EXPERIMENT / "analyze_attention.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inputs():
    ptx = (b".version 8.7\n.target sm_90\n"
           b".visible .entry causal_attention_probe(.param .u64 p0) {\n"
           b"ld.param.u64 %rd1, [p0];\nld.global.f32 %f1, [%rd1];\n"
           b"ex2.approx.f32 %f2, %f1;\n"
           b"st.global.f32 [%rd1], %f2;\nbra DONE;\nDONE:\nret;\n}\n")
    manifest = {
        "manifest_id": "test", "gpu_model": "NVIDIA H100 80GB HBM3",
        "form_factor": "SXM", "compute_capability": "9.0", "gpu_count": 1,
        "mig_configuration": {},
    }
    result = {
        "schema_version": 1, "kernel": "causal_attention_probe",
        "shape": {"sequence": 256, "head_dimension": 64},
        "correctness": {"max_absolute_error": 0},
        "compiler_resources": {"registers_per_thread": 64},
        "ptx_bytes": len(ptx),
    }
    provenance = {
        "experiment": "real_triton_attention_overlay_v1",
        "source_sha256": hashlib.sha256((EXPERIMENT / "attention_inside.py").read_bytes()).hexdigest(),
        "ptx_sha256": hashlib.sha256(ptx).hexdigest(),
        "worker_manifest": manifest,
    }
    return result, provenance, ptx


def test_attention_analysis_keeps_unmeasured_links_unknown():
    report, overlay = _analyzer().build_analysis(*_inputs())
    assert report["ptx_status"] == "partial"
    assert report["physical_link_calibrations"] == {}
    assert report["static_operation_counts"]["scalar_or_vector_compute"] >= 1
    assert report["ncu"]["status"] == "unavailable"
    assert overlay is not None
    assert overlay["whole_kernel_evidence"]["scope"] == "whole_kernel"


def test_attention_analysis_rejects_stale_source_identity():
    result, provenance, ptx = _inputs()
    provenance["source_sha256"] = "stale"
    with pytest.raises(ValueError, match="hash mismatch"):
        _analyzer().build_analysis(result, provenance, ptx)


def test_attention_analysis_flags_out_of_range_ncu_hit_rate():
    analyzer = _analyzer()
    rows = ['"Metric Name","Metric Unit","Metric Value"']
    for name in analyzer.METRICS:
        value = "101.2" if name == "lts__t_sector_hit_rate.pct" else "0"
        rows.append(f'"{name}","%","{value}"')
    report, _ = analyzer.build_analysis(*_inputs(), ncu_csv="\n".join(rows))
    assert report["ncu"]["counter_warnings"] == ["l2_hit_rate_outside_0_100"]
    assert report["ncu"]["metrics"]["lts__t_sector_hit_rate.pct"]["valid"] is False
    assert report["physical_link_calibrations"] == {}
