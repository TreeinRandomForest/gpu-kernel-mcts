"""Guard the selected cuBLAS kernel and comparable NCU metric parsing."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


RUN = Path(__file__).resolve().parents[1] / "docs/experiments/triton-h100-cublas-profile/run.py"
ANALYZE = RUN.with_name("analyze.py")


def _run_module():
    spec = importlib.util.spec_from_file_location("triton_h100_cublas_profile_run", RUN)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _analysis_module():
    spec = importlib.util.spec_from_file_location("triton_h100_cublas_profile_analysis", ANALYZE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_vendor_ncu_command_profiles_first_launch_and_uses_same_workload():
    module = _run_module()
    command = module.cublas_profile_command(("metric_a", "metric_b"))
    assert command[:5] == ["ncu", "--csv", "--page", "raw", "--log-file"]
    assert command[command.index("--launch-count") + 1] == "1"
    assert command[command.index("--metrics") + 1] == "metric_a,metric_b"
    assert "--implementation=cublas" in command
    assert all(f"--{dimension}=4096" in command for dimension in ("M", "N", "K"))


def test_vendor_ncu_parser_requires_one_gemm_kernel():
    module = _run_module()
    metrics = ("metric_a", "metric_b")
    header = '"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n'
    rows = ('"1","sm90_gemm_bf16","metric_a","count","10"\n'
            '"1","sm90_gemm_bf16","metric_b","count","20"\n')
    result = module.parse_cublas_profile(header + rows, metrics)
    assert result["kernel_name"] == "sm90_gemm_bf16"
    assert result["metrics"]["metric_b"]["value"] == 20
    with pytest.raises(ValueError, match="expected one profiled kernel"):
        module.parse_cublas_profile(header + rows +
                                    '"2","other_kernel","metric_a","count","5"\n', metrics)


def test_vendor_ncu_parser_accepts_opaque_name_in_wide_csv():
    module = _run_module()
    metrics = ("metric_a", "metric_b")
    output = (
        '"ID","Process Name","Kernel Name","Block Size","Grid Size",'
        '"launch__shared_mem_per_block","metric_a","metric_b"\n'
        '"","","","","","byte","count","count"\n'
        '"0","bf16-gemm-vendor-baselines","nvjet_tst_opaque","(384, 1, 1)",'
        '"(2, 66, 1)","214212","10","20"\n'
    )
    result = module.parse_cublas_profile(output, metrics)
    assert result["kernel_name"] == "nvjet_tst_opaque"
    assert result["launch"]["Grid Size"] == "(2, 66, 1)"
    assert result["metrics"]["metric_a"]["value"] == 10


def test_analysis_requires_matching_metric_sets_and_paired_samples():
    analyzer = _analysis_module()
    summary = {key: 2.0 for key in analyzer.KEYS}
    observations = []
    for round_index in range(3):
        observations.extend([
            {"round": round_index, "implementation": "triton",
             "benchmark": {"median_us": 200.0, "timings_us": [200.0] * 30}},
            {"round": round_index, "implementation": "cublas",
             "median_us": 180.0, "timings_us": [180.0] * 30},
        ])
    run = {
        "complete": True, "triton_source_sha256": "source", "observations": observations,
        "triton_profile": {"metric_set": "diagnostic_v2", "resolved_metrics": ["x"],
                           "summary": summary},
        "cublas_profile": {"metric_set": "diagnostic_v2", "resolved_metrics": ["x"],
                           "summary": summary, "kernel_name": "gemm"},
    }
    assert analyzer.build_analysis(run)["median_paired_delta_us"] == 20.0
    run["cublas_profile"]["resolved_metrics"] = ["y"]
    with pytest.raises(ValueError, match="metric sets differ"):
        analyzer.build_analysis(run)
