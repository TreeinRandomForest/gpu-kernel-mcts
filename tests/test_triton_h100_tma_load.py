"""Validate the standalone TMA comparison's analysis gates without a GPU."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


PATH = (Path(__file__).resolve().parents[1]
        / "docs/experiments/triton-h100-tma-load/analyze.py")
SPEC = importlib.util.spec_from_file_location("triton_tma_analysis", PATH)
assert SPEC is not None and SPEC.loader is not None
analysis_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis_module)


def sass_csv(opcode: str) -> str:
    return ("Address,Source,Instructions Executed\n"
            f"0x10,{opcode}.E,120\n")


def raw_csv() -> str:
    return ("ID,Kernel Name,smsp__inst_executed.sum,launch__registers_per_thread\n"
            "unit,text,inst,registers\n"
            "1,gemm,120,64\n")


def report() -> dict:
    observations = []
    for round_index in range(3):
        for name, value in (("baseline", 200.0), ("tma", 190.0), ("cublas", 180.0)):
            observations.append({"round": round_index, "implementation": name,
                                 "timings_us": [value] * 30, "median_us": value})
    return {"complete": True, "source_sha256": {"baseline": "a", "tma": "b"},
            "benchmark": {"status": "ok", "source_sha256": {"baseline": "a", "tma": "b"},
                          "correctness": {"tma": {"exact_mismatches_vs_baseline": 0}},
                          "observations": observations},
            "profiles": {"baseline": {"ncu_report_sha256": "c"},
                         "tma": {"ncu_report_sha256": "d"}}}


def test_valid_tma_profile_and_paired_timing() -> None:
    result = analysis_module.build_analysis(
        report(), {"baseline": sass_csv("LDGSTS"), "tma": sass_csv("UTMALDG")},
        {"baseline": raw_csv(), "tma": raw_csv()},
    )
    assert result["median_tma_minus_baseline_us"] == -10.0
    assert result["profiles"]["tma"]["sass"]["opcode_executions"]["UTMALDG"] == 120


def test_rejects_candidate_without_executed_tma() -> None:
    with pytest.raises(ValueError, match="no executed TMA loads"):
        analysis_module.build_analysis(
            report(), {"baseline": sass_csv("LDGSTS"), "tma": sass_csv("LDGSTS")},
            {"baseline": raw_csv(), "tma": raw_csv()},
        )


def test_rejects_missing_round() -> None:
    data = report()
    data["benchmark"]["observations"].pop()
    with pytest.raises(ValueError, match="missing timing round"):
        analysis_module.build_analysis(
            data, {"baseline": sass_csv("LDGSTS"), "tma": sass_csv("UTMALDG")},
            {"baseline": raw_csv(), "tma": raw_csv()},
        )
