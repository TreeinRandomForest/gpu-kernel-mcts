"""Validate the matched tile latency analysis gates without a GPU."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


PATH = (Path(__file__).resolve().parents[1]
        / "docs/experiments/triton-h100-latency-tile/analyze.py")
SPEC = importlib.util.spec_from_file_location("latency_tile_analysis", PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def raw_csv() -> str:
    header = ["ID", "Kernel Name", *module.RAW_METRICS]
    values = ["1", "gemm", *(["1"] * len(module.RAW_METRICS))]
    return (",".join(header) + "\n" + ",".join(["unit"] * len(header))
            + "\n" + ",".join(values) + "\n")


def sass_csv() -> str:
    return "Address,Source,Instructions Executed\n0x10,UTMALDG.E,120\n"


def report() -> dict:
    observations = [{"round": index, "implementation": name,
                     "timings_us": [200.0 + offset] * 30,
                     "median_us": 200.0 + offset}
                    for index in range(3)
                    for name, offset in ((module.NAMES[0], 0.0), (module.NAMES[1], 10.0))]
    return {"complete": True, "source_sha256": "abc",
            "benchmark": {"status": "ok", "source_sha256": "abc",
                          "correctness": {}, "observations": observations},
            "profiles": {name: {"ncu_report_sha256": "hash"} for name in module.NAMES}}


def test_matched_tile_delta_and_sass() -> None:
    result = module.build_analysis(
        report(), {name: raw_csv() for name in module.NAMES},
        {name: sass_csv() for name in module.NAMES},
    )
    assert result["median_paired_delta_us"] == 10.0
    assert result["profiles"][module.NAMES[1]]["sass"]["opcode_executions"]["UTMALDG"] == 120


def test_missing_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="missing or unexpected"):
        module.build_analysis(report(), {module.NAMES[0]: raw_csv()},
                              {name: sass_csv() for name in module.NAMES})
