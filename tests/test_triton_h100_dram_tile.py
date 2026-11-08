"""Check the tile-orientation profile analysis rejects incomplete evidence."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


PATH = (Path(__file__).resolve().parents[1]
        / "docs/experiments/triton-h100-dram-tile/analyze.py")
SPEC = importlib.util.spec_from_file_location("dram_tile_analysis", PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def raw_csv() -> str:
    header = ["ID", "Kernel Name", *module.METRICS]
    values = ["1", "gemm", "150", "5", "20", "10", "3", "70", "512", "160", "148"]
    return (",".join(header) + "\n" + ",".join(["unit"] * len(header))
            + "\n" + ",".join(values) + "\n")


def report() -> dict:
    observations = [{"round": index, "implementation": name,
                     "timings_us": [200.0] * 30, "median_us": 200.0}
                    for index in range(3) for name in module.NAMES]
    return {"complete": True, "source_sha256": "abc",
            "benchmark": {"status": "ok", "source_sha256": "abc",
                          "correctness": {}, "observations": observations},
            "profiles": {name: {"ncu_report_sha256": "hash"} for name in module.NAMES}}


def test_complete_tile_profiles() -> None:
    analysis = module.build_analysis(report(), {name: raw_csv() for name in module.NAMES})
    assert analysis["profiles"]["tile_256x128_g1"]["metrics"]["dram__bytes_read.sum"] == 150


def test_missing_tile_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="missing or unexpected"):
        module.build_analysis(report(), {name: raw_csv() for name in module.NAMES[:-1]})
