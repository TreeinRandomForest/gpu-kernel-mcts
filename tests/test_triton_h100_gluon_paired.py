"""Check paired crossover aggregation before spending an H100 worker."""

from __future__ import annotations

import importlib.util
import ast
from pathlib import Path

import pytest


PAIRED = Path("docs/experiments/triton-h100-gluon-search-adapter/paired.py")


def paired_module():
    spec = importlib.util.spec_from_file_location("gluon_search_paired", PAIRED)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_paired_summary_uses_within_round_differences_and_requires_both_arms() -> None:
    paired = paired_module()
    observations = []
    for index in range(paired.ROUNDS):
        observations.extend((
            {"round": index, "variant": "single_cta",
             "benchmark": {"median_us": 200.0 + 10 * index}},
            {"round": index, "variant": "two_cta",
             "benchmark": {"median_us": 210.0 + 10 * index}},
        ))
    summary = paired.summarize(observations)
    assert summary["paired_delta_us"] == [10.0] * paired.ROUNDS
    assert summary["median_paired_delta_us"] == 10.0
    assert summary["contender_median_of_round_medians_us"] == 230.0
    with pytest.raises(ValueError, match="incomplete paired"):
        paired.summarize(observations[:-1])


def test_paired_summary_accepts_cublas_as_third_arm() -> None:
    paired = paired_module()
    observations = []
    for index in range(paired.ROUNDS):
        for name, latency in (("single_cta", 250), ("persistent", 210), ("cublas", 190)):
            observations.append({"round": index, "variant": name,
                                 "benchmark": {"median_us": latency + index}})
    assert paired.summarize(observations, "persistent")["median_paired_delta_us"] == -40
    assert paired.summarize(observations, "cublas")["median_paired_delta_us"] == -60


def test_persistent_search_adapter_preserves_factorial_device_kernel() -> None:
    root = Path("docs/experiments")
    original = ast.parse((root / "triton-h100-gluon-factorial/kernel.py").read_text())
    adapted = ast.parse((root / "triton-h100-gluon-search-adapter/"
                         "persistent_factorial_fixture.py").read_text())
    device = lambda tree: next(node for node in tree.body
                               if isinstance(node, ast.FunctionDef) and node.name == "gemm")
    assert ast.dump(device(original)) == ast.dump(device(adapted))
    launcher = next(node for node in adapted.body
                    if isinstance(node, ast.FunctionDef) and node.name == "launch")
    assert [arg.arg for arg in launcher.args.args] == ["A", "B", "C", "M", "N", "K"]


@pytest.mark.parametrize("warps,name", [(4, "Four"), (16, "Sixteen")])
def test_warp_probe_changes_only_warp_count_and_mma_layout(warps: int, name: str) -> None:
    root = Path("docs/experiments/triton-h100-gluon-search-adapter")
    base = (root / "single_cta_gluon_fixture.py").read_text()
    suffix = "four" if warps == 4 else "sixteen"
    changed = (root / f"single_cta_{suffix}_warp_fixture.py").read_text()
    expected = base.replace("One-CTA diagnostic variant of the validated Gluon wait-one kernel.",
                            f"{name}-warp one-CTA variant of the validated Gluon wait-one kernel.")
    expected = expected.replace("WARPS = 8", f"WARPS = {warps}")
    expected = expected.replace("warps_per_cta=[8, 1]", f"warps_per_cta=[{warps}, 1]")
    assert changed == expected


def test_mma_v2_small_tile_probe_changes_only_launch_tile() -> None:
    root = Path("docs/experiments/triton-h100-gluon-search-adapter")
    parent = (root / "v4-mma-v2-thread-layout-repair/candidate.py").read_text()
    child = (root / "v4_mma_v2_small_tile_fixture.py").read_text()
    assert child.rstrip() == parent.replace("BM, BN, BK = 64, 128, 64",
                                            "BM, BN, BK = 32, 64, 64").rstrip()


def test_mma_v2_coalesced_layout_probe_changes_only_memory_axis_orders() -> None:
    root = Path("docs/experiments/triton-h100-gluon-search-adapter")
    parent = (root / "v4-mma-v2-thread-layout-repair/candidate.py").read_text()
    child = (root / "v4_mma_v2_coalesced_layout_fixture.py").read_text()
    # Change A and B independently so replacing one cannot silently match the other.
    expected = parent.replace("gl.BlockedLayout([1, 16], [4, 8], [2, 2], [0, 1])",
                              "gl.BlockedLayout([1, 16], [4, 8], [2, 2], [1, 0])")
    expected = expected.replace("gl.BlockedLayout([16, 1], [8, 4], [2, 2], [1, 0])",
                                "gl.BlockedLayout([16, 1], [8, 4], [2, 2], [0, 1])")
    expected = expected.replace("gl.BlockedLayout([1, 8], [4, 8], [4, 1], [0, 1])",
                                "gl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])")
    assert child.rstrip() == expected.rstrip()
