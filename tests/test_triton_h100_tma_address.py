"""GPU-free checks for isolated TMA arithmetic and store variants."""

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "docs/experiments/triton-h100-tma-address"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VARIANTS = load("tma_address_variants_test", HERE / "variants.py")
RUNNER = load("tma_address_runner_test", HERE / "run.py")


def test_variants_change_only_requested_source_sites():
    baseline = VARIANTS.BASELINE.read_text()
    variants = VARIANTS.sources(baseline)
    assert variants["baseline"] == baseline
    assert variants["unmasked_store"] == baseline.replace(
        VARIANTS.STORE, "tl.store(c_ptrs, c)")
    assert variants["constant_group"] == baseline.replace(
        VARIANTS.GROUP, "group_size_m = GROUP_M").replace(
        VARIANTS.GUARD, "if M % 128 or N % 256 or K % 64 or M % 1024:")
    assert variants["both"] == variants["constant_group"].replace(
        VARIANTS.STORE, "tl.store(c_ptrs, c)")
    assert all("def prepare" in source for source in variants.values())
    with pytest.raises(ValueError):
        VARIANTS.sources(baseline.replace(VARIANTS.STORE, ""))


def test_profile_targets_only_named_variants():
    command = RUNNER.profile_command("baseline", "/tmp/report", VARIANTS.NAMES)
    assert "--cache-control" in command and "--replay-mode" in command
    assert "TMA_ADDRESS_BASELINE/" in command
    with pytest.raises(ValueError):
        RUNNER.profile_command("cublas", "/tmp/report", VARIANTS.NAMES)
