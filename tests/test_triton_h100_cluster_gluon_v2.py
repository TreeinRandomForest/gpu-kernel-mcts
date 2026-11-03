"""CPU checks for the matched cluster/multicast/pipeline experiment."""

import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "docs/experiments/triton-h100-cluster-gluon-v2"


def load(name):
    spec = importlib.util.spec_from_file_location(f"cluster_gluon_v2_{name}", HERE / name)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


KERNEL = load("kernel.py")
PAIRED = load("paired.py")


def test_factorial_arms_keep_three_matched_cluster_choices_per_pipeline():
    assert len(PAIRED.ARMS) == 6
    assert set(PAIRED.ARMS.values()) == {
        (False, False, False), (True, False, False), (True, True, False),
        (False, False, True), (True, False, True), (True, True, True),
    }
    assert (KERNEL.CTA_M, KERNEL.CTA_N, KERNEL.BLOCK_K, KERNEL.WARPS) == (128, 256, 64, 8)


def test_multicast_requires_cluster_before_gpu_work():
    a = torch.empty((256, 64), dtype=torch.bfloat16)
    b = torch.empty((256, 64), dtype=torch.bfloat16)
    c = torch.empty((256, 256), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="requires a two-CTA cluster"):
        KERNEL.launch(a, b, c, clustered=False, multicast=True, pipelined=False)
