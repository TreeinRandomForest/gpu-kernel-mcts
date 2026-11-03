"""CPU checks for the isolated Gluon clustered GEMM launcher contract."""

import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/experiments/triton-h100-cluster-gluon/kernel.py"
SPEC = importlib.util.spec_from_file_location("cluster_gluon_kernel_test", SOURCE)
assert SPEC is not None and SPEC.loader is not None
KERNEL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(KERNEL)


def test_matched_cta_tile_contract():
    assert (KERNEL.CTA_M, KERNEL.CTA_N, KERNEL.BLOCK_K, KERNEL.WARPS) == (128, 256, 64, 8)


def test_launcher_rejects_incomplete_cluster_m_tile_before_gpu_work():
    a = torch.empty((128, 64), dtype=torch.bfloat16)
    b = torch.empty((256, 64), dtype=torch.bfloat16)
    c = torch.empty((128, 256), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="full tiles"):
        KERNEL.launch(a, b, c, clustered=True)


def test_launcher_rejects_noncontiguous_physical_b():
    a = torch.empty((256, 64), dtype=torch.bfloat16)
    b = torch.empty((64, 256), dtype=torch.bfloat16).T
    c = torch.empty((256, 256), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="physically contiguous"):
        KERNEL.launch(a, b, c, clustered=True)
