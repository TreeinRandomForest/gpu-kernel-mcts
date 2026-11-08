"""CPU-only checks for the paired CuTe cluster artifact capture."""

import importlib.util
import json
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docs/experiments/triton-h100-cluster-cute/paired.py"
SPEC = importlib.util.spec_from_file_location("cluster_cute_paired_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_cluster_schedules_change_only_cluster_shape():
    single = MODULE.SCHEDULES["single"]
    clustered = MODULE.SCHEDULES["cluster_b_multicast"]
    assert (single.tile_m, single.tile_n) == (clustered.tile_m, clustered.tile_n) == (128, 256)
    assert (single.cluster_m, single.cluster_n) == (1, 1)
    assert (clustered.cluster_m, clustered.cluster_n) == (2, 1)


def test_embedded_binary_has_valid_fatbinary_length():
    record = json.loads((ROOT / "cutedsl-cluster-v58.json").read_text())
    ir = record["jit_diagnostics"]["mlir"]["normalized_text"]
    binaries = MODULE.embedded_binaries(ir)
    assert len(binaries) == 1
    assert len(binaries[0]) == 16 + struct.unpack_from("<Q", binaries[0], 8)[0]
    assert binaries[0].find(b"\x7fELF") >= 0
