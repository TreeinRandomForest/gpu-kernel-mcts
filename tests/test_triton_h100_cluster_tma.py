"""GPU-free checks for Triton cluster-lowering probe."""

import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parents[1] / "docs/experiments/triton-h100-cluster-tma"
SPEC = importlib.util.spec_from_file_location("cluster_tma_probe_test", HERE / "probe.py")
assert SPEC is not None and SPEC.loader is not None
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


def test_probe_configs_and_multicast_classifier():
    assert PROBE.CONFIGS["baseline"] == (128, 256, 1)
    assert all(config[2] == 2 for name, config in PROBE.CONFIGS.items()
               if name != "baseline")
    ptx = "cp.async.bulk.tensor.2d.shared::cluster.global.multicast::cluster\n"
    found = PROBE.classify_ptx(ptx)
    assert found["multicast_mentions"] == 1
    assert found["cluster_mentions"] == 1
    assert found["tma_load_mentions"] == 1
