"""CPU checks for the isolated Gluon B descriptor experiment."""

import importlib.util
from pathlib import Path


def test_ordinary_arms_only_change_b_descriptor_layout():
    path = (Path(__file__).resolve().parents[1] /
            "docs/experiments/triton-h100-cluster-gluon-local-b/paired.py")
    spec = importlib.util.spec_from_file_location("gluon_local_b_paired", path)
    assert spec is not None and spec.loader is not None
    paired = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(paired)
    assert paired.ARMS["ordinary_replicated"][:4] == paired.ARMS["ordinary_local"][:4]
    assert paired.ARMS["ordinary_replicated"][4] is False
    assert paired.ARMS["ordinary_local"][4] is True
    assert paired.ARMS["multicast_replicated"] == (True, True, True, False, False)
