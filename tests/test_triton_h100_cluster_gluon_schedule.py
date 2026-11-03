"""CPU checks for the isolated Gluon request-order experiment."""

import importlib.util
from pathlib import Path


HERE = (Path(__file__).resolve().parents[1] /
        "docs/experiments/triton-h100-cluster-gluon-schedule")


def test_only_request_order_changes_between_ordinary_arms():
    spec = importlib.util.spec_from_file_location("gluon_schedule_paired", HERE / "paired.py")
    assert spec is not None and spec.loader is not None
    paired = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(paired)
    assert paired.ARMS["ordinary_a_first"][:3] == paired.ARMS["ordinary_b_first"][:3]
    assert paired.ARMS["ordinary_a_first"][3] is False
    assert paired.ARMS["ordinary_b_first"][3] is True
    assert paired.ARMS["multicast_a_first"] == (True, True, True, False)
