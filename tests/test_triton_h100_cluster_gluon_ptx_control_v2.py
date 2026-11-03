"""CPU check for the corrected one-issuer-per-CTA PTX edit."""

import importlib.util
from pathlib import Path


def test_patch_keeps_first_warp_guard():
    path = (Path(__file__).resolve().parents[1] /
            "docs/experiments/triton-h100-cluster-gluon-schedule/patch_ptx_control_v2.py")
    spec = importlib.util.spec_from_file_location("patch_ptx_v2", path)
    assert spec is not None and spec.loader is not None
    patcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patcher)
    source = patcher.OLD_PREDICATE + "\n" + patcher.OLD_LOAD
    actual = patcher.patch(source)
    assert patcher.NEW_PREDICATE in actual
    assert patcher.NEW_LOAD in actual
    assert patcher.OLD_PREDICATE not in actual
    assert patcher.OLD_LOAD not in actual
