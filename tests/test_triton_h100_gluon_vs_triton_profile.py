"""Guard the selected launch in the three-arm H100 profiler."""

import importlib.util
from pathlib import Path


HERE = (Path(__file__).resolve().parents[1] /
        "docs/experiments/triton-h100-gluon-vs-triton-profile")


def load(name):
    spec = importlib.util.spec_from_file_location(f"gluon_vs_triton_{name}", HERE / name)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_each_arm_has_one_scoped_ncu_launch():
    run = load("run.py")
    profile = load("profile_one.py")
    assert run.ARMS == profile.ARMS
    for name in run.ARMS:
        command = run.profile_command(name)
        assert command[command.index("--nvtx-include") + 1] == (
            f"GLUON_VS_TRITON_{name.upper()}/")
        assert command[command.index("--launch-count") + 1] == "1"
        assert command[command.index("--cache-control") + 1] == "all"
