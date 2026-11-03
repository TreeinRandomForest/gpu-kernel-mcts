"""CPU checks for the pinned PTX replacement and separate module handle."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_patch_changes_only_elected_b_tma():
    patcher = load(ROOT / "docs/experiments/triton-h100-cluster-gluon-schedule/patch_ptx_control.py")
    source = "elect.sync \t%r120|%p20, -1;\n" + patcher.OLD + "\n"
    actual = patcher.patch(source)
    assert actual == "elect.sync \t%r120|%p20, -1;\n" + patcher.NEW + "\n"


def test_cloned_binary_has_separate_uninitialized_module():
    harness = load(ROOT / "docs/experiments/triton-h100-cluster-gluon-ptx-control/paired.py")
    original = SimpleNamespace(kernel=b"old", module=1, _module_pid=12,
                               function=2, _run=3)
    clone = harness.clone_with_cubin(original, b"new")
    assert original.kernel == b"old" and original.module == 1
    assert clone.kernel == b"new" and clone.module is None
    assert clone.function is None and clone._run is None
