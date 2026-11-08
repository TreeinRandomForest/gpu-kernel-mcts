from __future__ import annotations

import io
import tarfile

from kernel_mcts.nebius_overlay import package_bundle


def test_package_bundle_contains_only_package_tree(tmp_path) -> None:
    package = tmp_path / "src/kernel_mcts"
    package.mkdir(parents=True)
    (package / "probe.py").write_text("ok = True\n")
    cache = package / "__pycache__"
    cache.mkdir()
    (cache / "probe.pyc").write_bytes(b"cached")
    bundle = package_bundle(tmp_path)
    with tarfile.open(fileobj=io.BytesIO(bundle), mode="r:gz") as archive:
        names = archive.getnames()
        assert "kernel_mcts/probe.py" in names
        assert not any("__pycache__" in name for name in names)
