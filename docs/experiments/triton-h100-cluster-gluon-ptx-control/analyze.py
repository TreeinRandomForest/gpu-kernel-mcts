"""Audit the pinned PTX delta and the unsuccessful runtime attempts."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SCHEDULE = ROOT / "docs/experiments/triton-h100-cluster-gluon-schedule"
OUTPUT = SCHEDULE / "output"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze() -> dict:
    original = OUTPUT / "multicast_original_v87.ptx"
    patched = OUTPUT / "ordinary_patched_from_multicast_v87.ptx"
    original_sass = OUTPUT / "multicast_original_v87.sass"
    patched_sass = OUTPUT / "ordinary_patched_from_multicast_v87.sass"
    original_cubin = OUTPUT / "multicast_original_v87.cubin"
    patched_cubin = OUTPUT / "ordinary_patched_from_multicast_v87.cubin"
    spec = importlib.util.spec_from_file_location("gluon_ptx_patcher", SCHEDULE / "patch_ptx_control.py")
    assert spec is not None and spec.loader is not None
    patcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patcher)
    if patcher.patch(original.read_text()) != patched.read_text():
        raise ValueError("patched PTX differs from pinned one-instruction edit")
    attempts = {name: json.loads((HERE / f"{name}.json").read_text())
                for name in ("grid_failed", "runtime_hung")}
    if attempts["runtime_hung"]["patched_cubin_sha256"] != sha256(patched_cubin):
        raise ValueError("runtime attempt used another patched cubin")
    if attempts["runtime_hung"]["original_cubin_sha256"] != sha256(original_cubin):
        raise ValueError("runtime attempt used another original cubin")
    original_text = original_sass.read_text()
    patched_text = patched_sass.read_text()
    return {"schema_version": 1,
            "original_ptx_sha256": sha256(original),
            "patched_ptx_sha256": sha256(patched),
            "original_cubin_sha256": sha256(original_cubin),
            "patched_cubin_sha256": sha256(patched_cubin),
            "original_sass": {
                "warpgroup_wait_sites": original_text.count("WARPGROUP.DEPBAR.LE"),
                "multicast_tma_sites": original_text.count("UTMALDG.2D.MULTICAST"),
                "hgmma_sites": original_text.count("HGMMA.")},
            "patched_sass": {
                "warpgroup_wait_sites": patched_text.count("WARPGROUP.DEPBAR.LE"),
                "multicast_tma_sites": patched_text.count("UTMALDG.2D.MULTICAST"),
                "ordinary_tma_sites": patched_text.count("UTMALDG.2D"),
                "hgmma_sites": patched_text.count("HGMMA.")},
            "runtime_status": "patched arm did not return; terminated after over one minute",
            "runtime_correctness_established": False,
            "runtime_latency_established": False,
            "attempt_ids": {name: row["run_id"] for name, row in attempts.items()}}


if __name__ == "__main__":
    (HERE / "analysis.json").write_text(json.dumps(analyze(), indent=2, sort_keys=True) + "\n")
