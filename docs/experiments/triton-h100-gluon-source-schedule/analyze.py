"""Audit compiler artifacts, exact output, and paired CUDA Graph timing."""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path

from paired import ARMS


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CONTRASTS = {
    "source_ordinary_vs_static_ordinary": ("source_ordinary", "static_ordinary"),
    "source_multicast_vs_static_multicast": ("source_multicast", "static_multicast"),
    "source_ordinary_vs_source_multicast": ("source_ordinary", "source_multicast"),
    "source_ordinary_vs_triton_b": ("source_ordinary", "triton_b"),
    "triton_b_vs_cublas": ("triton_b", "cublas"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze() -> dict:
    run = json.loads((HERE / "results.json").read_text())
    paired = json.loads((HERE / "output/paired.json").read_text())
    for field, path in (
        ("source_sha256", HERE / "kernel.py"),
        ("static_sha256", ROOT / "docs/experiments/triton-h100-cluster-gluon-v2/kernel.py"),
        ("fast_b_sha256", ROOT / "docs/experiments/triton-h100-bd-crossover/trace_B.py"),
    ):
        if run[field] != sha256(path) or paired[field] != run[field]:
            raise ValueError(f"source identity mismatch: {field}")
    if not run["complete"] or not paired["complete"]:
        return {"schema_version": 1, "complete": False, "run_id": run["run_id"],
                "remote_exit_code": run.get("remote_exit_code"),
                "error": paired.get("error"), "compiler": paired.get("compiler"),
                "correctness": paired.get("correctness")}
    rows = paired["observations"]
    by_key = {(row["round"], row["name"]): row for row in rows}
    if len(rows) != 18 or len(by_key) != 18:
        raise ValueError("expected six arms in three rounds")
    if set(paired["correctness"]) != set(ARMS) or not all(
            item["exact"] and item["maximum_error"] == 0
            for item in paired["correctness"].values()):
        raise ValueError("non-exact output")
    compiler = paired["compiler"]
    for name, item in compiler.items():
        for extension in ("ptx", "cubin"):
            path = HERE / "output" / f"{name}.{extension}"
            if sha256(path) != item[f"{extension}_sha256"]:
                raise ValueError(f"artifact hash mismatch: {name}.{extension}")
    report = {"schema_version": 1, "complete": True, "run_id": run["run_id"],
              "source_sha256": run["source_sha256"],
              "dynamic_same_cubin": paired["dynamic_same_cubin"],
              "compiler": compiler, "arms": {}, "contrasts": {}}
    for name in ARMS:
        arm_rows = [by_key[round_index, name] for round_index in range(3)]
        if any(len(row["benchmark"]["timings_us"]) != 30 for row in arm_rows):
            raise ValueError(f"missing samples: {name}")
        medians = [row["benchmark"]["median_us"] for row in arm_rows]
        report["arms"][name] = {"round_medians_us": medians,
                                "median_round_us": statistics.median(medians)}
    for label, (treatment, control) in CONTRASTS.items():
        deltas = [by_key[i, treatment]["benchmark"]["median_us"] -
                  by_key[i, control]["benchmark"]["median_us"] for i in range(3)]
        report["contrasts"][label] = {"treatment": treatment, "control": control,
                                      "round_deltas_us": deltas,
                                      "median_paired_delta_us": statistics.median(deltas)}
    return report


if __name__ == "__main__":
    (HERE / "analysis.json").write_text(json.dumps(analyze(), indent=2, sort_keys=True) + "\n")
