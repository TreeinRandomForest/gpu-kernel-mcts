"""Summarize four-arm PTX control correctness and paired event timings."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from paired import ARMS


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"
SASS_OUTPUT = HERE.parents[2] / "docs/experiments/triton-h100-cluster-gluon-schedule/output"
CONTRASTS = {
    "patched_vs_generated_ordinary": ("patched_ordinary", "generated_ordinary"),
    "patched_vs_reassembled_multicast": ("patched_ordinary", "reassembled_multicast"),
    "reassembly_effect": ("reassembled_multicast", "generated_multicast"),
}


def analyze() -> dict:
    run = json.loads((HERE / "results.json").read_text())
    paired = json.loads((OUTPUT / "paired.json").read_text())
    if paired["source_sha256"] != run["source_sha256"]:
        raise ValueError("source mismatch")
    if (paired["original_cubin_sha256"] != run["original_cubin_sha256"] or
            paired["patched_cubin_sha256"] != run["patched_cubin_sha256"]):
        raise ValueError("cubin mismatch")
    if not run["complete"] or not paired["complete"]:
        return {"schema_version": 1, "complete": False,
                "run_id": run["run_id"], "remote_exit_code": run.get("remote_exit_code"),
                "observations": paired["observations"]}
    rows = paired["observations"]
    by_key = {(row["round"], row["name"]): row for row in rows}
    if len(rows) != 12 or len(by_key) != 12:
        raise ValueError("expected four arms in three rounds")
    report = {"schema_version": 1, "complete": True, "run_id": run["run_id"],
              "source_sha256": run["source_sha256"], "ptx_sha256": paired["ptx_sha256"],
              "original_cubin_sha256": run["original_cubin_sha256"],
              "patched_cubin_sha256": run["patched_cubin_sha256"],
              "arms": {}, "contrasts": {},
              "reassembled_sass": {
                  "multicast_wait_sites": (SASS_OUTPUT / "multicast_original_v87.sass")
                      .read_text().count("WARPGROUP.DEPBAR.LE"),
                  "ordinary_wait_sites": (SASS_OUTPUT / "ordinary_patched_v2_v87.sass")
                      .read_text().count("WARPGROUP.DEPBAR.LE"),
                  "ordinary_multicast_tma_sites": (SASS_OUTPUT / "ordinary_patched_v2_v87.sass")
                      .read_text().count("UTMALDG.2D.MULTICAST"),
              }}
    for name in ARMS:
        arm_rows = [by_key[i, name] for i in range(3)]
        if any(row["status"] != "ok" or not row["correctness"]["exact"] or
               row["correctness"]["maximum_error"] != 0 for row in arm_rows):
            raise ValueError(f"failed exact correctness: {name}")
        if any(len(row["benchmark"]["timings_us"]) != 30 for row in arm_rows):
            raise ValueError(f"missing timing samples: {name}")
        medians = [row["benchmark"]["median_us"] for row in arm_rows]
        report["arms"][name] = {"round_medians_us": medians,
                                "median_round_us": statistics.median(medians)}
    for label, (treatment, control) in CONTRASTS.items():
        differences = [by_key[i, treatment]["benchmark"]["median_us"] -
                       by_key[i, control]["benchmark"]["median_us"] for i in range(3)]
        report["contrasts"][label] = {
            "round_deltas_us": differences,
            "median_paired_delta_us": statistics.median(differences)}
    return report


if __name__ == "__main__":
    (HERE / "analysis.json").write_text(json.dumps(analyze(), indent=2, sort_keys=True) + "\n")
