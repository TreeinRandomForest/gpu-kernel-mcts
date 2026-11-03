"""Audit the six-arm Gluon cluster and pipeline experiment."""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path

from paired import ARMS


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"
CONTRASTS = {
    "serial_cluster_cost_without_multicast": ("cluster_serial_ordinary", "single_serial"),
    "serial_multicast_increment": ("cluster_serial_multicast", "cluster_serial_ordinary"),
    "single_pipeline_gain": ("single_pipeline", "single_serial"),
    "cluster_pipeline_gain_without_multicast": ("cluster_pipeline_ordinary", "cluster_serial_ordinary"),
    "cluster_pipeline_gain_with_multicast": ("cluster_pipeline_multicast", "cluster_serial_multicast"),
    "pipeline_multicast_increment": ("cluster_pipeline_multicast", "cluster_pipeline_ordinary"),
    "pipelined_cluster_gap": ("cluster_pipeline_multicast", "single_pipeline"),
}


def main() -> None:
    run = json.loads((HERE / "results.json").read_text())
    paired = json.loads((OUTPUT / "paired.json").read_text())
    if not run["complete"] or not run["paired_complete"] or not paired["complete"]:
        raise ValueError("incomplete experiment")
    if paired["triton_version"] != "3.9.0" or paired["source_sha256"] != run["source_sha256"]:
        raise ValueError("toolchain or source mismatch")
    rows = paired["observations"]
    by_key = {(row["round"], row["name"]): row for row in rows}
    if len(rows) != 18 or len(by_key) != 18:
        raise ValueError("expected six arms in three rounds")
    report = {"schema_version": 1, "triton_version": paired["triton_version"],
              "source_sha256": run["source_sha256"], "arms": {}, "contrasts": {}}
    for name, factors in ARMS.items():
        arm_rows = [by_key[round_index, name] for round_index in range(3)]
        for round_index, row in enumerate(arm_rows):
            if row["status"] != "ok" or row["correctness"]["maximum_error"] != 0:
                raise ValueError(f"{name} failed exact correctness")
            if len(row["benchmark"]["timings_us"]) != 30:
                raise ValueError("expected 30 timing samples")
            for suffix in ("ptx", "cubin"):
                payload = (OUTPUT / f"{name}-r{round_index}.{suffix}").read_bytes()
                if hashlib.sha256(payload).hexdigest() != row[f"{suffix}_sha256"]:
                    raise ValueError(f"{name} {suffix} hash mismatch")
        cubin_hashes = {row["cubin_sha256"] for row in arm_rows}
        if len(cubin_hashes) != 1:
            raise ValueError(f"{name} cubin changed across rounds")
        sass = (OUTPUT / f"{name}-r0.sass").read_text()
        multicast_sites = sass.count("UTMALDG.2D.MULTICAST")
        if bool(multicast_sites) != factors[1]:
            raise ValueError(f"{name} multicast SASS contradicts source setting")
        report["arms"][name] = {
            "clustered": factors[0], "multicast": factors[1], "pipelined": factors[2],
            "round_medians_us": [row["benchmark"]["median_us"] for row in arm_rows],
            "median_round_us": statistics.median(row["benchmark"]["median_us"] for row in arm_rows),
            "cubin_sha256": cubin_hashes.pop(),
            "registers_per_thread": arm_rows[0]["registers_per_thread"],
            "spills": arm_rows[0]["spills"],
            "shared_memory_bytes": arm_rows[0]["shared_memory_bytes"],
            "ptx_multicast_sites": arm_rows[0]["ptx_multicast_sites"],
            "sass_tma_load_sites": sass.count("UTMALDG.2D"),
            "sass_tma_multicast_sites": multicast_sites,
        }
    if len({record["cubin_sha256"] for record in report["arms"].values()}) != len(ARMS):
        raise ValueError("two arms compiled to the same cubin")
    for label, (treatment, control) in CONTRASTS.items():
        deltas = [by_key[i, treatment]["benchmark"]["median_us"] -
                  by_key[i, control]["benchmark"]["median_us"] for i in range(3)]
        report["contrasts"][label] = {
            "treatment": treatment, "control": control,
            "round_deltas_us": deltas, "median_paired_delta_us": statistics.median(deltas),
        }
    (HERE / "analysis.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
