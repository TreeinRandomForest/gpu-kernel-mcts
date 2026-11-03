"""Audit the three-arm Gluon B-descriptor layout experiment."""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path

from paired import ARMS


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"
CONTRASTS = {
    "local_b_effect": ("ordinary_local", "ordinary_replicated"),
    "multicast_vs_replicated": ("multicast_replicated", "ordinary_replicated"),
    "multicast_vs_local": ("multicast_replicated", "ordinary_local"),
}


def main() -> None:
    run = json.loads((HERE / "results.json").read_text())
    paired = json.loads((OUTPUT / "paired.json").read_text())
    if not run["complete"]:
        raise ValueError("incomplete worker run")
    if paired["triton_version"] != "3.9.0" or paired["source_sha256"] != run["source_sha256"]:
        raise ValueError("toolchain or source mismatch")
    rows = paired["observations"]
    by_key = {(row["round"], row["name"]): row for row in rows}
    if len(rows) != 9 or len(by_key) != 9:
        raise ValueError("expected three arms in three rounds")
    report = {"schema_version": 1, "triton_version": paired["triton_version"],
              "source_sha256": run["source_sha256"], "arms": {}, "contrasts": {}}
    for name, factors in ARMS.items():
        arm_rows = [by_key[round_index, name] for round_index in range(3)]
        if any(row["status"] != "ok" for row in arm_rows):
            report["arms"][name] = {
                "status": "failed", "b_local": factors[4],
                "errors": sorted({row.get("traceback_tail", row.get("error", ""))[-1500:]
                                  for row in arm_rows if row["status"] != "ok"}),
            }
            continue
        for round_index, row in enumerate(arm_rows):
            if row["correctness"]["maximum_error"] != 0:
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
        report["arms"][name] = {
            "status": "ok",
            "clustered": factors[0], "multicast": factors[1], "pipelined": factors[2],
            "b_first": factors[3], "b_local": factors[4],
            "round_medians_us": [row["benchmark"]["median_us"] for row in arm_rows],
            "median_round_us": statistics.median(row["benchmark"]["median_us"] for row in arm_rows),
            "cubin_sha256": cubin_hashes.pop(),
            "registers_per_thread": arm_rows[0]["registers_per_thread"],
            "spills": arm_rows[0]["spills"],
            "shared_memory_bytes": arm_rows[0]["shared_memory_bytes"],
            "ptx_multicast_sites": arm_rows[0]["ptx_multicast_sites"],
        }
    if len({record["cubin_sha256"] for record in report["arms"].values()
            if record["status"] == "ok"}) != sum(record["status"] == "ok"
                                                  for record in report["arms"].values()):
        raise ValueError("two successful arms compiled to the same cubin")
    for label, (treatment, control) in CONTRASTS.items():
        if (report["arms"][treatment]["status"] != "ok" or
                report["arms"][control]["status"] != "ok"):
            continue
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
