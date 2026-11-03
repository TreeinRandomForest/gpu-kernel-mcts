"""Validate and summarize the retained Gluon cluster experiment."""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"


def main() -> None:
    run = json.loads((HERE / "results.json").read_text())
    paired = json.loads((OUTPUT / "paired.json").read_text())
    if not run["complete"] or not paired["complete"]:
        raise ValueError("incomplete paired experiment")
    if paired["triton_version"] != "3.9.0":
        raise ValueError("unexpected Triton version")
    if paired["source_sha256"] != run["source_sha256"]:
        raise ValueError("source fingerprint mismatch")
    rows = paired["observations"]
    by_key = {(row["round"], row["name"]): row for row in rows}
    if len(rows) != 6 or len(by_key) != 6:
        raise ValueError("expected three matched rounds")
    summary = {"schema_version": 1, "source_sha256": run["source_sha256"],
               "triton_version": paired["triton_version"], "rounds": [],
               "compiler": {}}
    for name in ("single", "cluster_b_multicast"):
        hashes = set()
        for round_index in range(3):
            row = by_key[round_index, name]
            if row["status"] != "ok" or not row["correctness"]["success"]:
                raise ValueError(f"{name} failed correctness")
            if len(row["benchmark"]["timings_us"]) != 30:
                raise ValueError("wrong sample count")
            for suffix, key in (("ptx", "ptx_sha256"), ("cubin", "cubin_sha256")):
                payload = (OUTPUT / f"{name}-r{round_index}.{suffix}").read_bytes()
                if hashlib.sha256(payload).hexdigest() != row[key]:
                    raise ValueError(f"{name} {suffix} hash mismatch")
            hashes.add(row["cubin_sha256"])
        if len(hashes) != 1:
            raise ValueError(f"{name} cubin varied across rounds")
        sass = (OUTPUT / f"{name}-r0.sass").read_text()
        summary["compiler"][name] = {
            "cubin_sha256": hashes.pop(),
            "registers_per_thread": by_key[0, name]["registers_per_thread"],
            "spills": by_key[0, name]["spills"],
            "shared_memory_bytes": by_key[0, name]["shared_memory_bytes"],
            "ptx_multicast_sites": by_key[0, name]["ptx_multicast_sites"],
            "sass_tma_load_sites": sass.count("UTMALDG.2D"),
            "sass_tma_multicast_sites": sass.count("UTMALDG.2D.MULTICAST"),
        }
    if summary["compiler"]["single"]["sass_tma_multicast_sites"] != 0:
        raise ValueError("single-CTA control contains multicast")
    if summary["compiler"]["cluster_b_multicast"]["sass_tma_multicast_sites"] == 0:
        raise ValueError("clustered arm lacks multicast")
    for round_index in range(3):
        single = by_key[round_index, "single"]["benchmark"]["median_us"]
        cluster = by_key[round_index, "cluster_b_multicast"]["benchmark"]["median_us"]
        summary["rounds"].append({"round": round_index,
                                  "single_median_us": single,
                                  "cluster_median_us": cluster,
                                  "cluster_minus_single_us": cluster - single,
                                  "cluster_slowdown_percent": 100 * (cluster - single) / single})
    summary["median_paired_delta_us"] = statistics.median(
        row["cluster_minus_single_us"] for row in summary["rounds"])
    summary["median_paired_slowdown_percent"] = statistics.median(
        row["cluster_slowdown_percent"] for row in summary["rounds"])
    (HERE / "analysis.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
