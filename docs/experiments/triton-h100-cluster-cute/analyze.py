"""Summarize the retained paired H100 CuTe cluster measurements."""

from __future__ import annotations

import json
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"


def main() -> None:
    run = json.loads((HERE / "results.json").read_text())
    paired = json.loads((OUTPUT / "paired.json").read_text())
    binaries = json.loads((HERE / "binary_audit.json").read_text())
    if not run["complete"] or not paired["complete"]:
        raise ValueError("incomplete experiment")
    observations = paired["observations"]
    if len(observations) != 6 or len(binaries) != 6:
        raise ValueError("expected three paired rounds")
    by_key = {(item["round"], item["name"]): item for item in observations}
    binary_by_key = {(item["round"], item["name"]): item for item in binaries}
    if len(by_key) != 6 or len(binary_by_key) != 6:
        raise ValueError("duplicate or missing arm")
    if len({item["example_sha256"] for item in observations}) != 1:
        raise ValueError("example source changed between arms")
    result = {"schema_version": 1, "experiment": paired["experiment"],
              "source_sha256": observations[0]["example_sha256"],
              "rounds": [], "cubin_sha256": {}}
    for name in ("single", "cluster_b_multicast"):
        hashes = {binary_by_key[round_index, name]["cubin_sha256"]
                  for round_index in range(3)}
        if len(hashes) != 1:
            raise ValueError(f"{name} cubin changed between rounds")
        result["cubin_sha256"][name] = hashes.pop()
    if len(set(result["cubin_sha256"].values())) != 2:
        raise ValueError("arm cubins are identical")
    for round_index in range(3):
        measurements = {}
        for name in ("single", "cluster_b_multicast"):
            item = by_key[round_index, name]
            if (not item["correctness"]["success"]
                    or item["correctness"]["maximum_error"] != 0
                    or len(item["benchmark"]["timings_us"]) != 30):
                raise ValueError(f"failed correctness or timing contract: {name}")
            measurements[name] = item["benchmark"]["median_us"]
        delta = measurements["single"] - measurements["cluster_b_multicast"]
        result["rounds"].append({"round": round_index, "median_us": measurements,
                                 "cluster_savings_us": delta,
                                 "cluster_savings_percent": 100 * delta / measurements["single"]})
    result["median_paired_savings_us"] = statistics.median(
        row["cluster_savings_us"] for row in result["rounds"])
    result["median_paired_savings_percent"] = statistics.median(
        row["cluster_savings_percent"] for row in result["rounds"])
    result["sass"] = {}
    for name in ("single", "cluster_b_multicast"):
        sass = (OUTPUT / f"{name}-r0.sass").read_text()
        result["sass"][name] = {
            "utmaldg_sites": sass.count("UTMALDG.3D"),
            "utmaldg_multicast_sites": sass.count("UTMALDG.3D.MULTICAST"),
        }
    if result["sass"]["single"]["utmaldg_multicast_sites"] != 0:
        raise ValueError("single CTA binary unexpectedly uses multicast")
    if result["sass"]["cluster_b_multicast"]["utmaldg_multicast_sites"] == 0:
        raise ValueError("clustered binary lacks TMA multicast")
    (HERE / "analysis.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
