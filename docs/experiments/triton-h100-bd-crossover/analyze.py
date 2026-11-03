"""Reconcile exact B/D, two-factor hybrids, PTX, and same-worker profiles."""

from __future__ import annotations

import difflib
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

from kernel_mcts.ptx_dataflow import extract_ptx_dataflow
from kernel_mcts.resource_graph import h100_sxm_resource_graph
from kernel_mcts.resource_overlay import overlay_ptx_dataflow


HERE = Path(__file__).resolve().parent
NAMES = ("B_exact", "B_loop_only", "B_mask_only", "B_both", "D_exact")
OPCODES = ("cp.async", "ld.global", "st.global", "ld.shared", "st.shared",
           "ldmatrix", "mma.sync", "wgmma", "shfl", "bar.sync", "ld.local", "st.local")
PROFILE_KEYS = ("tensor_pipe_utilization_pct", "sm_throughput_pct",
                "math_pipe_throttle_warps_per_issue", "shared_bank_conflicts",
                "instructions_executed", "tensor_instructions_executed",
                "local_load_sectors", "local_store_sectors", "dram_bytes_read",
                "dram_bytes_written", "registers_per_thread")
DEBUG_FILE_MARKER = b"\n\t.file\t1 "


def pre_debug_ptx(ptx: bytes) -> bytes:
    """Bounded comparison of code text before the artifact-path debug section."""
    if ptx.count(DEBUG_FILE_MARKER) != 1:
        raise ValueError("PTX lacks one debug-file boundary")
    return ptx.split(DEBUG_FILE_MARKER, 1)[0]


def ptx_summary(ptx: bytes, source_sha256: str, hardware: dict) -> dict:
    text = ptx.decode()
    entries = [line.split(".entry", 1)[1].split("(", 1)[0].strip()
               for line in text.splitlines() if ".entry" in line]
    if len(entries) != 1:
        raise ValueError("PTX must have one entry")
    digest = hashlib.sha256(ptx).hexdigest()
    graph = extract_ptx_dataflow(ptx, digest, entries[0])
    if graph["status"] not in ("ok", "partial"):
        raise ValueError(f"PTX dataflow unavailable: {graph.get('reason')}")
    overlay = overlay_ptx_dataflow(graph, hardware, source_sha256=source_sha256)
    counts = Counter(node.get("opcode", "") for node in graph["nodes"])
    return {
        "sha256": digest, "bytes": len(ptx),
        "site_count": len(graph["nodes"]), "proven_edge_count": len(graph["edges"]),
        "opcode_family_sites": {name: sum(count for opcode, count in counts.items()
                                          if opcode.startswith(name)) for name in OPCODES},
        "operation_classes": overlay["static_operation_counts"],
        "path_gaps": overlay["path_gaps"], "uncertainty": overlay["ptx_uncertainty"],
    }


def build_analysis(run: dict, trace: dict, ptx_by_variant: dict[str, bytes]) -> dict:
    if not run.get("complete") or run.get("rounds") != 3:
        raise ValueError("crossover incomplete")
    if set(run["sources"]) != set(NAMES):
        raise ValueError("unexpected variants")
    hardware = h100_sxm_resource_graph(run["environment_manifest"])
    if hardware is None:
        raise ValueError("crossover worker is not nominal H100 SXM")
    by_name = {name: [] for name in NAMES}
    for observation in run["observations"]:
        name = observation["variant"]
        if name not in by_name or observation["source_sha256"] != run["sources"][name]["sha256"]:
            raise ValueError("unexpected source identity")
        by_name[name].append(observation)
    summaries = []
    for name in NAMES:
        observations = sorted(by_name[name], key=lambda item: item["round"])
        if [item["round"] for item in observations] != [0, 1, 2]:
            raise ValueError(f"missing or duplicate round: {name}")
        ptx = ptx_by_variant[name]
        ptx_hash = hashlib.sha256(ptx).hexdigest()
        if any(item["ptx_sha256"] != ptx_hash for item in observations):
            raise ValueError(f"PTX changed or mismatched: {name}")
        medians = [item["benchmark"]["median_us"] for item in observations]
        if any(statistics.median(item["benchmark"]["timings_us"]) != median
               for item, median in zip(observations, medians)):
            raise ValueError(f"benchmark samples do not match medians: {name}")
        profile = observations[0].get("profile")
        if not isinstance(profile, dict) or profile.get("metric_set") != "spill_v1":
            raise ValueError(f"missing first-round spill profile: {name}")
        summary = profile["summary"]
        row = {
            "variant": name, "source_sha256": run["sources"][name]["sha256"],
            "round_medians_us": medians,
            "median_of_round_medians_us": statistics.median(medians),
            "range_of_round_medians_us": [min(medians), max(medians)],
            "compiler_resources": observations[0]["compiler_resources"],
            "profile_summary": {key: summary.get(key) for key in PROFILE_KEYS},
            "ptx": ptx_summary(ptx, run["sources"][name]["sha256"], hardware),
        }
        if name in ("B_exact", "D_exact"):
            arm = name[0]
            original = next(item for item in trace["arms"] if item["arm"] == arm)
            row["matches_original_ptx"] = row["ptx"]["sha256"] == original["ptx_sha256"]
            original_ptx = (HERE / f"trace_{arm}.ptx").read_bytes()
            if hashlib.sha256(original_ptx).hexdigest() != original["ptx_sha256"]:
                raise ValueError(f"original trace PTX mismatch: {arm}")
            row["matches_original_pre_debug_ptx"] = (
                pre_debug_ptx(ptx) == pre_debug_ptx(original_ptx)
            )
            row["original_median_us"] = original["median_us"]
        summaries.append(row)
    round_map = {row["variant"]: row["round_medians_us"] for row in summaries}
    paired = {}
    for name in NAMES[1:]:
        differences = [a - b for a, b in zip(round_map[name], round_map["B_exact"])]
        paired[name] = {"delta_us_each_round": differences,
                        "median_delta_us": statistics.median(differences)}
    return {
        "schema_version": 1, "status": "same_worker_source_crossover",
        "worker_manifest_id": hardware["manifest_id"],
        "hardware_graph_topology_revision": hardware["topology_revision"],
        "variants": summaries, "paired_vs_B": paired,
        "limitations": (
            "Three randomized same-worker rounds and one separate NCU replay per source. "
            "PTX sites are static; NCU counters are whole-kernel. Hybrids isolate source "
            "factors, but compiler transformations can interact. This does not identify "
            "which prompt factor caused the original search outcomes."
        ),
    }


def main() -> None:
    run = json.loads((HERE / "results.json").read_text())
    trace = json.loads((HERE / "trace_manifest.json").read_text())
    ptx_by_variant = {name: (HERE / f"{name}.ptx").read_bytes() for name in NAMES}
    analysis = build_analysis(run, trace, ptx_by_variant)
    (HERE / "analysis.json").write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    source_b = (HERE / "trace_B.py").read_text().splitlines(keepends=True)
    source_d = (HERE / "trace_D.py").read_text().splitlines(keepends=True)
    (HERE / "B_D_source.diff").write_text("".join(difflib.unified_diff(
        source_b, source_d, fromfile="trace_B.py", tofile="trace_D.py"
    )))
    for row in analysis["variants"]:
        print(row["variant"], round(row["median_of_round_medians_us"], 3),
              row["compiler_resources"], row["ptx"]["site_count"])


if __name__ == "__main__":
    main()
