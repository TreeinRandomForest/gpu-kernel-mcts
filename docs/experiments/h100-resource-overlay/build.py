"""Overlay saved Triton PTX dataflow on the nominal H100 resource graph."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from kernel_mcts.ptx_dataflow import extract_ptx_dataflow
from kernel_mcts.resource_graph import h100_sxm_resource_graph
from kernel_mcts.resource_overlay import overlay_ptx_dataflow


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
WARP = EXPERIMENTS / "triton-h100-warp-cliff"
CROSSOVER = EXPERIMENTS / "triton-h100-winner-crossover"
SOURCE_REPORT = EXPERIMENTS / "triton-h100-source-report"
OUTPUT = HERE / "output"
VARIANTS = ("grouped_bn128_w4", "grouped_bn256_w4", "grouped_bn256_w8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    run = json.loads((WARP / "results.json").read_text())
    source_report = json.loads((SOURCE_REPORT / "analysis.json").read_text())
    if not run.get("complete") or not source_report.get("observations"):
        raise ValueError("selected-kernel evidence is incomplete")
    hardware = h100_sxm_resource_graph(run["environment_manifest"])
    if hardware is None:
        raise ValueError("selected-kernel worker is not a nominal H100 SXM")
    by_source_hash = {item["source_sha256"]: item
                      for item in source_report["observations"]}
    observations = {item["variant"]: item for item in run["observations"]}
    summary = []
    for variant in VARIANTS:
        observation = observations[variant]
        source_path = CROSSOVER / f"{variant}.py"
        source_hash = sha256(source_path)
        if source_hash != run["variants"][variant]["source_sha256"]:
            raise ValueError(f"source hash mismatch: {variant}")
        ptx_path = WARP / f"{variant}.ptx"
        ptx_raw = ptx_path.read_bytes()
        ptx_hash = hashlib.sha256(ptx_raw).hexdigest()
        recorded_hash = observation["compiler_context"]["ptx"]["artifacts"][0]["sha256"]
        if ptx_hash != recorded_hash:
            raise ValueError(f"PTX hash mismatch: {variant}")
        entry = re.findall(rb"\.entry\s+([^\s(]+)", ptx_raw)
        if len(entry) != 1:
            raise ValueError(f"PTX entry is ambiguous: {variant}")
        dataflow = extract_ptx_dataflow(ptx_raw, ptx_hash, entry[0].decode())
        if dataflow.get("status") not in ("ok", "partial"):
            raise ValueError(f"PTX dataflow unavailable: {variant}")
        source_evidence = by_source_hash.get(source_hash)
        measured = [{
            "source": "warp_cliff_results",
            "worker_manifest_id": run["environment_manifest"]["manifest_id"],
            "metrics": observation["profile"]["summary"],
        }]
        if source_evidence is not None:
            measured.append({
                "source": "separate_worker_source_report",
                "worker_image": source_report["worker_image"],
                "metrics": source_evidence["metrics"],
                "sass_local_load_sites": source_evidence["sass_local_load_sites"],
                "sass_local_store_sites": source_evidence["sass_local_store_sites"],
            })
        overlay = overlay_ptx_dataflow(
            dataflow, hardware, source_sha256=source_hash,
            whole_kernel_evidence={"scope": "whole_kernel", "measurements": measured},
        )
        (OUTPUT / f"{variant}.overlay.json").write_text(
            json.dumps(overlay, indent=2, sort_keys=True) + "\n"
        )
        benchmark = observation["benchmark"]
        summary.append({
            "variant": variant,
            "source_sha256": source_hash,
            "ptx_sha256": ptx_hash,
            "ptx_dataflow_status": dataflow["status"],
            "ptx_node_count": len(overlay["nodes"]),
            "ptx_edge_count": len(overlay["edges"]),
            "static_operation_counts": overlay["static_operation_counts"],
            "path_gaps": overlay["path_gaps"],
            "ptx_uncertainty": overlay["ptx_uncertainty"],
            "compiler_resources": observation["compiler_context"]["resources"],
            "benchmark": benchmark,
            "whole_kernel_profile": observation["profile"]["summary"],
            "separate_worker_source_metrics": (
                source_evidence["metrics"] if source_evidence is not None else None
            ),
            "separate_worker_sass_local_sites": (
                {"load": source_evidence["sass_local_load_sites"],
                 "store": source_evidence["sass_local_store_sites"]}
                if source_evidence is not None else None
            ),
        })
    (OUTPUT / "summary.json").write_text(json.dumps({
        "schema_version": 1,
        "status": "partial_diagnostic",
        "hardware_graph_manifest_id": hardware["manifest_id"],
        "worker_image": run["image"],
        "variants": summary,
        "limitations": (
            "PTX def-use is partial and static; cache destinations, memory aliases, "
            "dynamic loop counts, and per-site traffic are unresolved. Source-report "
            "NCU/SASS evidence came from a separate worker. No whole-kernel metric "
            "is attached to an individual PTX site."
        ),
    }, indent=2, sort_keys=True) + "\n")
    for item in summary:
        print(item["variant"], item["ptx_dataflow_status"],
              item["ptx_node_count"], item["ptx_edge_count"],
              item["compiler_resources"])


if __name__ == "__main__":
    main()
