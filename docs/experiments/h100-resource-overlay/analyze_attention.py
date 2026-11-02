"""Reconcile verified attention PTX with the generic resource overlay."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from kernel_mcts.cuda_backend import _parse_ncu_csv
from kernel_mcts.overlay_view import summarize_overlay
from kernel_mcts.ptx_dataflow import extract_ptx_dataflow
from kernel_mcts.resource_graph import h100_sxm_resource_graph
from kernel_mcts.resource_overlay import overlay_ptx_dataflow


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"
METRICS = ("dram__bytes_read.sum", "dram__bytes_write.sum",
           "lts__t_sector_hit_rate.pct", "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum",
           "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum")


def build_analysis(result: dict, provenance: dict, ptx: bytes,
                   ncu_csv: str | None = None) -> tuple[dict, dict | None]:
    if result.get("schema_version") != 1 or result.get("kernel") != "causal_attention_probe":
        raise ValueError("unexpected attention result")
    if provenance.get("experiment") != "real_triton_attention_overlay_v1":
        raise ValueError("unexpected attention provenance")
    if result["ptx_bytes"] != len(ptx):
        raise ValueError("PTX byte count mismatch")
    source_hash = hashlib.sha256((HERE / "attention_inside.py").read_bytes()).hexdigest()
    ptx_hash = hashlib.sha256(ptx).hexdigest()
    if source_hash != provenance["source_sha256"] or ptx_hash != provenance["ptx_sha256"]:
        raise ValueError("attention source/PTX hash mismatch")
    hardware = h100_sxm_resource_graph(provenance["worker_manifest"])
    if hardware is None:
        raise ValueError("attention was not compiled on a nominal H100 SXM")
    entries = re.findall(rb"\.entry\s+([^\s(]+)", ptx)
    if len(entries) != 1:
        raise ValueError("attention PTX entry is ambiguous")
    dataflow = extract_ptx_dataflow(ptx, ptx_hash, entries[0].decode())
    ncu = {"status": "unavailable"}
    if ncu_csv is not None:
        try:
            ncu = {"status": "measured", "scope": "whole_kernel_separate_replay",
                   "metrics": _parse_ncu_csv(ncu_csv, METRICS)}
            hit = ncu["metrics"]["lts__t_sector_hit_rate.pct"]["value"]
            if not 0 <= hit <= 100:
                ncu["counter_warnings"] = ["l2_hit_rate_outside_0_100"]
                ncu["metrics"]["lts__t_sector_hit_rate.pct"]["valid"] = False
        except Exception as error:
            ncu = {"status": "parse_failed", "error_type": type(error).__name__}
    report = {
        "schema_version": 1,
        "kernel": result["kernel"], "shape": result["shape"],
        "correctness": result["correctness"],
        "compiler_resources": result["compiler_resources"],
        "source_sha256": source_hash, "ptx_sha256": ptx_hash,
        "hardware_graph_topology_revision": hardware["topology_revision"],
        "ptx_status": dataflow["status"],
        "ptx_reason": dataflow.get("reason"),
        "ncu": ncu,
        "physical_link_calibrations": {},
    }
    if dataflow["status"] not in ("ok", "partial"):
        return report, None
    overlay = overlay_ptx_dataflow(
        dataflow, hardware, source_sha256=source_hash,
        whole_kernel_evidence={"scope": "whole_kernel", "ncu": ncu},
    )
    summary = summarize_overlay(overlay)
    unparsed_def_use = Counter(node.get("opcode", "unknown") for node in dataflow["nodes"]
                               if node.get("kind") == "unsupported")
    unsupported = Counter(node.get("opcode", "unknown") for node in overlay["nodes"]
                          if node["hardware"]["operation_class"] == "unsupported")
    report.update({
        "ptx_site_count": summary["site_count"],
        "ptx_proven_edge_count": summary["proven_edge_count"],
        "sites_without_proven_edges": summary["sites_without_proven_edges"],
        "static_operation_counts": summary["operation_classes"],
        "candidate_link_sites": summary["candidate_link_sites"],
        "path_gaps": summary["path_gaps"],
        "ptx_uncertainty": summary["ptx_uncertainty"],
        "ptx_unparsed_def_use_opcode_sites": dict(sorted(unparsed_def_use.items())),
        "unsupported_opcode_sites": dict(sorted(unsupported.items())),
        "limitations": (
            "PTX def-use is partial; static site counts are not execution counts. "
            "Whole-kernel NCU replay is not instruction-attributed, and no "
            "physical link is calibrated by this attention case."
        ),
    })
    return report, overlay


def main() -> None:
    result = json.loads((OUTPUT / "attention_result.json").read_text())
    provenance = json.loads((OUTPUT / "attention_provenance.json").read_text())
    ptx = (OUTPUT / "attention.ptx").read_bytes()
    ncu_path = OUTPUT / "attention.ncu.csv"
    report, overlay = build_analysis(
        result, provenance, ptx,
        ncu_path.read_text() if ncu_path.exists() else None,
    )
    (OUTPUT / "attention_analysis.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    if overlay is not None:
        (OUTPUT / "attention.overlay.json").write_text(
            json.dumps(overlay, indent=2, sort_keys=True) + "\n"
        )
    print(report["ptx_status"], report.get("ptx_site_count"),
          report.get("ptx_proven_edge_count"), report.get("path_gaps"))


if __name__ == "__main__":
    main()
