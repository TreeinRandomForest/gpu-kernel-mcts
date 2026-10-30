"""Build a bounded, evidence-labeled comparison from saved NCU exports."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from kernel_mcts.ncu_source_report import has_source_correlation, summarize_correlated_source


HERE = Path(__file__).resolve().parent
METRICS = {
    "local_load_sectors": "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum",
    "local_store_sectors": "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum",
    "dram_bytes_read_total": "dram__bytes_read.sum",
    "dram_bytes_written_total": "dram__bytes_write.sum",
    "l2_sector_hit_rate_total_pct": "lts__t_sector_hit_rate.pct",
}


def diagnostic_metrics(path: Path) -> dict[str, float]:
    rows = list(csv.reader(path.read_text(encoding="utf-8").splitlines()))
    header = next(row for row in rows if row and row[0] == "ID")
    values = rows[rows.index(header) + 2]
    return {alias: float(values[header.index(name)]) for alias, name in METRICS.items()}


def main() -> None:
    run = json.loads((HERE / "results.json").read_text())
    if not run["complete"]:
        raise RuntimeError("source-report run is incomplete")
    observations = []
    for recorded in run["observations"]:
        name = recorded["variant"]
        correlated_csv = (HERE / f"{name}.cuda_sass.csv").read_text(encoding="utf-8")
        source_lines = summarize_correlated_source(correlated_csv)
        observations.append({
            "variant": name,
            "source_sha256": recorded["source_sha256"],
            "median_us": recorded["median_us"],
            "compiler_resources": recorded["compiler_resources"],
            "sass_local_load_sites": recorded["source_report"]["sass_local_load_sites"],
            "sass_local_store_sites": recorded["source_report"]["sass_local_store_sites"],
            "report_sha256": recorded["source_report"]["report_sha256"],
            "top_local_source_lines": source_lines,
            "source_correlated": has_source_correlation(correlated_csv),
            "metrics": diagnostic_metrics(HERE / f"{name}.diagnostic.csv"),
        })
    output = {
        "schema_version": 1,
        "run_id": run["run_id"],
        "benchmark_id": "bf16_gemm_4096_h100",
        "worker_image": run["image"],
        "metric_scope": "whole_kernel_NCU_replay",
        "local_sector_note": "local memory includes spills and other thread-private data",
        "dram_note": "total DRAM traffic is not spill-attributed",
        "source_site_note": "static SASS sites and executed instructions are distinct",
        "observations": observations,
    }
    (HERE / "analysis.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
