"""The crossover varies only loop-offset and row/column mask construction."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parents[1] / "docs/experiments/triton-h100-bd-crossover"


def _variants():
    spec = importlib.util.spec_from_file_location("triton_bd_variants", HERE / "variants.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _analysis():
    spec = importlib.util.spec_from_file_location("triton_bd_analysis", HERE / "analyze.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_trace_sources_and_two_factor_hybrids():
    values = _variants().sources()
    manifest = json.loads((HERE / "trace_manifest.json").read_text())
    hashes = {f'{item["arm"]}_exact': item["source_sha256"] for item in manifest["arms"]}
    assert set(values) == {"B_exact", "B_loop_only", "B_mask_only", "B_both", "D_exact"}
    for name, expected in hashes.items():
        assert hashlib.sha256(values[name].encode()).hexdigest() == expected
    for name, source in values.items():
        ast.parse(source)
        assert "num_warps=8, num_stages=3" in source
        assert "tl.dot(a, b, acc)" in source
        assert "128, 256, 64" in source
    for name in ("B_loop_only", "B_both"):
        assert "for k0 in range(0, K, BK):" in values[name]
    for name in ("B_exact", "B_mask_only"):
        assert "for k in range(0, tl.cdiv(K, BK)):" in values[name]
    for name in ("B_mask_only", "B_both"):
        assert "a_mask_row = row[:, None] < M" in values[name]
        assert "b_mask_col = col[None, :] < N" in values[name]
    for name in ("B_exact", "B_loop_only"):
        assert "a_mask_row =" not in values[name]


def test_trace_ptx_hashes_and_static_analysis():
    analyzer = _analysis()
    manifest = json.loads((HERE / "trace_manifest.json").read_text())
    hardware = {
        "manifest_id": "test", "gpu_model": "H100 80GB", "form_factor": "SXM",
        "compute_capability": "9.0", "gpu_count": 1, "mig_configuration": {},
    }
    from kernel_mcts.resource_graph import h100_sxm_resource_graph
    graph = h100_sxm_resource_graph(hardware)
    assert graph is not None
    for arm in manifest["arms"]:
        ptx = (HERE / f'trace_{arm["arm"]}.ptx').read_bytes()
        assert hashlib.sha256(ptx).hexdigest() == arm["ptx_sha256"]
        summary = analyzer.ptx_summary(ptx, arm["source_sha256"], graph)
        assert summary["site_count"] > 0
        assert summary["proven_edge_count"] > 0
        assert summary["opcode_family_sites"]["wgmma"] > 0


def test_crossover_analysis_pairs_rounds_and_keeps_profile_scope():
    analyzer = _analysis()
    values = _variants().sources()
    ptx_b = (HERE / "trace_B.ptx").read_bytes()
    ptx_d = (HERE / "trace_D.ptx").read_bytes()
    ptx = {name: ptx_d if name == "D_exact" else ptx_b for name in values}
    sources = {name: {"sha256": hashlib.sha256(source.encode()).hexdigest()}
               for name, source in values.items()}
    observations = []
    for name in values:
        for round_index in range(3):
            item = {
                "round": round_index, "variant": name,
                "source_sha256": sources[name]["sha256"],
                "ptx_sha256": hashlib.sha256(ptx[name]).hexdigest(),
                "benchmark": {"median_us": 200.0 + round_index + (10 if name == "D_exact" else 0),
                              "timings_us": [200.0 + round_index + (10 if name == "D_exact" else 0)]},
                "compiler_resources": {"registers_per_thread": 200},
            }
            if round_index == 0:
                item["profile"] = {"metric_set": "spill_v1", "summary": {
                    "tensor_pipe_utilization_pct": 80.0,
                }}
            observations.append(item)
    run = {"complete": True, "rounds": 3, "sources": sources,
           "environment_manifest": {
               "manifest_id": "test", "gpu_model": "H100 80GB", "form_factor": "SXM",
               "compute_capability": "9.0", "gpu_count": 1, "mig_configuration": {},
           }, "observations": observations}
    trace = json.loads((HERE / "trace_manifest.json").read_text())
    result = analyzer.build_analysis(run, trace, ptx)
    assert result["paired_vs_B"]["D_exact"]["median_delta_us"] == 10.0
    assert result["variants"][0]["profile_summary"]["tensor_pipe_utilization_pct"] == 80.0
    assert result["variants"][0]["matches_original_ptx"] is True
    assert result["variants"][0]["matches_original_pre_debug_ptx"] is True
    assert result["variants"][-1]["matches_original_pre_debug_ptx"] is True
