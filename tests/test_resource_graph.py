from __future__ import annotations

from kernel_mcts.resource_graph import (
    h100_constraint_checks,
    h100_sxm_resource_graph,
    ncu_resource_observations,
    prompt_h100_resource_graph,
)


MANIFEST = {
    "manifest_id": "manifest-h100",
    "gpu_model": "NVIDIA H100 80GB HBM3",
    "form_factor": "SXM",
    "compute_capability": "9.0",
    "gpu_count": 1,
    "total_memory_bytes": 85_520_809_984,
    "sm_count": None,
    "mig_configuration": {},
}


def test_h100_graph_separates_observed_nominal_and_unknown_values() -> None:
    graph = h100_sxm_resource_graph(MANIFEST)
    assert graph is not None
    assert graph["schema_version"] == 2
    assert graph["purpose"] == "diagnostic_only"
    nodes = {node["id"]: node for node in graph["nodes"]}
    assert nodes["hbm"]["capacity"]["value"] == 85_520_809_984
    assert nodes["hbm"]["capacity"]["basis"] == "observed"
    assert nodes["sm"]["count"]["value"] == 132
    assert nodes["sm"]["count"]["basis"] == "published_nominal"
    assert nodes["shared"]["maximum_capacity"]["value"] == 228 * 1024
    assert nodes["registers"]["capacity"]["value"] == 65_536
    assert nodes["l1"]["capacity"]["value"] is None
    constraints = {row["id"]: row for row in graph["constraints"]}
    assert constraints["l1_shared_pool_per_sm"]["combined_maximum"]["value"] == 256 * 1024
    assert constraints["registers_per_thread"]["maximum"]["value"] == 255
    assert constraints["registers_per_block"]["maximum"]["value"] == 65_536
    assert constraints["threads_per_block"]["maximum"]["value"] == 1024
    assert constraints["shared_memory_per_block"]["maximum"]["value"] == 227 * 1024

    links = {link["id"]: link for link in graph["links"]}
    assert links["hbm_l2"]["published_peak_bandwidth"]["value"] == 3_350_000_000_000
    assert links["hbm_l2"]["sustainable_bandwidth"]["value"] is None
    assert all(link["latency"]["value"] is None for link in links.values())
    assert {"l2_shared", "l1_shared", "shared_registers"} <= links.keys()


def test_h100_graph_rejects_other_devices_and_mig() -> None:
    assert h100_sxm_resource_graph({**MANIFEST, "form_factor": "PCIe"}) is None
    assert h100_sxm_resource_graph({**MANIFEST, "gpu_model": "A100"}) is None
    assert h100_sxm_resource_graph({**MANIFEST, "gpu_model": "H100 SXM 94GB"}) is None
    assert h100_sxm_resource_graph({**MANIFEST, "mig_configuration": {"enabled": True}}) is None
    assert h100_sxm_resource_graph({**MANIFEST, "gpu_count": 2}) is None


def test_h100_prompt_view_preserves_limits_and_unknown_link_properties() -> None:
    graph = h100_sxm_resource_graph(MANIFEST)
    assert graph is not None
    view = prompt_h100_resource_graph(graph)
    assert view["schema_version"] == graph["schema_version"]
    assert view["manifest_id"] == MANIFEST["manifest_id"]
    assert {row["id"] for row in view["nodes"]} == {row["id"] for row in graph["nodes"]}
    assert {row["id"] for row in view["links"]} == {row["id"] for row in graph["links"]}
    assert {row["id"] for row in view["constraints"]} == {row["id"] for row in graph["constraints"]}
    links = {row["id"]: row for row in view["links"]}
    assert links["hbm_l2"]["published_peak_bandwidth"]["basis"] == "published_peak"
    assert links["l2_l1"]["published_peak_bandwidth"]["value"] is None
    assert "uncalibrated" in view["uncertainty"]


def test_ncu_observations_keep_percentage_units_and_metric_provenance() -> None:
    graph = h100_sxm_resource_graph(MANIFEST)
    assert graph is not None
    profile = {
        "profiler": "ncu",
        "metric_set": "lightweight_v1",
        "metrics": {
            "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": {
                "value": 1.4, "unit": "%"
            },
            "lts__throughput.avg.pct_of_peak_sustained_elapsed": {
                "value": 11.36, "unit": "%"
            },
            "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active": {
                "value": 0, "unit": "%"
            },
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": {
                "value": float("nan"), "unit": "%"
            },
        },
    }
    record = ncu_resource_observations(graph, profile)
    assert record is not None
    assert record["manifest_id"] == "manifest-h100"
    assert [(row["resource_id"], row["value"]) for row in record["observations"]] == [
        ("hbm", 1.4), ("l2", 11.36), ("tensor_compute", 0.0)
    ]
    assert all(row["unit"] == "%" for row in record["observations"])
    assert ncu_resource_observations(graph, {"profiler": "other"}) is None


def test_compiled_kernel_constraint_comparison_distinguishes_limits_from_unknowns() -> None:
    graph = h100_sxm_resource_graph(MANIFEST)
    context = {
        "architecture": "sm_90",
        "ptxas": {"status": "parsed", "resources": {
            "registers_per_thread": 30, "static_shared_memory_bytes": 1024,
        }},
    }
    launch = {"block": [16, 16, 1], "dynamic_shared_memory_bytes": 0}
    checks = {
        row["constraint_id"]: row
        for row in h100_constraint_checks(graph, context, launch)
    }
    assert checks["registers_per_thread"]["observed"]["value"] == 30
    assert checks["registers_per_thread"]["status"] == "within_architectural_limit"
    assert checks["registers_per_block"]["observed"]["value"] == 30 * 256
    assert checks["registers_per_block"]["status"] == "lower_bound_within_limit"
    assert checks["threads_per_block"]["observed"]["value"] == 256
    assert checks["shared_memory_per_block"]["observed"]["value"] == 1024

    missing_dynamic = {
        row["constraint_id"]: row
        for row in h100_constraint_checks(graph, context, {"block": [16, 16, 1]})
    }
    assert missing_dynamic["shared_memory_per_block"]["status"] == "lower_bound_within_limit"
    mismatch = {
        row["constraint_id"]: row for row in h100_constraint_checks(
            graph, {**context, "architecture": "sm_80"}, launch
        )
    }
    assert mismatch["registers_per_thread"]["status"] == "unknown"

    impossible = {
        row["constraint_id"]: row for row in h100_constraint_checks(
            graph,
            {**context, "ptxas": {"status": "parsed", "resources": {
                "registers_per_thread": 255,
                "static_shared_memory_bytes": 228 * 1024,
            }}},
            {"block": [1025, 1, 1], "dynamic_shared_memory_bytes": 0},
        )
    }
    assert impossible["threads_per_block"]["status"] == "exceeds_architectural_limit"
    assert impossible["registers_per_block"]["status"] == "exceeds_architectural_limit"
    assert impossible["shared_memory_per_block"]["status"] == "exceeds_architectural_limit"
