"""Small, evidence-labelled H100 resource graph for diagnostics only."""

from __future__ import annotations

import math
from typing import Mapping


SCHEMA_VERSION = 2
HOPPER_GUIDE = "https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html"
CUDA_CAPABILITIES = (
    "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/"
    "compute-capabilities.html"
)
H100_PRODUCT = "https://www.nvidia.com/en-us/data-center/h100/"
HOPPER_ARCHITECTURE = (
    "https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/"
)


def _evidence(
    value: int | float | None,
    unit: str,
    basis: str,
    source: str | None = None,
) -> dict[str, object]:
    return {"value": value, "unit": unit, "basis": basis, "source": source}


def _link(
    link_id: str,
    source: str,
    target: str,
    pathway: str,
    *,
    direction: str = "bidirectional",
    peak_bytes_per_second: int | None = None,
) -> dict[str, object]:
    return {
        "id": link_id,
        "source": source,
        "target": target,
        "pathway": pathway,
        "direction": direction,
        "published_peak_bandwidth": _evidence(
            peak_bytes_per_second,
            "bytes/s",
            "published_peak" if peak_bytes_per_second is not None else "unknown",
            H100_PRODUCT if peak_bytes_per_second is not None else None,
        ),
        "sustainable_bandwidth": _evidence(None, "bytes/s", "uncalibrated"),
        "latency": _evidence(None, "ns", "uncalibrated"),
    }


def h100_sxm_resource_graph(
    manifest: Mapping[str, object],
) -> dict[str, object] | None:
    """Build a nominal single-H100 SXM graph tied to an observed worker manifest.

    Unknown sustainable bandwidths and latencies remain explicit nulls. Published
    maxima are not treated as observed capacity or performance.
    """
    model = manifest.get("gpu_model")
    form_factor = manifest.get("form_factor")
    if (
        not isinstance(model, str)
        or "H100" not in model.upper()
        or not any(size in model.upper() for size in ("80GB", "80 GB"))
        or not isinstance(form_factor, str)
        or not form_factor.upper().startswith("SXM")
        or manifest.get("compute_capability") != "9.0"
        or manifest.get("gpu_count", 1) != 1
        or manifest.get("mig_configuration")
    ):
        return None
    manifest_id = manifest.get("manifest_id")
    if not isinstance(manifest_id, str) or not manifest_id:
        return None

    memory_bytes = manifest.get("total_memory_bytes")
    observed_memory = (
        isinstance(memory_bytes, int)
        and not isinstance(memory_bytes, bool)
        and memory_bytes > 0
    )
    sm_count = manifest.get("sm_count")
    observed_sms = (
        isinstance(sm_count, int) and not isinstance(sm_count, bool) and sm_count > 0
    )
    nodes = [
        {
            "id": "hbm",
            "kind": "memory",
            "scope": "gpu",
            "capacity": _evidence(
                memory_bytes if observed_memory else 80_000_000_000,
                "bytes",
                "observed" if observed_memory else "published_nominal",
                "worker_manifest.total_memory_bytes" if observed_memory else H100_PRODUCT,
            ),
        },
        {
            "id": "l2",
            "kind": "cache",
            "scope": "gpu",
            "capacity": _evidence(
                50_000_000, "bytes", "published_nominal", HOPPER_ARCHITECTURE
            ),
        },
        {
            "id": "sm",
            "kind": "execution_container",
            "scope": "gpu",
            "count": _evidence(
                sm_count if observed_sms else 132,
                "SMs",
                "observed" if observed_sms else "published_nominal",
                "worker_manifest.sm_count" if observed_sms else HOPPER_ARCHITECTURE,
            ),
        },
        {
            "id": "l1",
            "kind": "cache",
            "scope": "per_sm",
            "capacity": _evidence(None, "bytes", "carveout_dependent"),
        },
        {
            "id": "shared",
            "kind": "memory",
            "scope": "per_sm",
            "maximum_capacity": _evidence(
                228 * 1024, "bytes", "published_configurable_max", HOPPER_GUIDE
            ),
        },
        {
            "id": "registers",
            "kind": "register_file",
            "scope": "per_sm",
            "capacity": _evidence(
                65_536, "32-bit registers", "published_nominal", HOPPER_GUIDE
            ),
        },
        {
            "id": "scalar_compute",
            "kind": "compute",
            "scope": "per_sm",
            "units": _evidence(
                128, "FP32 CUDA cores", "published_nominal", HOPPER_ARCHITECTURE
            ),
        },
        {
            "id": "tensor_compute",
            "kind": "compute",
            "scope": "per_sm",
            "units": _evidence(
                4, "tensor cores", "published_nominal", HOPPER_ARCHITECTURE
            ),
        },
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "architecture": "sm_90",
        "model": "H100 SXM 80 GB",
        "manifest_id": manifest_id,
        "purpose": "diagnostic_only",
        "nodes": nodes,
        "links": [
            _link("hbm_l2", "hbm", "l2", "memory_fabric", peak_bytes_per_second=3_350_000_000_000),
            _link("l2_l1", "l2", "l1", "cache_fill"),
            _link("l1_registers", "l1", "registers", "global_load_store"),
            _link("l1_shared", "l1", "shared", "async_copy"),
            _link("l2_shared", "l2", "shared", "tma"),
            _link("shared_registers", "shared", "registers", "shared_load_store"),
            _link("registers_scalar", "registers", "scalar_compute", "operands"),
            _link("shared_tensor", "shared", "tensor_compute", "operands", direction="forward"),
            _link("tensor_registers", "tensor_compute", "registers", "accumulators", direction="forward"),
        ],
        "constraints": [
            {
                "id": "registers_per_thread",
                "resource_id": "registers",
                "scope": "per_thread",
                "maximum": _evidence(255, "32-bit registers", "published_architectural_limit", HOPPER_GUIDE),
            },
            {
                "id": "registers_per_block",
                "resource_id": "registers",
                "scope": "per_block",
                "maximum": _evidence(65_536, "32-bit registers", "published_architectural_limit", CUDA_CAPABILITIES),
            },
            {
                "id": "threads_per_block",
                "resource_id": "sm",
                "scope": "per_block",
                "maximum": _evidence(1024, "threads", "published_architectural_limit", CUDA_CAPABILITIES),
            },
            {
                "id": "resident_threads_per_sm",
                "resource_id": "sm",
                "scope": "per_sm",
                "maximum": _evidence(2048, "threads", "published_architectural_limit", CUDA_CAPABILITIES),
            },
            {
                "id": "resident_blocks_per_sm",
                "resource_id": "sm",
                "scope": "per_sm",
                "maximum": _evidence(32, "blocks", "published_architectural_limit", HOPPER_GUIDE),
            },
            {
                "id": "shared_memory_per_block",
                "resource_id": "shared",
                "scope": "per_block",
                "maximum": _evidence(227 * 1024, "bytes", "published_architectural_limit", HOPPER_GUIDE),
                "note": "Large per-block shared-memory allocations can require an explicit opt-in.",
            },
            {
                "id": "l1_shared_pool_per_sm",
                "resources": ["l1", "shared"],
                "combined_maximum": _evidence(
                    256 * 1024, "bytes", "published_configurable_max", HOPPER_GUIDE
                ),
                "note": "L1 and shared memory use one configurable pool; maxima are not additive.",
            }
        ],
    }


def h100_constraint_checks(
    graph: Mapping[str, object],
    compiler_context: Mapping[str, object] | None,
    launch_config: Mapping[str, object] | None,
) -> list[dict[str, object]]:
    """Compare recorded kernel resources with maxima; never infer search validity."""
    if graph.get("schema_version") != SCHEMA_VERSION:
        return []
    constraints = {
        row["id"]: row for row in graph.get("constraints", [])
        if isinstance(row, Mapping) and "maximum" in row
    }
    context = compiler_context if isinstance(compiler_context, Mapping) else {}
    ptxas = (
        context.get("ptxas")
        if context.get("architecture") == graph.get("architecture")
        else None
    )
    resources = (
        ptxas.get("resources")
        if isinstance(ptxas, Mapping) and ptxas.get("status") == "parsed"
        else None
    )
    resources = resources if isinstance(resources, Mapping) else {}
    launch = launch_config if isinstance(launch_config, Mapping) else {}

    def natural(value: object, *, allow_zero: bool = False) -> int | None:
        if not isinstance(value, int) or isinstance(value, bool):
            return None
        return value if (value >= 0 if allow_zero else value > 0) else None

    block = launch.get("block")
    dimensions = (
        [natural(value) for value in block]
        if isinstance(block, (list, tuple)) and len(block) == 3
        else []
    )
    threads = (
        math.prod(dimensions)
        if dimensions and all(value is not None for value in dimensions)
        else None
    )
    registers = natural(resources.get("registers_per_thread"), allow_zero=True)
    static_shared = natural(resources.get("static_shared_memory_bytes"), allow_zero=True)
    dynamic_shared = natural(launch.get("dynamic_shared_memory_bytes"), allow_zero=True)

    measurements = {
        "registers_per_thread": (registers, "ptxas registers/thread", "reported"),
        "registers_per_block": (
            registers * threads if registers is not None and threads is not None else None,
            "ptxas registers/thread × block threads; allocation rounding excluded",
            "lower_bound",
        ),
        "threads_per_block": (threads, "launch_config.block", "recorded"),
        "shared_memory_per_block": (
            static_shared + dynamic_shared
            if static_shared is not None and dynamic_shared is not None
            else static_shared,
            "ptxas static shared + launch dynamic shared"
            if dynamic_shared is not None
            else "ptxas static shared only; dynamic amount unknown",
            "recorded_total"
            if static_shared is not None and dynamic_shared is not None
            else "lower_bound",
        ),
    }
    checks = []
    for constraint_id, (amount, source, basis) in measurements.items():
        constraint = constraints.get(constraint_id)
        if constraint is None:
            continue
        maximum = constraint["maximum"]
        status = (
            "unknown" if amount is None else
            "exceeds_architectural_limit" if amount > maximum["value"] else
            "within_architectural_limit" if basis != "lower_bound" else
            "lower_bound_within_limit"
        )
        checks.append({
            "constraint_id": constraint_id,
            "observed": _evidence(amount, maximum["unit"], basis, source),
            "status": status,
        })
    return checks


_NCU_RESOURCE_METRICS = {
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": "hbm",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed": "l2",
    "l1tex__throughput.avg.pct_of_peak_sustained_active": "l1",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "sm",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "sm",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active": "tensor_compute",
}


def ncu_resource_observations(
    graph: Mapping[str, object], profile: Mapping[str, object]
) -> dict[str, object] | None:
    """Map NCU percentages onto nodes without inferring link bandwidth/latency."""
    if graph.get("schema_version") != SCHEMA_VERSION or profile.get("profiler") != "ncu":
        return None
    metrics = profile.get("metrics")
    if not isinstance(metrics, Mapping):
        return None
    observations = []
    for metric, resource_id in _NCU_RESOURCE_METRICS.items():
        measurement = metrics.get(metric)
        if not isinstance(measurement, Mapping) or measurement.get("unit") != "%":
            continue
        value = measurement.get("value")
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value < 0
        ):
            continue
        observations.append({
            "resource_id": resource_id,
            "metric": metric,
            "value": float(value),
            "unit": "%",
            "basis": "ncu_observed_percentage",
        })
    if not observations:
        return None
    return {
        "schema_version": 1,
        "graph_schema_version": SCHEMA_VERSION,
        "manifest_id": graph["manifest_id"],
        "profile_metric_set": profile.get("metric_set"),
        "observations": observations,
    }
