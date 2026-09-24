from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping


LIGHTWEIGHT_V1_METRICS = (
    "launch__registers_per_thread",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__throughput.avg.pct_of_peak_sustained_active",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__inst_executed.sum",
)

# NCU reports launch metadata during profiling but does not include it in
# ``--query-metrics-mode all`` device-metric output.
NCU_QUERY_VALIDATION_EXEMPT_METRICS = frozenset(
    {"launch__registers_per_thread"}
)

LIGHTWEIGHT_V1_ALIASES = {
    "launch__registers_per_thread": "registers_per_thread",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "achieved_occupancy_pct",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "sm_throughput_pct",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": "dram_throughput_pct",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed": "l2_throughput_pct",
    "l1tex__throughput.avg.pct_of_peak_sustained_active": "l1tex_throughput_pct",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active": (
        "tensor_pipe_utilization_pct"
    ),
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio": (
        "long_scoreboard_warps_per_issue"
    ),
    "smsp__inst_executed.sum": "instructions_executed",
}

DIAGNOSTIC_V2_ADDITIONAL_METRICS = (
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "lts__t_sector_hit_rate.pct",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum",
    "smsp__warps_eligible.avg.per_cycle_active",
    "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "l1tex__data_bank_conflicts_pipe_lsu.sum",
    "sm__inst_executed_pipe_tensor.sum",
    "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio",
)

DIAGNOSTIC_V2_ADDITIONAL_ALIASES = {
    "dram__bytes_read.sum": "dram_bytes_read",
    "dram__bytes_write.sum": "dram_bytes_written",
    "lts__t_sector_hit_rate.pct": "l2_sector_hit_rate_pct",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum": "local_load_sectors",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum": "local_store_sectors",
    "smsp__warps_eligible.avg.per_cycle_active": "eligible_warps_per_cycle",
    "smsp__issue_active.avg.pct_of_peak_sustained_active": "issue_active_pct",
    "l1tex__data_bank_conflicts_pipe_lsu.sum": "shared_bank_conflicts",
    "sm__inst_executed_pipe_tensor.sum": "tensor_instructions_executed",
    "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio": (
        "barrier_stalled_warps_per_issue"
    ),
    "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio": (
        "short_scoreboard_warps_per_issue"
    ),
    "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio": (
        "math_pipe_throttle_warps_per_issue"
    ),
    "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio": (
        "not_selected_warps_per_issue"
    ),
}


@dataclass(frozen=True, slots=True)
class ProfileMetricSet:
    id: str
    schema_version: int
    architecture: str
    ncu_version_prefixes: tuple[str, ...] | None
    metrics: tuple[str, ...]
    aliases: Mapping[str, str]


_METRIC_SETS = {
    ("lightweight_v1", "*"): ProfileMetricSet(
        "lightweight_v1",
        1,
        "*",
        None,
        LIGHTWEIGHT_V1_METRICS,
        LIGHTWEIGHT_V1_ALIASES,
    ),
    ("diagnostic_v2", "sm_90"): ProfileMetricSet(
        "diagnostic_v2",
        2,
        "sm_90",
        ("2025.1", "2025.2"),
        LIGHTWEIGHT_V1_METRICS + DIAGNOSTIC_V2_ADDITIONAL_METRICS,
        {**LIGHTWEIGHT_V1_ALIASES, **DIAGNOSTIC_V2_ADDITIONAL_ALIASES},
    ),
}

PROFILE_METRIC_SET_IDS = ("lightweight_v1", "diagnostic_v2")


def resolve_profile_metric_set(
    metric_set_id: str,
    architecture: str,
    ncu_version: str | None,
) -> ProfileMetricSet:
    definition = _METRIC_SETS.get((metric_set_id, architecture)) or _METRIC_SETS.get(
        (metric_set_id, "*")
    )
    if definition is None:
        raise ValueError(
            f"profile metric set {metric_set_id!r} is unavailable for {architecture!r}"
        )
    if definition.ncu_version_prefixes is not None:
        normalized_version = normalize_ncu_version(ncu_version)
        if normalized_version not in definition.ncu_version_prefixes:
            supported = ", ".join(definition.ncu_version_prefixes)
            raise ValueError(
                f"profile metric set {metric_set_id!r} requires NCU "
                f"one of ({supported}), got {normalized_version or 'unknown'}"
            )
    return definition


def normalize_ncu_version(value: str | None) -> str | None:
    if value is None:
        return None
    match = re.search(r"\b(20[0-9]{2}\.[0-9]+)(?:\.[0-9]+)?\b", value)
    return match.group(1) if match is not None else None
