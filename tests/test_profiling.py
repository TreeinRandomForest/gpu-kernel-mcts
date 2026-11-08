from __future__ import annotations

import pytest

from kernel_mcts.profiling import (
    LIGHTWEIGHT_V1_METRICS,
    REUSE_V1_ADDITIONAL_METRICS,
    SPILL_V1_ADDITIONAL_METRICS,
    normalize_ncu_version,
    resolve_profile_metric_set,
)


@pytest.mark.parametrize("version", ("2025.1.1.0", "2025.2.0.0"))
def test_diagnostic_v2_is_an_sm90_supported_ncu_superset(version: str) -> None:
    definition = resolve_profile_metric_set(
        "diagnostic_v2",
        "sm_90",
        f"NVIDIA Nsight Compute Version {version}",
    )

    assert definition.schema_version == 2
    assert definition.metrics[: len(LIGHTWEIGHT_V1_METRICS)] == (
        LIGHTWEIGHT_V1_METRICS
    )
    assert len(definition.metrics) > len(LIGHTWEIGHT_V1_METRICS)


def test_diagnostic_v2_rejects_unvalidated_architecture_and_ncu_version() -> None:
    with pytest.raises(ValueError, match="unavailable for 'sm_80'"):
        resolve_profile_metric_set("diagnostic_v2", "sm_80", "2025.1")
    with pytest.raises(ValueError, match=r"requires NCU one of \(2025.1, 2025.2\)"):
        resolve_profile_metric_set("diagnostic_v2", "sm_90", "2026.1")


def test_reuse_v1_adds_read_path_counters_without_changing_diagnostic_v2() -> None:
    diagnostic = resolve_profile_metric_set("diagnostic_v2", "sm_90", "2025.1")
    reuse = resolve_profile_metric_set("reuse_v1", "sm_90", "2025.1")
    assert reuse.metrics == (
        diagnostic.metrics + SPILL_V1_ADDITIONAL_METRICS
        + REUSE_V1_ADDITIONAL_METRICS
    )
    assert reuse.aliases["lts__t_sectors_srcunit_tex_op_read.sum"] == "l2_tex_read_sectors"
    assert reuse.aliases["lts__t_sector_op_read_hit_rate.pct"] == "l2_read_sector_hit_rate_pct"
    assert reuse.aliases[
        "l1tex__t_sectors_pipe_lsu_mem_local_op_ld_lookup_miss.sum"
    ] == "local_load_l1_miss_sectors"
    with pytest.raises(ValueError, match="unavailable for 'sm_80'"):
        resolve_profile_metric_set("reuse_v1", "sm_80", "2025.1")


def test_ncu_version_normalization_is_stable() -> None:
    assert normalize_ncu_version("Version 2025.1.1.0 (build 35528883)") == "2025.1"
    assert normalize_ncu_version("Version 2025.2.0.0") == "2025.2"
    assert normalize_ncu_version(None) is None
