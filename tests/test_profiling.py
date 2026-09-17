from __future__ import annotations

import pytest

from kernel_mcts.profiling import (
    LIGHTWEIGHT_V1_METRICS,
    normalize_ncu_version,
    resolve_profile_metric_set,
)


def test_diagnostic_v2_is_an_sm90_ncu_2025_1_superset() -> None:
    definition = resolve_profile_metric_set(
        "diagnostic_v2",
        "sm_90",
        "NVIDIA Nsight Compute Version 2025.1.1.0",
    )

    assert definition.schema_version == 2
    assert definition.metrics[: len(LIGHTWEIGHT_V1_METRICS)] == (
        LIGHTWEIGHT_V1_METRICS
    )
    assert len(definition.metrics) > len(LIGHTWEIGHT_V1_METRICS)


def test_diagnostic_v2_rejects_unvalidated_architecture_and_ncu_version() -> None:
    with pytest.raises(ValueError, match="unavailable for 'sm_80'"):
        resolve_profile_metric_set("diagnostic_v2", "sm_80", "2025.1")
    with pytest.raises(ValueError, match="requires NCU 2025.1"):
        resolve_profile_metric_set("diagnostic_v2", "sm_90", "2026.1")


def test_ncu_version_normalization_is_stable() -> None:
    assert normalize_ncu_version("Version 2025.1.1.0 (build 35528883)") == "2025.1"
    assert normalize_ncu_version(None) is None
