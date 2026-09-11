from __future__ import annotations

import argparse

import pytest

from kernel_mcts.runpod_volume import resolve_reusable_volume


def arguments(**changes):
    values = {
        "network_volume_id": None,
        "data_center_id": None,
        "auto_volume": False,
        "ephemeral_storage": False,
        "preferred_data_center_id": None,
        "gpu_type": "NVIDIA H100 80GB HBM3",
        "volume_name": "gpu-kernel-mcts",
    }
    values.update(changes)
    return argparse.Namespace(**values)


def test_ephemeral_storage_skips_volume_discovery() -> None:
    assert resolve_reusable_volume(
        argparse.ArgumentParser(),
        arguments(ephemeral_storage=True),
        "unused-api-key",
    ) == (None, None)


@pytest.mark.parametrize(
    "changes",
    [
        {"ephemeral_storage": True, "auto_volume": True},
        {
            "ephemeral_storage": True,
            "network_volume_id": "volume-1",
            "data_center_id": "US-CA-2",
        },
        {"ephemeral_storage": True, "preferred_data_center_id": "US-CA-2"},
    ],
)
def test_ephemeral_storage_rejects_network_volume_options(changes) -> None:
    with pytest.raises(SystemExit):
        resolve_reusable_volume(
            argparse.ArgumentParser(),
            arguments(**changes),
            "unused-api-key",
        )


def test_storage_mode_is_required() -> None:
    with pytest.raises(SystemExit):
        resolve_reusable_volume(
            argparse.ArgumentParser(),
            arguments(),
            "unused-api-key",
        )
