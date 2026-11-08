"""GPU-free checks for the matched reuse-strategy experiment."""

import importlib.util
import json
from pathlib import Path

import pytest

from kernel_mcts.config import load_data, parse_strategies


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "docs/experiments/triton-h100-reuse-strategy/run.py"
SPEC = importlib.util.spec_from_file_location("reuse_strategy_experiment", RUNNER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_only_new_cross_cta_strategy_is_added():
    old = parse_strategies(load_data(MODULE.OLD_CATALOG))
    new = parse_strategies(load_data(MODULE.NEW_CATALOG))
    MODULE.verify_catalogs(old, new)
    with pytest.raises(ValueError):
        MODULE.verify_catalogs(old, tuple(reversed(new)))


def test_smoke_prompt_requires_reuse_and_spill_evidence():
    payload = {
        "strategy": {"id": "improve_cross_cta_reuse", "backend_prompt": "Change BM and BN"},
        "parent_profile": {"metric_set": "reuse_v1", "cache_control": "all",
                           "replay_mode": "kernel",
                           "summary": {name: 0 for name in (
                               "dram_bytes_read", "l2_tex_read_sectors",
                               "l2_tex_read_lookup_hit_sectors",
                               "l2_tex_read_lookup_miss_sectors",
                               "l2_read_sector_hit_rate_pct", "compiler_spills",
                               "local_load_l1_hit_sectors", "local_load_l1_miss_sectors")}},
    }
    assert MODULE.verify_smoke_prompt(payload)["profile_metric_set"] == "reuse_v1"
    del payload["parent_profile"]["summary"]["l2_tex_read_sectors"]
    with pytest.raises(ValueError):
        MODULE.verify_smoke_prompt(payload)
