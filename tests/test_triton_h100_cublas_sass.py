"""Check the same-input comparison's identity and NCU target selection."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parents[1] / "docs/experiments/triton-h100-cublas-sass"
SOURCE = HERE.parent / "triton-h100-bd-crossover/trace_B.py"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"paired_{name}", HERE / name)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_paired_harness_pins_the_exact_fast_b_source():
    harness = _load("paired_harness.py")
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == harness.SOURCE_SHA256


def test_ncu_capture_selects_one_named_range_for_each_implementation():
    runner = _load("run.py")
    for name in ("cublas", "triton"):
        command = runner.source_profile_command(name, "/tmp/profile")
        assert command[command.index("--nvtx-include") + 1] == f"PAIRED_{name.upper()}/"
        assert command[command.index("--launch-count") + 1] == "1"
        assert command[-2:] == ["--target", name]
    assert runner.REMOTE_SOURCE in runner.target_command("benchmark")


def test_sass_parser_counts_dynamic_warp_instructions_by_opcode():
    analyzer = _load("analyze.py")
    csv_text = (
        '"Kernel Name","example"\n'
        '"Address","Source","Instructions Executed"\n'
        '"0x10","      WGMMA.MMA_ASYNC R1, R2","10"\n'
        '"0x20","      @P0 BRA 0x10","2"\n'
    )
    parsed = analyzer.parse_sass_csv(csv_text)
    assert parsed["site_count"] == 2
    assert parsed["executed_warp_instructions"] == 12
    assert parsed["opcode_executions"] == {"BRA": 2, "WGMMA": 10}


def test_raw_parser_keeps_ncu_display_units():
    analyzer = _load("analyze.py")
    raw = (
        '"ID","Kernel Name","Block Size","Grid Size","smsp__inst_executed.sum",'
        '"dram__bytes_read.sum"\n'
        '"","","","","inst","Mbyte"\n'
        '"0","gemm","(256, 1, 1)","(512, 1, 1)","39968768.000000","186.688000"\n'
    )
    parsed = analyzer.parse_raw_csv(raw)
    assert parsed["metrics"]["smsp__inst_executed.sum"]["value"] == 39968768.0
    assert parsed["metrics"]["dram__bytes_read.sum"]["unit"] == "Mbyte"
