from __future__ import annotations

import json
import subprocess

import pytest

from kernel_mcts.ncu_source_report import (
    collect_source_report, has_source_correlation, summarize_correlated_source,
)


def test_selected_source_report_exports_sass_and_bounded_summary(tmp_path) -> None:
    calls = []
    prefix = tmp_path / "selected"

    def fake_runner(command, **kwargs):
        calls.append(command)
        if "--export" in command:
            prefix.with_suffix(".ncu-rep").write_bytes(b"ncu report")
            return subprocess.CompletedProcess(command, 0, "", "")
        view = command[command.index("--print-source") + 1]
        text = {
            "sass": "0 LDL.32 R1, [R2];\n1 STL.32 [R2], R1;\n",
            "ptx": "ld.global.b32\n",
            "cuda,sass": (
                '"File Path","/tmp/candidate.py"\n'
                '"Line No","Source","Address","Source","Instructions Executed"\n'
                '"26","acc = tl.dot(a, b, acc)","-","-","-"\n'
                '"","","0x1","LDL.32 R1, [R2]","128"\n'
            ),
        }[view]
        return subprocess.CompletedProcess(command, 0, text, "")

    result = collect_source_report(
        prefix, ["python", "-m", "kernel_mcts.triton_harness"],
        kernel_regex="bf16_gemm_triton", runner=fake_runner,
    )

    assert result["sass_local_load_sites"] == 1
    assert result["sass_local_store_sites"] == 1
    assert result["source_correlated"] is True
    assert result["top_local_source_lines"][0] == {
        "line": 26, "source": "acc = tl.dot(a, b, acc)",
        "load_sites": 1, "store_sites": 0, "executed_local_instructions": 128,
    }
    assert json.loads(prefix.with_suffix(".json").read_text()) == result
    assert calls[0][:4] == ["ncu", "--set", "full", "--import-source"]
    assert "--launch-count" in calls[0]
    with pytest.raises(FileExistsError):
        collect_source_report(prefix, ["python"], kernel_regex="x", runner=fake_runner)


def test_source_correlation_can_exist_without_local_memory_sites() -> None:
    source = ('"Line No","Source","Address","Source","Instructions Executed"\n'
              '"38","acc = tl.dot(a, b, acc)","-","-","-"\n')
    assert has_source_correlation(source)
    assert summarize_correlated_source(source) == []
