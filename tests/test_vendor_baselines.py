from __future__ import annotations

import json
import subprocess
from importlib.resources import files

from kernel_mcts.benchmarks import BF16_GEMM_WORKLOAD
from kernel_mcts.vendor_baselines import VendorBaselineConfig, VendorBaselineSuite


def test_cutlass_baseline_matches_workload_tensor_layouts() -> None:
    source = (
        files("kernel_mcts.benchmarks")
        .joinpath("kernels", "bf16_gemm_vendor_baselines.cu")
        .read_text(encoding="utf-8")
    )

    assert "cutlass::layout::RowMajor,\n    Element,\n    cutlass::layout::ColumnMajor" in source
    assert "{reinterpret_cast<const Element*>(B), K}" in source


class FakeRunner:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, command, **kwargs):
        self.calls.append((list(command), kwargs))
        if "-o" in command:
            return subprocess.CompletedProcess(command, 0, "", "")
        implementation = next(
            value.split("=", 1)[1]
            for value in command
            if value.startswith("--implementation=")
        )
        median = 1.0 if implementation == "cublas" else 2.0
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(
                {
                    "status": "ok",
                    "success": True,
                    "maximum_error": 0.0,
                    "mean_error": 0.0,
                    "timings_us": [median, median, median],
                }
            ),
            "",
        )


def test_vendor_baselines_share_workload_and_measurement_protocol(tmp_path) -> None:
    cutlass = tmp_path / "cutlass"
    (cutlass / "include").mkdir(parents=True)
    runner = FakeRunner()
    suite = VendorBaselineSuite(
        VendorBaselineConfig(
            artifact_root=tmp_path / "artifacts",
            cutlass_path=cutlass,
            warmup_count=2,
            measurement_count=3,
            seed=7,
        ),
        runner=runner,
    )

    results = suite.run(BF16_GEMM_WORKLOAD)

    assert results["cublas"]["benchmark"].median_us == 1.0
    assert results["cutlass"]["benchmark"].median_us == 2.0
    execution_commands = [call[0] for call in runner.calls[1:]]
    assert {command[1] for command in execution_commands} == {
        "--implementation=cublas",
        "--implementation=cutlass",
    }
    for command in execution_commands:
        assert "--M=4096" in command
        assert "--N=4096" in command
        assert "--K=4096" in command
        assert "--warmups=2" in command
        assert "--measurements=3" in command
        assert "--seed=7" in command
