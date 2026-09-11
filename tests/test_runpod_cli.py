from __future__ import annotations

import io

from kernel_mcts.providers import EnvironmentManifest
from kernel_mcts.runpod_cli import ReadinessProgress
from kernel_mcts.runpod_cli import _print_calibration


class TTYBuffer(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_readiness_progress_animates_one_terminal_line() -> None:
    stream = TTYBuffer()
    progress = ReadinessProgress("pod pod-1", stream)

    progress("CREATED", 0.0)
    progress("RUNNING", 2.0)
    progress.finish()

    output = stream.getvalue()
    assert "⠋ Waiting for pod pod-1 — status: CREATED — 0s elapsed" in output
    assert "⠙ Waiting for pod pod-1 — status: RUNNING — 2s elapsed" in output
    assert output.count("\n") == 1


def test_noninteractive_progress_logs_only_status_changes() -> None:
    stream = io.StringIO()
    progress = ReadinessProgress("pod pod-1", stream)

    progress("CREATED", 0.0)
    progress("CREATED", 2.0)
    progress("RUNNING", 4.0)
    progress.finish()

    assert stream.getvalue().splitlines() == [
        "Waiting for pod pod-1 — status: CREATED — 0s elapsed",
        "Waiting for pod pod-1 — status: RUNNING — 4s elapsed",
    ]


def test_calibration_report_includes_hardware_correctness_and_latency(capsys) -> None:
    _print_calibration(
        EnvironmentManifest(
            worker_id="pod-1",
            provider="runpod",
            gpu_model="NVIDIA H100 80GB HBM3",
            compute_capability="9.0",
            form_factor="SXM",
            captured_at="2026-09-10T00:00:00+00:00",
        ),
        {
            "benchmark_id": "bf16_gemm_h100_v1",
            "compile": {"success": True},
            "correctness": {
                "success": True,
                "maximum_error": 0.01,
                "mean_error": 0.001,
            },
            "benchmark": {
                "median_us": 10.0,
                "mean_us": 11.0,
                "min_us": 9.0,
                "max_us": 13.0,
                "timings_us": [9.0, 10.0, 13.0],
            },
            "vendor_baselines": {
                "cublas": {
                    "correctness": {"success": True, "maximum_error": 0.0},
                    "benchmark": {
                        "median_us": 1.0,
                        "mean_us": 1.1,
                        "timings_us": [1.0, 1.2],
                    },
                }
            },
        },
    )

    output = capsys.readouterr().out
    assert "NVIDIA H100 80GB HBM3 (SXM)" in output
    assert "correctness=True" in output
    assert "median=10.0 us" in output
    assert "samples=3" in output
    assert "cublas baseline: correctness=True" in output
    assert "median=1.0 us" in output
