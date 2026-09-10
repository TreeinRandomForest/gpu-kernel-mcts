from __future__ import annotations

import json
import os
import shutil
import statistics
import subprocess
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from .domain import BenchmarkResult, WorkloadContract
from .evaluation import EvaluationInfrastructureError


class BaselineRunner(Protocol):
    def __call__(
        self,
        command: Sequence[str],
        *,
        timeout: float,
        env: Mapping[str, str],
    ) -> subprocess.CompletedProcess[str]: ...


@dataclass(frozen=True, slots=True)
class VendorBaselineConfig:
    artifact_root: Path
    cutlass_path: Path = Path("/opt/cutlass")
    nvcc: str = "nvcc"
    architecture: str = "sm_90a"
    warmup_count: int = 10
    measurement_count: int = 30
    seed: int = 0
    timeout_seconds: float = 180.0
    subprocess_environment: Mapping[str, str] = field(
        default_factory=lambda: {"PATH": os.defpath, "LANG": "C", "LC_ALL": "C"}
    )


class VendorBaselineSuite:
    """Fixed cuBLAS and CUTLASS references outside the MCTS state space."""

    def __init__(
        self,
        config: VendorBaselineConfig,
        *,
        runner: BaselineRunner = subprocess.run,
    ) -> None:
        self.config = config
        self._runner = runner

    def run(self, workload: WorkloadContract) -> dict[str, object]:
        executable = self._compile()
        return {
            implementation: self._measure(executable, workload, implementation)
            for implementation in ("cublas", "cutlass")
        }

    def _compile(self) -> Path:
        include = self.config.cutlass_path / "include"
        if not include.is_dir():
            raise EvaluationInfrastructureError("pinned CUTLASS headers are unavailable")
        self.config.artifact_root.mkdir(parents=True, exist_ok=True)
        executable = self.config.artifact_root / "bf16-gemm-vendor-baselines"
        source = files("kernel_mcts.benchmarks").joinpath(
            "kernels", "bf16_gemm_vendor_baselines.cu"
        )
        command = [
            shutil.which(self.config.nvcc) or self.config.nvcc,
            f"-arch={self.config.architecture}",
            "-std=c++17",
            "-O3",
            f"-I{include}",
            str(source),
            "-lcublas",
            "-o",
            str(executable),
        ]
        try:
            result = self._runner(
                command,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
                check=False,
                env=self.config.subprocess_environment,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise EvaluationInfrastructureError("vendor baseline compilation failed") from error
        if result.returncode != 0:
            raise EvaluationInfrastructureError("vendor baseline compilation failed")
        return executable

    def _measure(
        self,
        executable: Path,
        workload: WorkloadContract,
        implementation: str,
    ) -> dict[str, object]:
        shape = workload.shapes[0].dimensions
        command = [
            str(executable),
            f"--implementation={implementation}",
            f"--M={shape['M']}",
            f"--N={shape['N']}",
            f"--K={shape['K']}",
            f"--rtol={workload.rtol}",
            f"--atol={workload.atol}",
            f"--seed={self.config.seed}",
            f"--warmups={self.config.warmup_count}",
            f"--measurements={self.config.measurement_count}",
        ]
        try:
            result = self._runner(
                command,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
                check=False,
                env=self.config.subprocess_environment,
            )
            payload = json.loads(result.stdout)
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
            raise EvaluationInfrastructureError("vendor baseline execution failed") from error
        if result.returncode != 0 or payload.get("status") != "ok":
            raise EvaluationInfrastructureError("vendor baseline execution failed")
        timings = tuple(float(value) for value in payload["timings_us"])
        if len(timings) != self.config.measurement_count:
            raise EvaluationInfrastructureError("vendor baseline timing count is invalid")
        benchmark = BenchmarkResult(
            timings,
            statistics.median(timings),
            warmup_count=self.config.warmup_count,
            mean_us=statistics.fmean(timings),
            stddev_us=statistics.pstdev(timings),
            min_us=min(timings),
            max_us=max(timings),
        )
        return {
            "correctness": {
                "success": bool(payload["success"]),
                "maximum_error": float(payload["maximum_error"]),
                "mean_error": float(payload["mean_error"]),
            },
            "benchmark": benchmark,
        }
