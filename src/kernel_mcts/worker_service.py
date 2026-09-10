from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from .benchmarks import BF16_GEMM_WORKLOAD, load_bf16_gemm_root
from .cuda_backend import CudaBackendConfig, CudaCppBackend
from .evaluation import BackendKernelEvaluator, EvaluationContext
from .providers import EnvironmentManifest
from .serialization import serialize_benchmark
from .vendor_baselines import VendorBaselineConfig, VendorBaselineSuite
from .worker_protocol import WorkerApplication


def build_application(environ=None) -> WorkerApplication:
    environment = os.environ if environ is None else environ
    token = environment.get("KERNEL_MCTS_WORKER_TOKEN")
    pod_id = environment.get("RUNPOD_POD_ID")
    if not token or not pod_id:
        raise RuntimeError("worker token and RUNPOD_POD_ID are required")
    manifest = capture_environment_manifest(environment)
    backend = CudaCppBackend(
        CudaBackendConfig(
            artifact_root=Path(
                environment.get("KERNEL_MCTS_ARTIFACT_ROOT", "/tmp/kernel-mcts-artifacts")
            )
        )
    )
    root_compilation = backend.compile(load_bf16_gemm_root(), BF16_GEMM_WORKLOAD)
    if not root_compilation.success or root_compilation.artifact is None:
        raise RuntimeError("fixed root kernel failed to compile on the worker")
    root_correctness = backend.check_correctness(
        root_compilation.artifact,
        BF16_GEMM_WORKLOAD,
    )
    if not root_correctness.success:
        raise RuntimeError("fixed root kernel failed correctness on the worker")
    root_benchmark = backend.benchmark(root_compilation.artifact, BF16_GEMM_WORKLOAD)
    vendor_results = VendorBaselineSuite(
        VendorBaselineConfig(
            artifact_root=Path(
                environment.get("KERNEL_MCTS_ARTIFACT_ROOT", "/tmp/kernel-mcts-artifacts")
            ),
            cutlass_path=Path(environment.get("CUTLASS_PATH", "/opt/cutlass")),
        )
    ).run(BF16_GEMM_WORKLOAD)
    evaluator = BackendKernelEvaluator(
        backend=backend,
        root_benchmark=root_benchmark,
        context=EvaluationContext(
            worker_id=pod_id,
            environment_manifest_id=manifest.manifest_id,
            launch_config=BF16_GEMM_WORKLOAD.metadata["abi"]["launch"],
            hardware_toolchain={
                "gpu_model": manifest.gpu_model,
                "compute_capability": manifest.compute_capability,
                "form_factor": manifest.form_factor,
                "toolchain_versions": manifest.toolchain_versions,
            },
        ),
    )
    calibration = {
        "benchmark_id": BF16_GEMM_WORKLOAD.benchmark_id,
        "compile": {
            "success": root_compilation.success,
            "duration_seconds": root_compilation.duration_seconds,
            "artifact_id": root_compilation.artifact.artifact_id,
        },
        "correctness": {
            "success": root_correctness.success,
            "maximum_error": root_correctness.maximum_error,
            "mean_error": root_correctness.mean_error,
            "failed_test_id": root_correctness.failed_test_id,
            "reference_metadata": dict(root_correctness.reference_metadata),
        },
        "benchmark": serialize_benchmark(root_benchmark),
        "vendor_baselines": {
            name: {
                "correctness": result["correctness"],
                "benchmark": serialize_benchmark(result["benchmark"]),
            }
            for name, result in vendor_results.items()
        },
    }
    return WorkerApplication(
        auth_token=token,
        manifest=manifest,
        evaluator=evaluator,
        calibration=calibration,
    )


def capture_environment_manifest(environ=None) -> EnvironmentManifest:
    environment = os.environ if environ is None else environ
    pod_id = environment.get("RUNPOD_POD_ID")
    if not pod_id:
        raise RuntimeError("RUNPOD_POD_ID is required")
    gpu_line = _command(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,compute_cap,memory.total",
            "--format=csv,noheader,nounits",
        ]
    ).splitlines()[0]
    gpu_model, gpu_uuid, compute_capability, memory_mib = (
        item.strip() for item in gpu_line.split(",", 3)
    )
    nvcc_version = _extract_nvcc_version(_command(["nvcc", "--version"]))
    toolchains = {
        "nvcc": nvcc_version,
        "cuda_toolkit": environment.get("CUDA_VERSION", "unknown"),
        "cutlass": environment.get("CUTLASS_VERSION", "unknown"),
    }
    profilers = {}
    ncu_version = _optional_command(["ncu", "--version"])
    if ncu_version:
        profilers["ncu"] = ncu_version
    cuobjdump_version = _optional_command(["cuobjdump", "--version"])
    if cuobjdump_version:
        toolchains["cuobjdump"] = cuobjdump_version
    gpu_count = int(environment.get("RUNPOD_GPU_COUNT", "1"))
    return EnvironmentManifest(
        worker_id=pod_id,
        provider="runpod",
        pod_id=pod_id,
        gpu_model=gpu_model,
        gpu_count=gpu_count,
        gpu_uuid=gpu_uuid,
        compute_capability=compute_capability,
        total_memory_bytes=int(float(memory_mib) * 1024 * 1024),
        form_factor=_infer_form_factor(gpu_model),
        captured_at=datetime.now(timezone.utc).isoformat(),
        toolchain_versions=toolchains,
        profiler_versions=profilers,
        host_information={
            "pod_hostname": environment.get("RUNPOD_POD_HOSTNAME"),
            "data_center_id": environment.get("RUNPOD_DC_ID"),
        },
        container_image=environment.get("KERNEL_MCTS_CONTAINER_IMAGE"),
        container_digest=environment.get("KERNEL_MCTS_CONTAINER_DIGEST"),
        project_git_commit=environment.get("KERNEL_MCTS_GIT_COMMIT"),
        dirty_tree=False,
    )


def serve(application: WorkerApplication, host: str = "0.0.0.0", port: int = 8000) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self._dispatch()

        def do_POST(self):
            self._dispatch()

        def _dispatch(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length else b""
            response = application.handle(self.command, self.path, self.headers, body)
            encoded = json.dumps(response.payload, separators=(",", ":")).encode()
            self.send_response(response.status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format, *args):
            return

    HTTPServer((host, port), Handler).serve_forever()


def _command(command: list[str]) -> str:
    resolved = shutil.which(command[0]) or command[0]
    command = [resolved, *command[1:]]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        env={"PATH": os.defpath, "LANG": "C", "LC_ALL": "C"},
    )
    if result.returncode != 0:
        raise RuntimeError(f"required worker tool {command[0]!r} failed")
    return " ".join(result.stdout.split())


def _optional_command(command: list[str]) -> str | None:
    try:
        return _command(command)
    except (OSError, RuntimeError, subprocess.TimeoutExpired):
        return None


def _infer_form_factor(gpu_model: str) -> str | None:
    folded = gpu_model.casefold()
    if "pcie" in folded:
        return "PCIe"
    if "hbm3" in folded or "sxm" in folded:
        return "SXM"
    return None


def _extract_nvcc_version(output: str) -> str:
    match = re.search(r"\brelease\s+([0-9]+(?:\.[0-9]+)+)", output, re.IGNORECASE)
    if match is None:
        raise RuntimeError("could not parse nvcc version")
    return match.group(1)


def main() -> None:
    port = int(os.environ.get("KERNEL_MCTS_WORKER_PORT", "8000"))
    serve(build_application(), port=port)


if __name__ == "__main__":
    main()
