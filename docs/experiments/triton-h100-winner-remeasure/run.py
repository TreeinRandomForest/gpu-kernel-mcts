"""Repeat the saved Triton winner and vendor references on one Nebius H100."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import statistics
import subprocess
from pathlib import Path
from uuid import uuid4

from kernel_mcts.benchmarks import BF16_GEMM_WORKLOAD, load_triton_bf16_gemm_root
from kernel_mcts.domain import KernelProgram, ProposalStatus
from kernel_mcts.nebius import NebiusConfig, create_nebius_provider
from kernel_mcts.providers import HardwareSpec
from kernel_mcts.provenance import capture_repository_state, image_digest_from_reference
from kernel_mcts.serialization import serialize_evaluation


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
WINNER = ROOT / "docs/experiments/triton-h100-glm53-bgen3-best.py"
RESULTS = HERE / "results.json"
IMAGE = (
    "docker.io/saarora/gpu-kernel-mcts@sha256:"
    "d0cd8e7b784ac8828691aa0add0e41d1802e22ef11745780de018afaa12c0469"
)
ORDER = (
    ("root", "cublas", "winner", "cutlass"),
    ("winner", "cutlass", "root", "cublas"),
    ("cutlass", "winner", "cublas", "root"),
    ("cublas", "root", "cutlass", "winner"),
)


def save(report: dict[str, object]) -> None:
    temporary = RESULTS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(RESULTS)


def instance_public_ip(instance_id: str) -> str:
    result = subprocess.run(
        ["nebius", "compute", "instance", "get", "--id", instance_id,
         "--format", "json", "--no-check-update"],
        capture_output=True, text=True, check=True, timeout=30,
    )
    value = json.loads(result.stdout)
    for interface in value.get("status", {}).get("network_interfaces", []):
        address = interface.get("public_ip_address", {}).get("address")
        if isinstance(address, str):
            return str(ipaddress.ip_interface(address).ip)
    raise RuntimeError("Nebius instance has no public IP")


def vendor_round(ip: str, implementation: str) -> dict[str, object]:
    command = [
        "ssh", "-F", "/dev/null", "-i", str(Path.home() / ".ssh/nebius"),
        "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=10",
        f"sanjay@{ip}", "sudo", "docker", "exec", "kernel-mcts-worker",
        "/tmp/kernel-mcts-artifacts/bf16-gemm-vendor-baselines",
        f"--implementation={implementation}", "--M=4096", "--N=4096", "--K=4096",
        "--rtol=0.02", "--atol=0.02", "--seed=0", "--warmups=10",
        "--measurements=30",
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=180)
    if result.returncode != 0:
        raise RuntimeError(f"{implementation} remote benchmark failed with exit {result.returncode}")
    payload = json.loads(result.stdout)
    timings = payload.get("timings_us")
    if payload.get("status") != "ok" or payload.get("success") is not True:
        raise RuntimeError(f"{implementation} failed correctness")
    if not isinstance(timings, list) or len(timings) != 30:
        raise RuntimeError(f"{implementation} returned an invalid timing sample count")
    return payload


def main() -> None:
    if RESULTS.exists():
        raise FileExistsError(f"refusing to overwrite {RESULTS}")
    winner_source = WINNER.read_text(encoding="utf-8")
    root_program = load_triton_bf16_gemm_root()
    programs = {
        "root": root_program,
        "winner": KernelProgram(winner_source, backend="triton_h100"),
    }
    repository = capture_repository_state(ROOT)
    report: dict[str, object] = {
        "run_id": str(uuid4()),
        "image": IMAGE,
        "git_commit": repository.commit,
        "dirty_tree": repository.dirty,
        "source_hashes": {
            name: hashlib.sha256(program.source.encode()).hexdigest()
            for name, program in programs.items()
        },
        "order": ORDER,
        "samples_per_round": 30,
        "warmups_per_round": 10,
        "seed": 0,
        "observations": [],
        "complete": False,
    }
    config = NebiusConfig(
        image=IMAGE,
        project_id="project-e00aymjmpr00grzcky46ba",
        subnet_id="vpcsubnet-e00a30bhyztjvq7yaz",
        ssh_public_key=Path.home() / ".ssh/nebius.pub",
        ssh_private_key=Path.home() / ".ssh/nebius",
        username="sanjay",
        startup_timeout_seconds=1200,
        backend="triton_h100",
        project_git_commit=repository.commit,
        project_dirty_tree=repository.dirty,
        container_digest=image_digest_from_reference(IMAGE),
    )
    hardware = HardwareSpec(
        "H100", form_factor="SXM", minimum_compute_capability="9.0",
        required_profilers=("ncu",),
    )
    provider = create_nebius_provider(
        config,
        readiness_progress=lambda stage, elapsed: print(
            f"worker {stage}: {elapsed:.1f}s", flush=True
        ),
    )
    save(report)
    try:
        with provider.worker_for_run(hardware) as worker:
            report["environment_manifest"] = worker.get_environment_manifest().as_dict()
            report["startup_calibration"] = worker.get_calibration()
            save(report)
            ip = instance_public_ip(worker.instance.instance_id)
            for cycle, names in enumerate(ORDER):
                for name in names:
                    if name in programs:
                        evaluation_id = f"triton-remeasure-{cycle}-{name}-{uuid4()}"
                        result = worker.evaluate(
                            evaluation_id, programs[name], BF16_GEMM_WORKLOAD, "tier0"
                        )
                        if result.status != ProposalStatus.VALID:
                            raise RuntimeError(f"{name} failed verification in round {cycle}")
                        payload = serialize_evaluation(result)
                        observation = {
                            "round": cycle, "implementation": name,
                            "evaluation_id": evaluation_id,
                            "status": payload["status"],
                            "compile_status": payload["compile_status"],
                            "correctness_status": payload["correctness_status"],
                            "correctness": payload.get("correctness"),
                            "benchmark": payload["benchmark"],
                        }
                        median = observation["benchmark"]["median_us"]
                    else:
                        payload = vendor_round(ip, name)
                        observation = {
                            "round": cycle, "implementation": name,
                            "status": payload["status"],
                            "correctness": {
                                "success": payload["success"],
                                "maximum_error": payload["maximum_error"],
                                "mean_error": payload["mean_error"],
                            },
                            "timings_us": payload["timings_us"],
                        }
                        median = statistics.median(payload["timings_us"])
                        observation["median_us"] = median
                    report["observations"].append(observation)
                    save(report)
                    print(f"round {cycle} {name}: median={median:.3f} us", flush=True)
            report["complete"] = True
            save(report)
    finally:
        save(report)


if __name__ == "__main__":
    main()
