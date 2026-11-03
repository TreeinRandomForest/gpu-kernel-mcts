"""Same-worker source crossover for exact seed-0 B/D winners and two hybrids."""

from __future__ import annotations

import hashlib
import io
import json
import random
import shlex
import subprocess
import tarfile
import time
from pathlib import Path
from uuid import uuid4

from kernel_mcts.benchmarks import BF16_GEMM_WORKLOAD
from kernel_mcts.domain import KernelProgram, ProposalStatus
from kernel_mcts.nebius import NebiusConfig, create_nebius_provider
from kernel_mcts.providers import HardwareSpec
from kernel_mcts.provenance import capture_repository_state, image_digest_from_reference
from kernel_mcts.serialization import serialize_evaluation

from variants import HERE, sources, write_sources


ROOT = HERE.parents[2]
RESULTS = HERE / "results.json"
IMAGE = (
    "docker.io/saarora/gpu-kernel-mcts@sha256:"
    "d0cd8e7b784ac8828691aa0add0e41d1802e22ef11745780de018afaa12c0469"
)
ROUNDS = 3


def save(report: dict) -> None:
    temporary = RESULTS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(RESULTS)


def checked(command: list[str], *, input_bytes: bytes | None = None,
            timeout: float = 180) -> bytes:
    result = subprocess.run(command, input=input_bytes, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): "
                           + result.stderr.decode(errors="replace")[-1200:])
    return result.stdout


def package_bundle() -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        archive.add(ROOT / "src/kernel_mcts", arcname="kernel_mcts",
                    filter=lambda info: None if "__pycache__" in info.name else info)
    return stream.getvalue()


def overlay_worker(worker, ssh_private_key: Path, bundle: bytes) -> None:
    instance = json.loads(checked([
        "nebius", "compute", "instance", "get", "--id", worker.instance.instance_id,
        "--format", "json", "--no-check-update",
    ], timeout=30))
    ip = instance["status"]["network_interfaces"][0]["public_ip_address"]["address"].split("/")[0]
    ssh = ["ssh", "-F", "/dev/null", "-i", str(ssh_private_key),
           "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
           f"sanjay@{ip}"]
    package_path = checked([*ssh, shlex.join([
        "sudo", "docker", "exec", "kernel-mcts-worker", "python", "-c",
        "import kernel_mcts,pathlib; print(pathlib.Path(kernel_mcts.__file__).parent.parent)",
    ])]).decode().strip()
    if not package_path.startswith("/opt/venv/") or not package_path.endswith("site-packages"):
        raise RuntimeError("unexpected worker package location")
    checked([*ssh, "sudo", "docker", "exec", "-i", "kernel-mcts-worker", "tar", "-xz",
             "-C", package_path], input_bytes=bundle)
    checked([*ssh, "sudo", "docker", "restart", "kernel-mcts-worker"], timeout=120)
    for _ in range(30):
        try:
            worker._transport.get_environment_manifest()
            return
        except Exception:
            time.sleep(2)
    raise RuntimeError("overlaid worker did not become ready")


def main() -> None:
    if RESULTS.exists():
        raise FileExistsError(RESULTS)
    manifest = json.loads((HERE / "trace_manifest.json").read_text())
    prepared = write_sources()
    programs = sources()
    for arm in manifest["arms"]:
        name = f'{arm["arm"]}_exact'
        if prepared[name]["sha256"] != arm["source_sha256"]:
            raise ValueError(f"exact trace source changed: {name}")
    repository = capture_repository_state(ROOT)
    bundle = package_bundle()
    report = {
        "schema_version": 1, "run_id": str(uuid4()),
        "image": IMAGE, "git_commit": repository.commit,
        "dirty_tree": repository.dirty,
        "source_bundle_sha256": hashlib.sha256(bundle).hexdigest(),
        "trace_manifest": manifest, "sources": prepared,
        "rounds": ROUNDS, "observations": [], "complete": False,
    }
    config = NebiusConfig(
        image=IMAGE, project_id="project-e00aymjmpr00grzcky46ba",
        subnet_id="vpcsubnet-e00a30bhyztjvq7yaz",
        ssh_public_key=Path.home() / ".ssh/nebius.pub",
        ssh_private_key=Path.home() / ".ssh/nebius", username="sanjay",
        startup_timeout_seconds=1200, backend="triton_h100",
        project_git_commit=repository.commit, project_dirty_tree=repository.dirty,
        container_digest=image_digest_from_reference(IMAGE),
    )
    provider = create_nebius_provider(
        config, readiness_progress=lambda stage, elapsed: print(
            f"worker {stage}: {elapsed:.1f}s", flush=True),
    )
    hardware = HardwareSpec("H100", form_factor="SXM",
                            minimum_compute_capability="9.0", required_profilers=("ncu",))
    save(report)
    try:
        with provider.worker_for_run(hardware) as worker:
            overlay_worker(worker, config.ssh_private_key, bundle)
            report["environment_manifest"] = worker.get_environment_manifest().as_dict()
            report["startup_calibration"] = worker.get_calibration()
            save(report)
            for round_index in range(ROUNDS):
                order = list(programs)
                random.Random(100 + round_index).shuffle(order)
                for name in order:
                    evaluation_id = f"bd-crossover-r{round_index}-{name}-{uuid4()}"
                    result = worker.evaluate(
                        evaluation_id, KernelProgram(programs[name], backend="triton_h100"),
                        BF16_GEMM_WORKLOAD, "tier0",
                    )
                    if result.status != ProposalStatus.VALID:
                        raise RuntimeError(f"{name} failed evaluation: {result.status.value}")
                    payload = serialize_evaluation(result)
                    context = result.metadata.get("compiler_context")
                    if not isinstance(context, dict):
                        raise RuntimeError(f"{name} has no compiler context")
                    artifacts = context.get("ptx", {}).get("artifacts", [])
                    if len(artifacts) != 1 or artifacts[0].get("storage_status") != "eligible":
                        raise RuntimeError(f"{name} has no fetchable PTX")
                    ptx_hash = artifacts[0]["sha256"]
                    ptx = worker.fetch_compiler_artifact(evaluation_id, ptx_hash)
                    if hashlib.sha256(ptx).hexdigest() != ptx_hash:
                        raise RuntimeError(f"{name} fetched PTX hash mismatch")
                    path = HERE / f"{name}.ptx"
                    if path.exists() and path.read_bytes() != ptx:
                        raise RuntimeError(f"{name} PTX changed across rounds")
                    path.write_bytes(ptx)
                    observation = {
                        "round": round_index, "order": order.index(name),
                        "variant": name, "evaluation_id": evaluation_id,
                        "source_sha256": prepared[name]["sha256"],
                        "ptx_sha256": ptx_hash,
                        "binary_fingerprint": payload.get("binary_hash"),
                        "benchmark": payload["benchmark"],
                        "correctness": payload.get("correctness"),
                        "compiler_resources": context.get("resources"),
                    }
                    if round_index == 0:
                        observation["profile"] = worker.profile(
                            evaluation_id, "lightweight", "spill_v1"
                        )
                    report["observations"].append(observation)
                    save(report)
                    print(round_index, name, round(payload["benchmark"]["median_us"], 3),
                          context.get("resources"), flush=True)
            report["complete"] = True
            save(report)
    finally:
        save(report)


if __name__ == "__main__":
    main()
