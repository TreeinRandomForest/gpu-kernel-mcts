"""Provision pinned H100 and test Triton two-CTA TMA lowering."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from uuid import uuid4

from kernel_mcts.nebius import NebiusConfig, create_nebius_provider
from kernel_mcts.providers import HardwareSpec
from kernel_mcts.provenance import capture_repository_state, image_digest_from_reference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / "docs/experiments/triton-h100-tma-load/tma_kernel.py"
PROBE = HERE / "probe.py"
RESULTS = HERE / "probe_results.json"
OUTPUT = HERE / "output"
IMAGE = (
    "docker.io/saarora/gpu-kernel-mcts@sha256:"
    "d0cd8e7b784ac8828691aa0add0e41d1802e22ef11745780de018afaa12c0469"
)
REMOTE_SOURCE = "/tmp/cluster-tma-kernel.py"
REMOTE_PROBE = "/tmp/cluster-tma-probe.py"
REMOTE_OUTPUT = "/tmp/cluster-tma-probe-output"
VARIANTS = ("baseline", "cluster_same_tile", "cluster_m", "cluster_n")


def load_helpers():
    path = ROOT / "docs/experiments/triton-h100-cublas-sass/run.py"
    spec = importlib.util.spec_from_file_location("cluster_probe_helpers", path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot import worker helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(report: dict) -> None:
    temporary = RESULTS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(RESULTS)


def main() -> None:
    if RESULTS.exists():
        raise FileExistsError(f"refusing to overwrite {RESULTS}")
    source, probe = SOURCE.read_bytes(), PROBE.read_bytes()
    previous = json.loads((SOURCE.parent / "results.json").read_text())
    if hashlib.sha256(source).hexdigest() != previous["source_sha256"]["tma"]:
        raise ValueError("TMA baseline changed")
    repository = capture_repository_state(ROOT)
    report = {"schema_version": 1, "run_id": str(uuid4()), "image": IMAGE,
              "git_commit": repository.commit, "dirty_tree": repository.dirty,
              "source_sha256": hashlib.sha256(source).hexdigest(),
              "probe_sha256": hashlib.sha256(probe).hexdigest(),
              "variants": {}, "complete": False}
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
    helpers = load_helpers()
    OUTPUT.mkdir(exist_ok=True)
    save(report)
    try:
        with provider.worker_for_run(hardware) as worker:
            report["environment_manifest"] = worker.get_environment_manifest().as_dict()
            save(report)
            ssh = helpers.ssh_prefix(helpers.instance_public_ip(worker.instance.instance_id))
            helpers.upload(ssh, REMOTE_SOURCE, source)
            helpers.upload(ssh, REMOTE_PROBE, probe)
            for name in VARIANTS:
                print("probing", name, flush=True)
                command = [*ssh, "python", REMOTE_PROBE, "--source", REMOTE_SOURCE,
                           "--variant", name, "--output-dir", REMOTE_OUTPUT]
                try:
                    result = json.loads(helpers.checked(command, timeout=300))
                except Exception as error:
                    result = {"status": "runner_failed", "variant": name,
                              "error": str(error)[-1000:]}
                report["variants"][name] = result
                if result.get("status") == "ok":
                    ptx = helpers.checked([*ssh, "cat", f"{REMOTE_OUTPUT}/{name}.ptx"],
                                          timeout=120)
                    if hashlib.sha256(ptx).hexdigest() != result["ptx_sha256"]:
                        raise ValueError("PTX hash mismatch")
                    (OUTPUT / f"{name}.ptx").write_bytes(ptx)
                save(report)
                print(name, result["status"],
                      "multicast", result.get("multicast_mentions"), flush=True)
            report["complete"] = True
            save(report)
    finally:
        save(report)


if __name__ == "__main__":
    main()
