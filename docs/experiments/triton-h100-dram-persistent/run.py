"""Run matched-input persistent-TMA traffic comparison on one H100."""

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
TMA = ROOT / "docs/experiments/triton-h100-tma-load/tma_kernel.py"
PERSISTENT = HERE / "persistent_kernel.py"
HARNESS = HERE / "harness.py"
RESULTS = HERE / "results.json"
OUTPUT = HERE / "output"
REMOTE_TMA = "/tmp/dram-persistent-tma-kernel.py"
REMOTE_PERSISTENT = "/tmp/dram-persistent-kernel.py"
REMOTE_HARNESS = "/tmp/dram-persistent-harness.py"
IMAGE = (
    "docker.io/saarora/gpu-kernel-mcts@sha256:"
    "d0cd8e7b784ac8828691aa0add0e41d1802e22ef11745780de018afaa12c0469"
)
NAMES = ("nonpersistent_8", "persistent_1", "persistent_8", "persistent_32", "cublas")
METRICS = (
    "dram__bytes_read.sum",
    "dram__sectors_read.sum",
    "lts__t_sectors_srcunit_tex_op_read.sum",
    "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum",
    "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum",
    "lts__t_sector_op_read_hit_rate.pct",
    "launch__grid_size",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block",
)


def _load_helpers():
    path = ROOT / "docs/experiments/triton-h100-cublas-sass/run.py"
    spec = importlib.util.spec_from_file_location("matched_sass_helpers", path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot import Nebius runner helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(report: dict) -> None:
    temporary = RESULTS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(RESULTS)


def target_command(mode: str, name: str | None = None) -> list[str]:
    command = ["python", REMOTE_HARNESS, "--tma", REMOTE_TMA,
               "--persistent", REMOTE_PERSISTENT, "--mode", mode]
    if name is not None:
        command += ["--target", name]
    return command


def profile_command(name: str, prefix: str) -> list[str]:
    if name not in NAMES:
        raise ValueError("unknown profile arm")
    return ["ncu", "--metrics", ",".join(METRICS),
            "--cache-control", "all", "--replay-mode", "application",
            "--nvtx", "--nvtx-include", f"DRAM_PERSISTENT_{name.upper()}/",
            "--launch-count", "1", "--export", prefix,
            *target_command("profile", name)]


def main() -> None:
    if RESULTS.exists():
        raise FileExistsError(f"refusing to overwrite {RESULTS}")
    files = {REMOTE_TMA: TMA.read_bytes(), REMOTE_PERSISTENT: PERSISTENT.read_bytes(),
             REMOTE_HARNESS: HARNESS.read_bytes()}
    prior = json.loads((TMA.parent / "results.json").read_text())
    if hashlib.sha256(files[REMOTE_TMA]).hexdigest() != prior["source_sha256"]["tma"]:
        raise ValueError("the measured TMA source has changed")
    source_hashes = {"tma": hashlib.sha256(files[REMOTE_TMA]).hexdigest(),
                     "persistent": hashlib.sha256(files[REMOTE_PERSISTENT]).hexdigest()}
    repository = capture_repository_state(ROOT)
    report = {"schema_version": 1, "run_id": str(uuid4()), "image": IMAGE,
              "git_commit": repository.commit, "dirty_tree": repository.dirty,
              "source_sha256": source_hashes,
              "harness_sha256": hashlib.sha256(files[REMOTE_HARNESS]).hexdigest(),
              "profile_metrics": METRICS, "cache_control": "all",
              "benchmark": None, "profiles": {}, "complete": False}
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
    helpers = _load_helpers()
    OUTPUT.mkdir(exist_ok=True)
    save(report)
    try:
        with provider.worker_for_run(hardware) as worker:
            report["environment_manifest"] = worker.get_environment_manifest().as_dict()
            save(report)
            ssh = helpers.ssh_prefix(helpers.instance_public_ip(worker.instance.instance_id))
            for remote, payload in files.items():
                helpers.upload(ssh, remote, payload)
            print("running matched-input timing and correctness", flush=True)
            benchmark = json.loads(helpers.checked(
                [*ssh, *target_command("benchmark")], timeout=360))
            if (benchmark.get("status") != "ok"
                    or benchmark.get("source_sha256") != source_hashes
                    or len(benchmark.get("observations", [])) != 15):
                raise RuntimeError("benchmark failed validation")
            report["benchmark"] = benchmark
            save(report)
            for item in benchmark["observations"]:
                print(item["round"], item["implementation"],
                      round(item["median_us"], 3), flush=True)
            for name in NAMES:
                prefix = f"/tmp/dram-persistent-{name}"
                print("profiling", name, flush=True)
                helpers.checked([*ssh, *profile_command(name, prefix)], timeout=900)
                remote_report = prefix + ".ncu-rep"
                report_bytes = helpers.checked([*ssh, "cat", remote_report], timeout=180)
                (OUTPUT / f"{name}.ncu-rep").write_bytes(report_bytes)
                raw = helpers.checked([*ssh, "ncu", "--import", remote_report,
                                       "--page", "raw", "--csv"], timeout=180)
                (OUTPUT / f"{name}.raw.csv").write_bytes(raw)
                report["profiles"][name] = {
                    "ncu_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
                    "ncu_report_size_bytes": len(report_bytes),
                }
                save(report)
            report["complete"] = True
            save(report)
    finally:
        save(report)


if __name__ == "__main__":
    main()
