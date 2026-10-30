"""Selected-node NCU Source validation on two saved H100 Triton variants."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from uuid import uuid4

from kernel_mcts.benchmarks import BF16_GEMM_WORKLOAD
from kernel_mcts.domain import KernelProgram, ProposalStatus
from kernel_mcts.nebius import NebiusConfig, create_nebius_provider
from kernel_mcts.profiling import resolve_profile_metric_set
from kernel_mcts.providers import HardwareSpec
from kernel_mcts.provenance import capture_repository_state, image_digest_from_reference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CROSSOVER = ROOT / "docs/experiments/triton-h100-winner-crossover"
COLLECTOR = ROOT / "src/kernel_mcts/ncu_source_report.py"
RESULTS = HERE / "results.json"
VARIANTS = ("grouped_bn256_w4", "grouped_bn256_w8")
IMAGE = (
    "docker.io/saarora/gpu-kernel-mcts@sha256:"
    "d0cd8e7b784ac8828691aa0add0e41d1802e22ef11745780de018afaa12c0469"
)
LOCAL_METRICS = (
    "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_ld_lookup_hit.sum",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_ld_lookup_miss.sum",
)


def save(report: dict[str, object]) -> None:
    temporary = RESULTS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(RESULTS)


def checked(command: list[str], *, input_bytes: bytes | None = None,
            timeout: float = 900.0) -> bytes:
    result = subprocess.run(command, input=input_bytes, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): "
                           + result.stderr.decode(errors="replace")[-1200:])
    return result.stdout


def main() -> None:
    if RESULTS.exists():
        raise FileExistsError(RESULTS)
    crossover = json.loads((CROSSOVER / "results.json").read_text())
    sources = {name: (CROSSOVER / f"{name}.py").read_text() for name in VARIANTS}
    for name, source in sources.items():
        if hashlib.sha256(source.encode()).hexdigest() != crossover["variants"][name]["source_sha256"]:
            raise ValueError(f"source mismatch: {name}")
    repository = capture_repository_state(ROOT)
    report: dict[str, object] = {
        "run_id": str(uuid4()), "image": IMAGE, "git_commit": repository.commit,
        "dirty_tree": repository.dirty, "observations": [], "complete": False,
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
            f"worker {stage}: {elapsed:.1f}s", flush=True
        ),
    )
    hardware = HardwareSpec("H100", form_factor="SXM",
                            minimum_compute_capability="9.0", required_profilers=("ncu",))
    save(report)
    try:
        with provider.worker_for_run(hardware) as worker:
            report["environment_manifest"] = worker.get_environment_manifest().as_dict()
            report["startup_calibration"] = worker.get_calibration()
            instance = json.loads(checked([
                "nebius", "compute", "instance", "get", "--id", worker.instance.instance_id,
                "--format", "json", "--no-check-update",
            ], timeout=30))
            ip = instance["status"]["network_interfaces"][0]["public_ip_address"]["address"].split("/")[0]
            ssh = ["ssh", "-F", "/dev/null", "-i", str(config.ssh_private_key),
                   "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
                   f"{config.username}@{ip}"]
            checked([*ssh, "sudo", "docker", "exec", "-i", "kernel-mcts-worker",
                     "tee", "/tmp/ncu_source_report.py"], input_bytes=COLLECTOR.read_bytes(), timeout=30)
            query = checked([*ssh, "sudo", "docker", "exec", "kernel-mcts-worker",
                             "ncu", "--query-metrics", "--query-metrics-mode", "all"], timeout=120).decode()
            report["local_metric_availability"] = {
                name: name in query for name in LOCAL_METRICS
            }
            save(report)
            for name in VARIANTS:
                evaluation_id = f"source-report-{name}-{uuid4()}"
                result = worker.evaluate(
                    evaluation_id, KernelProgram(sources[name], backend="triton_h100"),
                    BF16_GEMM_WORKLOAD, "tier0",
                )
                if result.status != ProposalStatus.VALID:
                    raise RuntimeError(f"{name} evaluation failed: {result.status.value}")
                source_path = next(Path(path) for path in result.compilation.artifact_paths
                                   if path.endswith("candidate.py"))
                prefix = f"/tmp/ncu-source-{name}"
                target = [
                    "python", "-m", "kernel_mcts.triton_harness",
                    "--candidate", str(source_path), "--artifact-dir", str(source_path.parent),
                    "--mode", "profile", "--M=4096", "--N=4096", "--K=4096",
                    "--rtol=0.02", "--atol=0.02", "--seed=0", "--warmups=10",
                    "--measurements=30",
                ]
                print(f"{name}: collecting Source report", flush=True)
                summary = json.loads(checked([
                    *ssh, "sudo", "docker", "exec", "kernel-mcts-worker", "python",
                    "/tmp/ncu_source_report.py", "--output-prefix", prefix,
                    "--kernel-regex", ".*bf16_gemm_triton.*", "--", *target,
                ], timeout=1800))
                for extension in ("ncu-rep", "sass.csv", "ptx.csv", "cuda_sass.csv"):
                    remote = f"{prefix}.{extension}"
                    if extension != "ncu-rep" and summary["source_views"][extension.removesuffix(".csv")] != "captured":
                        continue
                    raw = checked([*ssh, "sudo", "docker", "exec", "kernel-mcts-worker",
                                   "cat", remote], timeout=120)
                    (HERE / f"{name}.{extension}").write_bytes(raw)
                metrics = resolve_profile_metric_set("diagnostic_v2", "sm_90", "2025.1")
                raw = checked([*ssh, "sudo", "docker", "exec", "kernel-mcts-worker",
                               "ncu", "--csv", "--page", "raw", "--log-file", "stdout",
                               "--kernel-name-base", "function", "--kernel-name",
                               "regex:.*bf16_gemm_triton.*", "--launch-count", "1",
                               "--metrics", ",".join(metrics.metrics), *target], timeout=900)
                (HERE / f"{name}.diagnostic.csv").write_bytes(raw)
                observation = {
                    "variant": name, "source_sha256": hashlib.sha256(sources[name].encode()).hexdigest(),
                    "median_us": result.benchmark.median_us,
                    "compiler_resources": result.metadata["compiler_context"]["resources"],
                    "source_report": summary,
                }
                report["observations"].append(observation)
                save(report)
                print(f"{name}: {summary}", flush=True)
            report["complete"] = True
            save(report)
    finally:
        save(report)


if __name__ == "__main__":
    main()
