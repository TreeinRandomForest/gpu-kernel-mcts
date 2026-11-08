"""Paired H100 check of runtime versus constexpr GEMM dimensions."""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path
from uuid import uuid4

from kernel_mcts.benchmarks import BF16_GEMM_WORKLOAD
from kernel_mcts.domain import KernelProgram, ProposalStatus
from kernel_mcts.nebius import NebiusConfig, create_nebius_provider
from kernel_mcts.nebius_overlay import overlay_nebius_worker, package_bundle
from kernel_mcts.providers import HardwareSpec
from kernel_mcts.provenance import capture_repository_state, image_digest_from_reference
from kernel_mcts.serialization import serialize_evaluation


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = HERE / "constexpr-paired-results.json"
IMAGE = "docker.io/saarora/gpu-kernel-mcts:triton-gluon-3.9-cluster-v1"
SOURCES = {
    "runtime": HERE / "v19-packaged-root-bgen14-seed0-best.py",
    "constexpr": HERE / "v19-packaged-root-bgen14-seed0-best-constexpr.py",
}
ORDERS = (("runtime", "constexpr"), ("constexpr", "runtime"),
          ("runtime", "constexpr"))


def save(report: dict[str, object]) -> None:
    temporary = RESULTS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(RESULTS)


def main() -> None:
    if RESULTS.exists():
        raise FileExistsError(f"refusing to overwrite {RESULTS}")
    source = {name: path.read_text() for name, path in SOURCES.items()}
    expected = source["runtime"].replace(
        "M, N, K,\n", "M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,\n", 1)
    if expected != source["constexpr"]:
        raise ValueError("candidate must differ only in M/N/K constexpr annotations")
    repository = capture_repository_state(ROOT)
    bundle = package_bundle(ROOT)
    report: dict[str, object] = {
        "schema_version": 1, "image": IMAGE,
        "git_commit": repository.commit, "dirty_tree": repository.dirty,
        "overlay_sha256": hashlib.sha256(bundle).hexdigest(),
        "source_sha256": {name: hashlib.sha256(text.encode()).hexdigest()
                          for name, text in source.items()},
        "orders": ORDERS, "observations": [], "complete": False,
    }
    config = NebiusConfig(
        image=IMAGE, project_id="project-e00aymjmpr00grzcky46ba",
        subnet_id="vpcsubnet-e00a30bhyztjvq7yaz",
        ssh_public_key=Path.home() / ".ssh/nebius.pub",
        ssh_private_key=Path.home() / ".ssh/nebius", username="sanjay",
        startup_timeout_seconds=1200, backend="triton_h100",
        project_git_commit=repository.commit,
        project_dirty_tree=repository.dirty,
        container_digest=image_digest_from_reference(IMAGE),
    )
    hardware = HardwareSpec("H100", form_factor="SXM",
                            minimum_compute_capability="9.0",
                            required_profilers=("ncu",))
    provider = create_nebius_provider(
        config, readiness_progress=lambda stage, elapsed: print(
            f"worker {stage}: {elapsed:.1f}s", flush=True))
    save(report)
    try:
        with provider.worker_for_run(hardware) as worker:
            overlay_nebius_worker(worker, config.ssh_private_key, config.username, bundle)
            report["environment_manifest"] = worker.get_environment_manifest().as_dict()
            report["startup_calibration"] = worker.get_calibration()
            save(report)
            for round_index, names in enumerate(ORDERS):
                for name in names:
                    result = worker.evaluate(
                        f"constexpr-paired-{round_index}-{name}-{uuid4()}",
                        KernelProgram(source[name], backend="triton_h100"),
                        BF16_GEMM_WORKLOAD, "tier0",
                    )
                    payload = serialize_evaluation(result)
                    row = {
                        "round": round_index, "name": name,
                        "status": payload["status"],
                        "compile_status": payload["compile_status"],
                        "correctness_status": payload["correctness_status"],
                        "benchmark": payload.get("benchmark"),
                        "evaluation": payload,
                    }
                    report["observations"].append(row)
                    save(report)
                    print(f"round {round_index} {name}: {row['status']} "
                          f"{(row['benchmark'] or {}).get('median_us')}", flush=True)
                    if result.status != ProposalStatus.VALID:
                        raise RuntimeError(f"{name} failed in round {round_index}")
            by_round = {(row["round"], row["name"]): row["benchmark"]["median_us"]
                        for row in report["observations"]}
            deltas = [by_round[(i, "constexpr")] - by_round[(i, "runtime")]
                      for i in range(len(ORDERS))]
            report["paired_constexpr_minus_runtime_us"] = deltas
            report["median_paired_delta_us"] = statistics.median(deltas)
            report["complete"] = True
            save(report)
    finally:
        save(report)


if __name__ == "__main__":
    main()
