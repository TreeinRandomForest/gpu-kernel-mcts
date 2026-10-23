"""Reproduce the single-worker H100 CTA-placement/cache-policy crossover."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from uuid import uuid4

from kernel_mcts.benchmarks import BF16_GEMM_WORKLOAD
from kernel_mcts.domain import KernelProgram, ProposalStatus
from kernel_mcts.nebius import NebiusConfig, create_nebius_provider
from kernel_mcts.providers import HardwareSpec
from kernel_mcts.serialization import serialize_evaluation


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT / "sequential-sol-choice-freeform-bgen50-seed0-best.cu"
IMAGE = (
    "docker.io/saarora/gpu-kernel-mcts@sha256:"
    "f90da644bcabc3a07756ecea933f4e1bc2d7faa5785e1c5d39c851eecedb0429"
)
COMPACT = """    const int linear_block =
        (static_cast<int>(blockIdx.y) << 8) + static_cast<int>(blockIdx.x);
    if (linear_block >= 1024) {
        return;
    }

    const int tx = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);
    const int tid = (ty << 4) + tx;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    const int macro_row = (linear_block >> 5) << 7;
    const int macro_col = (linear_block & 31) << 7;"""
SPARSE = """    if ((blockIdx.x & 7u) != 0u || (blockIdx.y & 7u) != 0u) {
        return;
    }

    const int tx = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);
    const int tid = (ty << 4) + tx;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    const int macro_row = static_cast<int>(blockIdx.y) << 4;
    const int macro_col = static_cast<int>(blockIdx.x) << 4;"""
ORDER = (
    ("compact_cg", "sparse_ca", "sparse_cg", "compact_ca"),
    ("compact_ca", "sparse_cg", "sparse_ca", "compact_cg"),
    ("sparse_cg", "compact_cg", "compact_ca", "sparse_ca"),
    ("sparse_ca", "compact_ca", "compact_cg", "sparse_cg"),
)


def source_variants() -> dict[str, str]:
    base = BASE.read_text(encoding="utf-8")
    if base.count(COMPACT) != 1 or base.count("cp.async.ca.shared.global") != 1:
        raise ValueError("recorded best source no longer matches the experiment patch")
    result = {}
    for placement in ("compact", "sparse"):
        for policy in ("ca", "cg"):
            source = base if placement == "compact" else base.replace(COMPACT, SPARSE)
            if policy == "cg":
                source = source.replace(
                    "cp.async.ca.shared.global", "cp.async.cg.shared.global"
                )
            result[f"{placement}_{policy}"] = source
    return result


def save(report: dict[str, object]) -> None:
    path = HERE / "results.json"
    temporary = HERE / "results.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    variants = source_variants()
    for name, source in variants.items():
        (HERE / f"{name}.cu").write_text(source, encoding="utf-8")
    if (HERE / "results.json").exists():
        raise FileExistsError("results.json already exists")

    report: dict[str, object] = {
        "base": str(BASE.relative_to(ROOT)),
        "base_sha256": hashlib.sha256(BASE.read_bytes()).hexdigest(),
        "image": IMAGE,
        "order": ORDER,
        "variants": {
            name: {"sha256": hashlib.sha256(source.encode()).hexdigest()}
            for name, source in variants.items()
        },
        "measurements": [],
        "profiles": {},
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
    evaluation_ids: dict[str, str] = {}
    try:
        with provider.worker_for_run(hardware) as worker:
            report["environment_manifest"] = worker.get_environment_manifest().as_dict()
            save(report)
            for cycle, names in enumerate(ORDER):
                for name in names:
                    evaluation_id = f"crossover-{cycle}-{name}-{uuid4()}"
                    result = worker.evaluate(
                        evaluation_id,
                        KernelProgram(variants[name], backend="cuda_cpp"),
                        BF16_GEMM_WORKLOAD,
                        "tier0",
                    )
                    serialized = serialize_evaluation(result)
                    benchmark = serialized.get("benchmark") or {}
                    entry = {
                        "cycle": cycle,
                        "variant": name,
                        "evaluation_id": evaluation_id,
                        "status": serialized["status"],
                        "compile_status": serialized["compile_status"],
                        "correctness_status": serialized["correctness_status"],
                        "correctness": serialized.get("correctness"),
                        "benchmark": benchmark,
                        "compiler_context": result.metadata.get("compiler_context"),
                    }
                    report["measurements"].append(entry)
                    save(report)
                    print(
                        f"cycle {cycle} {name}: {entry['status']} "
                        f"median={benchmark.get('median_us')}", flush=True
                    )
                    if result.status != ProposalStatus.VALID:
                        raise RuntimeError(f"{name} failed verification; aborting crossover")
                    if name not in evaluation_ids:
                        evaluation_ids[name] = evaluation_id
                        context = result.metadata.get("compiler_context", {})
                        ptx = context.get("ptx", {}) if isinstance(context, dict) else {}
                        for artifact in ptx.get("artifacts", []):
                            if artifact.get("storage_status") == "eligible":
                                raw = worker.fetch_compiler_artifact(
                                    evaluation_id, artifact["sha256"]
                                )
                                (HERE / f"{name}.ptx").write_bytes(raw)
            for name, evaluation_id in evaluation_ids.items():
                report["profiles"][name] = worker.profile(
                    evaluation_id, "lightweight", "lightweight_v1"
                )
                save(report)
                print(f"profile {name}: complete", flush=True)
            report["complete"] = True
            save(report)
    finally:
        save(report)


if __name__ == "__main__":
    main()
