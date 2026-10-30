"""Capture resources, PTX, and lightweight profiles for three Triton variants."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from uuid import uuid4

from kernel_mcts.benchmarks import BF16_GEMM_WORKLOAD
from kernel_mcts.domain import KernelProgram, ProposalStatus
from kernel_mcts.nebius import NebiusConfig, create_nebius_provider
from kernel_mcts.providers import HardwareSpec
from kernel_mcts.provenance import capture_repository_state, image_digest_from_reference
from kernel_mcts.serialization import serialize_evaluation


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CROSSOVER = ROOT / "docs/experiments/triton-h100-winner-crossover"
RESULTS = HERE / "results.json"
VARIANTS = (
    "grouped_bn128_w4",
    "grouped_bn256_w4",
    "grouped_bn256_w8",
)
IMAGE = (
    "docker.io/saarora/gpu-kernel-mcts@sha256:"
    "d0cd8e7b784ac8828691aa0add0e41d1802e22ef11745780de018afaa12c0469"
)


def save(report: dict[str, object]) -> None:
    temporary = RESULTS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(RESULTS)


def main() -> None:
    if RESULTS.exists():
        raise FileExistsError(f"refusing to overwrite {RESULTS}")
    sources = {
        name: (CROSSOVER / f"{name}.py").read_text(encoding="utf-8")
        for name in VARIANTS
    }
    crossover = json.loads((CROSSOVER / "results.json").read_text(encoding="utf-8"))
    for name, source in sources.items():
        expected = crossover["variants"][name]["source_sha256"]
        if hashlib.sha256(source.encode()).hexdigest() != expected:
            raise ValueError(f"source hash mismatch for {name}")
    repository = capture_repository_state(ROOT)
    report: dict[str, object] = {
        "run_id": str(uuid4()),
        "image": IMAGE,
        "git_commit": repository.commit,
        "dirty_tree": repository.dirty,
        "variants": {name: crossover["variants"][name] for name in VARIANTS},
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
            for name in VARIANTS:
                evaluation_id = f"triton-warp-cliff-{name}-{uuid4()}"
                result = worker.evaluate(
                    evaluation_id,
                    KernelProgram(sources[name], backend="triton_h100"),
                    BF16_GEMM_WORKLOAD,
                    "tier0",
                )
                if result.status != ProposalStatus.VALID:
                    raise RuntimeError(f"{name} failed evaluation: {result.status.value}")
                context = result.metadata.get("compiler_context")
                if not isinstance(context, dict):
                    raise RuntimeError(f"{name} has no compiler context")
                artifacts = context.get("ptx", {}).get("artifacts", [])
                if len(artifacts) != 1 or artifacts[0].get("storage_status") != "eligible":
                    raise RuntimeError(f"{name} has no fetchable PTX")
                sha256 = artifacts[0]["sha256"]
                ptx = worker.fetch_compiler_artifact(evaluation_id, sha256)
                if hashlib.sha256(ptx).hexdigest() != sha256:
                    raise RuntimeError(f"{name} PTX hash mismatch")
                (HERE / f"{name}.ptx").write_bytes(ptx)
                source_ptx = ptx.decode("utf-8")
                payload = serialize_evaluation(result)
                observation = {
                    "variant": name,
                    "evaluation_id": evaluation_id,
                    "status": payload["status"],
                    "compile_status": payload["compile_status"],
                    "correctness_status": payload["correctness_status"],
                    "correctness": payload.get("correctness"),
                    "benchmark": payload.get("benchmark"),
                    "compiler_context": context,
                    "ptx_instruction_sites": {
                        "ld_local": source_ptx.count("ld.local"),
                        "st_local": source_ptx.count("st.local"),
                        "cp_async": source_ptx.count("cp.async"),
                        "wgmma": source_ptx.count("wgmma"),
                    },
                }
                report["observations"].append(observation)
                save(report)
                print(
                    f"{name}: median={payload['benchmark']['median_us']:.3f} us "
                    f"resources={context.get('resources')} "
                    f"PTX-local={observation['ptx_instruction_sites']['ld_local']}/"
                    f"{observation['ptx_instruction_sites']['st_local']}",
                    flush=True,
                )
                observation["profile"] = worker.profile(
                    evaluation_id, "lightweight", "lightweight_v1"
                )
                save(report)
                print(f"{name}: lightweight profile complete", flush=True)
            report["complete"] = True
            save(report)
    finally:
        save(report)


if __name__ == "__main__":
    main()
