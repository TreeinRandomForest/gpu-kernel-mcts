"""Measure a bounded 2x2x2 crossover around the saved Triton H100 winner."""

from __future__ import annotations

import ast
import hashlib
import json
import random
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
WINNER = ROOT / "docs/experiments/triton-h100-glm53-bgen3-best.py"
RESULTS = HERE / "results.json"
IMAGE = (
    "docker.io/saarora/gpu-kernel-mcts@sha256:"
    "d0cd8e7b784ac8828691aa0add0e41d1802e22ef11745780de018afaa12c0469"
)
GROUPED_SCHEDULER = """    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
"""
UNGROUPED_SCHEDULER = """    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
"""
GROUPED_GRID = "    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)"
UNGROUPED_GRID = "    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))"


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError(f"winner source no longer has exactly one {old!r}")
    return source.replace(old, new, 1)


def source_variants() -> dict[str, tuple[str, dict[str, object]]]:
    winner = WINNER.read_text(encoding="utf-8")
    variants = {}
    for grouped in (False, True):
        for bn in (128, 256):
            for warps in (4, 8):
                source = winner
                if not grouped:
                    source = replace_once(source, GROUPED_SCHEDULER, UNGROUPED_SCHEDULER)
                    source = replace_once(source, GROUPED_GRID, UNGROUPED_GRID)
                source = replace_once(
                    source, "BM, BN, BK, GROUP_M = 128, 256, 64, 8",
                    f"BM, BN, BK, GROUP_M = 128, {bn}, 64, 8",
                )
                source = replace_once(
                    source, "num_warps=8, num_stages=3",
                    f"num_warps={warps}, num_stages=3",
                )
                ast.parse(source)
                name = f"{'grouped' if grouped else 'ungrouped'}_bn{bn}_w{warps}"
                variants[name] = (source, {"grouped": grouped, "bn": bn, "warps": warps})
    if variants["grouped_bn256_w8"][0] != winner:
        raise AssertionError("winner-control variant differs from exported source")
    if len({source for source, _ in variants.values()}) != 8:
        raise AssertionError("crossover variants are not distinct")
    return variants


def round_order(names: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    rng = random.Random(20261030)
    rounds = []
    for _ in range(4):
        order = list(names)
        rng.shuffle(order)
        rounds.append(tuple(order))
    return tuple(rounds)


def save(report: dict[str, object]) -> None:
    temporary = RESULTS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(RESULTS)


def main() -> None:
    if RESULTS.exists():
        raise FileExistsError(f"refusing to overwrite {RESULTS}")
    variants = source_variants()
    for name, (source, _) in variants.items():
        (HERE / f"{name}.py").write_text(source, encoding="utf-8")
    order = round_order(tuple(variants))
    repository = capture_repository_state(ROOT)
    report: dict[str, object] = {
        "run_id": str(uuid4()),
        "image": IMAGE,
        "git_commit": repository.commit,
        "dirty_tree": repository.dirty,
        "winner_source_sha256": hashlib.sha256(WINNER.read_bytes()).hexdigest(),
        "variants": {
            name: {**factors, "source_sha256": hashlib.sha256(source.encode()).hexdigest()}
            for name, (source, factors) in variants.items()
        },
        "order": order,
        "samples_per_evaluation": 30,
        "warmups_per_evaluation": 10,
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
            for cycle, names in enumerate(order):
                for name in names:
                    source, _ = variants[name]
                    evaluation_id = f"triton-crossover-{cycle}-{name}-{uuid4()}"
                    result = worker.evaluate(
                        evaluation_id,
                        KernelProgram(source, backend="triton_h100"),
                        BF16_GEMM_WORKLOAD,
                        "tier0",
                    )
                    payload = serialize_evaluation(result)
                    observation = {
                        "round": cycle,
                        "variant": name,
                        "evaluation_id": evaluation_id,
                        "status": payload["status"],
                        "compile_status": payload["compile_status"],
                        "correctness_status": payload["correctness_status"],
                        "correctness": payload.get("correctness"),
                        "benchmark": payload.get("benchmark"),
                    }
                    report["observations"].append(observation)
                    save(report)
                    benchmark = observation["benchmark"] or {}
                    print(
                        f"round {cycle} {name}: status={observation['status']} "
                        f"median={benchmark.get('median_us')}", flush=True
                    )
                    if result.status == ProposalStatus.INFRASTRUCTURE_FAILURE:
                        raise RuntimeError(f"infrastructure failure for {name} in round {cycle}")
            report["complete"] = True
            save(report)
    finally:
        save(report)


if __name__ == "__main__":
    main()
