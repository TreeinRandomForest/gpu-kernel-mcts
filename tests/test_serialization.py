from __future__ import annotations

import json

import pytest

from kernel_mcts.domain import (
    BenchmarkResult,
    CompilationEvidence,
    CompileStatus,
    CorrectnessEvidence,
    CorrectnessStatus,
    EvaluationResult,
    KernelProgram,
    ProposalStatus,
    ShapeCase,
    WorkloadContract,
)
from kernel_mcts.generation import GenerationResult
from kernel_mcts.serialization import (
    serialize_environment_manifest,
    serialize_evaluation,
    serialize_generation,
    serialize_profile,
    serialize_workload,
)
from kernel_mcts.providers import EnvironmentManifest


def test_complete_evaluation_serializes_to_json() -> None:
    evaluation = EvaluationResult(
        status=ProposalStatus.VALID,
        program=KernelProgram("kernel"),
        state_key="state",
        reward=0.5,
        benchmark=BenchmarkResult((1.0, 2.0), 1.5, {"n=1": 1.5}),
        metadata={"registers": 32},
        compile_status=CompileStatus.SUCCESS,
        correctness_status=CorrectnessStatus.PASS,
        worker_id="worker",
        environment_manifest_id="manifest",
        source_hash="source-hash",
        binary_hash="binary-hash",
        launch_config={"block": [256, 1, 1]},
        compiled_artifact=object(),
        compilation=CompilationEvidence("artifact", "compiler output", ""),
        correctness=CorrectnessEvidence(0.01, 0.001),
    )

    serialized = serialize_evaluation(evaluation)

    assert serialized["compile_status"] == "SUCCESS"
    assert serialized["correctness_status"] == "PASS"
    assert serialized["benchmark"]["timings_us"] == [1.0, 2.0]
    assert serialized["launch_config"] == {"block": [256, 1, 1]}
    assert serialized["compilation"] == {
        "artifact_id": "artifact",
        "stdout": "compiler output",
        "stderr": "",
    }
    assert serialized["correctness"] == {
        "maximum_error": 0.01,
        "mean_error": 0.001,
    }
    assert "compiled_artifact" not in serialized
    json.dumps(serialized)


def test_generation_workload_and_profile_serialize_to_json() -> None:
    generation = GenerationResult(
        "generation",
        "raw",
        KernelProgram("kernel"),
        "prompt-hash",
        input_tokens=10,
        output_tokens=20,
        metadata={"model": "test"},
    )
    workload = WorkloadContract(
        "toy",
        "op",
        "bf16",
        (ShapeCase({"m": 16, "n": 32}, 1.0),),
        1e-2,
        1e-2,
    )

    values = {
        "generation": serialize_generation(generation),
        "workload": serialize_workload(workload),
        "profile": serialize_profile({"occupancy": 0.75, "stalls": ("memory",)}),
    }

    assert values["workload"]["shapes"][0]["dimensions"] == {"m": 16, "n": 32}
    json.dumps(values)


def test_serializer_rejects_opaque_objects() -> None:
    with pytest.raises(TypeError, match="not JSON serializable"):
        serialize_profile({"artifact": object()})


def test_environment_manifest_serializes_with_stable_id() -> None:
    manifest = EnvironmentManifest(
        worker_id="worker",
        provider="runpod",
        gpu_model="NVIDIA H100",
        compute_capability="9.0",
        captured_at="2026-09-09T12:00:00+00:00",
        form_factor="SXM",
        toolchain_versions={"cuda_toolkit": "12.4"},
    )

    serialized = serialize_environment_manifest(manifest)

    assert serialized["manifest_id"] == manifest.manifest_id
    assert serialized["form_factor"] == "SXM"
    json.dumps(serialized)
