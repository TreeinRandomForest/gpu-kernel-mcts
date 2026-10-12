from __future__ import annotations

import hashlib

from kernel_mcts.compiler_context import (
    cuda_compiler_context,
    parse_ptxas_resources,
    sass_instruction_categories,
)


PTXAS_OUTPUT = """ptxas info    : Compiling entry function 'helper' for 'sm_90'
ptxas info    : Function properties for helper
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 12 registers, 0 bytes smem
ptxas info    : Compiling entry function 'bf16_gemm_root' for 'sm_90'
ptxas info    : Function properties for bf16_gemm_root
    16 bytes stack frame, 8 bytes spill stores, 4 bytes spill loads
ptxas info    : Used 64 registers, 1024 bytes smem, 344 bytes cmem[0]
"""


def test_ptxas_parser_selects_entry_point_and_resources() -> None:
    assert parse_ptxas_resources(PTXAS_OUTPUT, "bf16_gemm_root") == {
        "stack_frame_bytes": 16,
        "spill_store_bytes": 8,
        "spill_load_bytes": 4,
        "registers_per_thread": 64,
        "static_shared_memory_bytes": 1024,
    }
    assert parse_ptxas_resources(PTXAS_OUTPUT, "unknown") is None


def test_context_records_ptx_provenance_without_embedding_source(tmp_path) -> None:
    ptx = tmp_path / "candidate.ptx"
    ptx.write_bytes(b".version 8.0\n.visible .entry bf16_gemm_root() {}")

    context = cuda_compiler_context(
        artifact_dir=tmp_path,
        artifact_id="artifact-1",
        architecture="sm_90",
        entry_point="bf16_gemm_root",
        compiler_output=PTXAS_OUTPUT,
    )

    assert context["schema_version"] == 2
    assert context["artifact_id"] == "artifact-1"
    assert context["ptx"] == {
        "status": "captured",
        "artifacts": [
            {
                "path": str(ptx),
                "sha256": hashlib.sha256(ptx.read_bytes()).hexdigest(),
                "size_bytes": ptx.stat().st_size,
                "storage_status": "eligible",
            }
        ],
    }
    assert context["ptxas"]["status"] == "parsed"
    assert context["ptxas"]["resources"]["registers_per_thread"] == 64
    assert context["sass"] == {"status": "pending"}
    assert ".version" not in str(context)


def test_missing_optional_evidence_has_explicit_status(tmp_path) -> None:
    context = cuda_compiler_context(
        artifact_dir=tmp_path,
        artifact_id="artifact-1",
        architecture="sm_90",
        entry_point="bf16_gemm_root",
        compiler_output="",
    )

    assert context["ptx"] == {"status": "missing", "artifacts": []}
    assert context["ptxas"]["status"] == "missing"
    assert context["ptxas"]["resources"] is None


def test_sass_summary_counts_only_requested_kernel() -> None:
    output = """Function : helper
/*0000*/ LDG.E R1, [R2];
Function : bf16_gemm_root
/*0000*/ LDG.E R1, [R2];
/*0010*/ LDS R3, [R4];
/*0020*/ HMMA.16816 R5, R6, R7, R8;
/*0030*/ FFMA R9, R1, R2, R3;
/*0040*/ STG.E [R2], R9;
/*0050*/ EXIT;
"""
    summary = sass_instruction_categories(output, "bf16_gemm_root")
    assert summary is not None
    assert summary["instruction_count"] == 6
    assert {key: count for key, count in summary["categories"].items() if count} == {
        "global_load": 1,
        "shared_load": 1,
        "tensor": 1,
        "arithmetic": 1,
        "global_store": 1,
        "control": 1,
    }
