from kernel_mcts.triton_compiler_summary import (
    prompt_triton_compiler_summary, ptx_instruction_sites,
)


def test_ptx_counts_static_instruction_sites_without_labels_or_directives() -> None:
    sites = ptx_instruction_sites("""
    .version 8.0
    loop:
    @%p1 ld.global.u32 %r1, [%rd1];
    st.shared.u32 [%rd2], %r1;
    mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 %f0, %r1, %r2, %f0;
    ld.local.u32 %r3, [%rd3]; // spill-like local access
    bar.sync 0;
    bra loop;
    """)
    assert sites["total"] == 6
    assert sites["categories"]["global_load"] == 1
    assert sites["categories"]["shared_store"] == 1
    assert sites["categories"]["tensor"] == 1
    assert sites["categories"]["local_load"] == 1
    assert sites["categories"]["barrier"] == 1
    assert sites["categories"]["control"] == 1


def test_triton_compiler_prompt_summary_is_bounded_and_versioned() -> None:
    context = {
        "schema_version": 1, "backend": "triton_h100", "architecture": "sm_90",
        "resources": {"registers_per_thread": 255,
                      "static_shared_memory_bytes": 4096, "spills": 256,
                      "source_path": "/secret"},
        "ptx": {"status": "captured", "artifacts": [{"sha256": "a" * 64,
                  "path": "/secret"}],
                "static_instruction_sites": ptx_instruction_sites("ld.local.u32 %r1, [%rd1];")},
    }
    summary = prompt_triton_compiler_summary(context)
    assert summary["status"] == "available"
    assert summary["compiler_resources"]["spills"] == 256
    assert summary["ptx_static_instruction_sites"]["categories"]["local_load"] == 1
    assert summary["ptx_sha256"] == "a" * 64
    assert "/secret" not in str(summary)
    assert prompt_triton_compiler_summary({"schema_version": 2})["status"] == "unavailable"
