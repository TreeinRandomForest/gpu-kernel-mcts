from __future__ import annotations

from types import SimpleNamespace

from kernel_mcts.cute_diagnostics import (
    ArtifactChange,
    artifact_changes,
    describe_kernel_callable,
    describe_mlir,
    describe_runtime_artifacts,
    discover_cache_roots,
    extract_mlir_embedded_artifacts,
    select_fingerprint_candidate,
    select_runtime_fingerprint,
    snapshot_files,
)


def test_discovers_existing_unique_cache_roots(tmp_path) -> None:
    home = tmp_path / "home"
    default = home / ".cache"
    explicit = tmp_path / "explicit"
    default.mkdir(parents=True)
    explicit.mkdir()

    roots = discover_cache_roots(
        {"CUDA_CACHE_PATH": str(explicit), "XDG_CACHE_HOME": str(explicit)},
        home=home,
        temporary_roots=(),
    )

    assert roots == (default.resolve(), explicit.resolve())


def test_snapshots_and_hashes_created_and_modified_files(tmp_path) -> None:
    existing = tmp_path / "existing.ptx"
    existing.write_text("old", encoding="utf-8")
    before = snapshot_files((tmp_path,))
    existing.write_text("new contents", encoding="utf-8")
    cubin = tmp_path / "kernel.cubin"
    cubin.write_bytes(b"cubin")

    changes = artifact_changes(before, snapshot_files((tmp_path,)))

    assert [(item.change, item.suffix) for item in changes] == [
        ("modified", ".ptx"),
        ("created", ".cubin"),
    ]
    assert all(len(item.sha256) == 64 for item in changes)


def test_fingerprint_candidate_prefers_binary_over_ptx() -> None:
    ptx = ArtifactChange("/cache/a.ptx", "created", 10, "p" * 64, ".ptx")
    cubin = ArtifactChange("/cache/b.cubin", "created", 20, "c" * 64, ".cubin")

    selected = select_fingerprint_candidate((ptx, cubin))

    assert selected["kind"] == "cubin_sha256"
    assert selected["path"] == "/cache/b.cubin"


def test_describes_callable_without_serializing_tensor_contents() -> None:
    def kernel(value):
        return value

    kernel.kernel_name = "gemm_kernel"
    workspace = SimpleNamespace(args=(object(),), kwargs={"output": object()})

    result = describe_kernel_callable(kernel, {"kernel_arguments": workspace})

    assert result["callable_name"] == "kernel"
    assert result["selected_attributes"]["kernel_name"] == "gemm_kernel"
    assert result["benchmark_keyword_names"] == ["kernel_arguments"]
    assert len(result["argument_types"]) == 1


def test_normalizes_mlir_identity_and_persists_text() -> None:
    first = describe_mlir('module @object_at__ABC123 loc("/tmp/a/file.py") 0x1234')
    second = describe_mlir('module @object_at__DEF456 loc("/tmp/b/file.py") 0x9999')

    assert first["normalized_sha256"] == second["normalized_sha256"]
    assert "<address>" in first["normalized_text"]
    assert "<tmp-path>" in first["normalized_text"]


def test_normalization_does_not_rewrite_semantic_copy_atom_name() -> None:
    value = describe_mlir("module @object_at__CopyAtom")

    assert "object_at__CopyAtom" in value["normalized_text"]


def test_extracts_embedded_cuda_fatbinary_from_mlir() -> None:
    artifacts = extract_mlir_embedded_artifacts(
        'llvm.mlir.global internal constant @kernels_binary("P\\EDU\\BA\\01\\00")'
    )

    assert len(artifacts) == 1
    assert artifacts[0]["kind"] == "fatbin"
    assert artifacts[0]["size"] == 6


def test_runtime_fingerprint_prefers_in_memory_cubin_then_mlir() -> None:
    runtime = SimpleNamespace(cubin_image=b"compiled-cubin")
    artifacts = describe_runtime_artifacts(runtime, "jit_module")
    diagnostics = {
        "runtime_artifacts": artifacts,
        "mlir": {"normalized_sha256": "m" * 64, "normalized_bytes": 10},
    }

    selected = select_runtime_fingerprint(diagnostics)

    assert selected["kind"] == "cubin"
    assert selected["attribute"] == "jit_module.cubin_image"

    fallback = select_runtime_fingerprint(
        {"runtime_artifacts": [], "mlir": diagnostics["mlir"]}
    )
    assert fallback["kind"] == "normalized_mlir"
