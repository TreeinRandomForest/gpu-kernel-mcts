from __future__ import annotations

from types import SimpleNamespace

from kernel_mcts.cute_diagnostics import (
    ArtifactChange,
    artifact_changes,
    describe_kernel_callable,
    discover_cache_roots,
    select_fingerprint_candidate,
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
