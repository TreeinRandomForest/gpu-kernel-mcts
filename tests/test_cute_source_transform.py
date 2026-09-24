from __future__ import annotations

import hashlib

import pytest

from kernel_mcts.cute_source_transform import (
    transform_wgmma_inflight_groups,
    validate_pinned_cute_gemm_source,
)


_SOURCE = """\
def kernel():
    k_pipe_mmas = 1
    return k_pipe_mmas
"""
_SHA256 = hashlib.sha256(_SOURCE.encode("utf-8")).hexdigest()


def test_transform_replaces_exactly_one_pinned_assignment() -> None:
    transformed = transform_wgmma_inflight_groups(
        _SOURCE, 2, expected_sha256=_SHA256
    )

    assert "k_pipe_mmas = 2" in transformed
    assert "k_pipe_mmas = 1" not in transformed
    compile(transformed, "fixture.py", "exec")


def test_transform_preserves_source_for_default() -> None:
    assert transform_wgmma_inflight_groups(
        _SOURCE, 1, expected_sha256=_SHA256
    ) == _SOURCE


def test_transform_rejects_hash_mismatch() -> None:
    with pytest.raises(ValueError, match="source hash mismatch"):
        transform_wgmma_inflight_groups(_SOURCE, 2)


def test_transform_rejects_ambiguous_assignment() -> None:
    source = _SOURCE + "k_pipe_mmas = 1\n"
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()

    with pytest.raises(ValueError, match="exactly one"):
        transform_wgmma_inflight_groups(source, 2, expected_sha256=digest)


def test_transform_rejects_out_of_range_group_count() -> None:
    with pytest.raises(ValueError, match="must be 1 or 2"):
        transform_wgmma_inflight_groups(_SOURCE, 3, expected_sha256=_SHA256)


def test_transform_rejects_changed_pinned_default() -> None:
    source = _SOURCE.replace("k_pipe_mmas = 1", "k_pipe_mmas = 0")
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()

    with pytest.raises(ValueError, match="must have value 1"):
        transform_wgmma_inflight_groups(source, 2, expected_sha256=digest)


def test_pinned_source_validator_accepts_expected_hash(tmp_path, monkeypatch) -> None:
    path = tmp_path / "dense_gemm.py"
    path.write_text(_SOURCE, encoding="utf-8")
    monkeypatch.setattr(
        "kernel_mcts.cute_source_transform.PINNED_CUTE_GEMM_SHA256", _SHA256
    )

    assert validate_pinned_cute_gemm_source(path) == _SHA256


def test_pinned_source_validator_rejects_other_source(tmp_path) -> None:
    path = tmp_path / "dense_gemm.py"
    path.write_text(_SOURCE, encoding="utf-8")

    with pytest.raises(ValueError, match="source hash mismatch"):
        validate_pinned_cute_gemm_source(path)
