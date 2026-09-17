from __future__ import annotations

import subprocess
from pathlib import Path

from kernel_mcts.provenance import (
    capture_repository_state,
    image_digest_from_reference,
)


class GitRunner:
    def __init__(self, *, status: str = "") -> None:
        self.status = status
        self.calls = []

    def __call__(self, command, **kwargs):
        self.calls.append((list(command), kwargs))
        output = (
            "a" * 40 + "\n"
            if command[-2:] == ("rev-parse", "HEAD")
            else self.status
        )
        return subprocess.CompletedProcess(command, 0, output, "")


def test_repository_state_records_commit_and_dirty_boolean_only(tmp_path: Path) -> None:
    clean = capture_repository_state(tmp_path, runner=GitRunner())
    dirty = capture_repository_state(
        tmp_path,
        runner=GitRunner(status=" M src/kernel_mcts/search.py\n?? local-output\n"),
    )

    assert clean.commit == "a" * 40
    assert clean.dirty is False
    assert dirty.commit == "a" * 40
    assert dirty.dirty is True


def test_repository_state_is_unknown_outside_git_repository(tmp_path: Path) -> None:
    def failed(command, **kwargs):
        return subprocess.CompletedProcess(command, 128, "", "not a repository")

    state = capture_repository_state(tmp_path, runner=failed)

    assert state.commit is None
    assert state.dirty is None


def test_image_digest_is_extracted_only_from_immutable_reference() -> None:
    digest = "a" * 64

    assert image_digest_from_reference(f"registry/worker@sha256:{digest}") == (
        f"sha256:{digest}"
    )
    assert image_digest_from_reference("registry/worker:v1") is None
    assert image_digest_from_reference("registry/worker@sha256:short") is None
