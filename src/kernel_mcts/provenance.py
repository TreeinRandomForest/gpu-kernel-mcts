from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


@dataclass(frozen=True, slots=True)
class RepositoryState:
    commit: str | None
    dirty: bool | None


GitRunner = Callable[..., subprocess.CompletedProcess[str]]


def capture_repository_state(
    path: Path,
    *,
    runner: GitRunner = subprocess.run,
) -> RepositoryState:
    """Capture controller Git provenance without exposing working-tree contents."""
    commit = _git(
        runner,
        ("git", "-C", str(path), "rev-parse", "HEAD"),
    )
    if commit is None:
        return RepositoryState(None, None)
    status = _git(
        runner,
        (
            "git",
            "-C",
            str(path),
            "status",
            "--porcelain=v1",
            "--untracked-files=normal",
        ),
    )
    return RepositoryState(commit, None if status is None else bool(status))


def image_digest_from_reference(image: str) -> str | None:
    marker = "@sha256:"
    if marker not in image:
        return None
    digest = image.rsplit(marker, 1)[1]
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        return None
    return f"sha256:{digest}"


def _git(
    runner: GitRunner,
    command: Sequence[str],
) -> str | None:
    try:
        result = runner(
            command,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None
