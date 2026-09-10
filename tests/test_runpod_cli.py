from __future__ import annotations

import io

from kernel_mcts.runpod_cli import _ReadinessProgress


class TTYBuffer(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_readiness_progress_animates_one_terminal_line() -> None:
    stream = TTYBuffer()
    progress = _ReadinessProgress("pod-1", stream)

    progress("CREATED", 0.0)
    progress("RUNNING", 2.0)
    progress.finish()

    output = stream.getvalue()
    assert "⠋ Waiting for pod pod-1 — status: CREATED — 0s elapsed" in output
    assert "⠙ Waiting for pod pod-1 — status: RUNNING — 2s elapsed" in output
    assert output.count("\n") == 1


def test_noninteractive_progress_logs_only_status_changes() -> None:
    stream = io.StringIO()
    progress = _ReadinessProgress("pod-1", stream)

    progress("CREATED", 0.0)
    progress("CREATED", 2.0)
    progress("RUNNING", 4.0)
    progress.finish()

    assert stream.getvalue().splitlines() == [
        "Waiting for pod pod-1 — status: CREATED — 0s elapsed",
        "Waiting for pod pod-1 — status: RUNNING — 4s elapsed",
    ]
