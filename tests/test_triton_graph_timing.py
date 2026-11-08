from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

from kernel_mcts import triton_harness


class FakeTensor:
    def __init__(self, value=1):
        self.value = value

    def __matmul__(self, other):
        return FakeTensor(1)

    def zero_(self):
        self.value = 0


def test_graph_replay_drives_reward_and_ordinary_launch_is_diagnostic(
    monkeypatch, tmp_path: Path,
) -> None:
    counts = {"capture": 0, "ordinary": 0, "replay": 0}
    cuda = SimpleNamespace(current_graph=None, last_kind=None)

    class Event:
        def __init__(self, enable_timing):
            assert enable_timing

        def record(self):
            pass

        def synchronize(self):
            pass

        def elapsed_time(self, other):
            return 0.1 if cuda.last_kind == "graph" else 0.25

    class Graph:
        output = None

        def replay(self):
            counts["replay"] += 1
            self.output.value = 1
            cuda.last_kind = "graph"

    class Capture:
        def __init__(self, graph):
            self.graph = graph

        def __enter__(self):
            counts["capture"] += 1
            cuda.current_graph = self.graph

        def __exit__(self, kind, value, traceback):
            cuda.current_graph = None

    cuda.is_available = lambda: True
    cuda.synchronize = lambda: None
    cuda.Event = Event
    cuda.CUDAGraph = Graph
    cuda.graph = Capture
    torch = SimpleNamespace(cuda=cuda,
                            allclose=lambda actual, expected, **kwargs:
                            actual.value == expected.value)
    monkeypatch.setitem(sys.modules, "torch", torch)
    a, b, c = FakeTensor(), FakeTensor(), FakeTensor()
    monkeypatch.setattr(triton_harness, "_inputs", lambda *args: (a, b, c))
    compiled = SimpleNamespace(asm={"cubin": b"compiled"})

    def launch(*args):
        if cuda.current_graph is not None:
            cuda.current_graph.output = c
        else:
            counts["ordinary"] += 1
        c.value = 1
        cuda.last_kind = "ordinary"
        return compiled

    monkeypatch.setattr(triton_harness, "_candidate", lambda path: launch)
    (tmp_path / "kernel.cubin").write_bytes(b"compiled")
    arguments = argparse.Namespace(
        M=4096, N=4096, K=4096, seed=0, candidate=Path("candidate.py"),
        artifact_dir=tmp_path, mode="benchmark", warmups=2, measurements=3,
        rtol=0.02, atol=0.02,
    )

    result = triton_harness.run(arguments)

    assert result["timing_contract"] == triton_harness.GRAPH_TIMING_CONTRACT
    assert result["timings_us"] == [100.0] * 3
    assert result["diagnostics"]["ordinary_launch_timings_us"] == [250.0] * 3
    assert counts == {"capture": 1, "ordinary": 6, "replay": 6}
