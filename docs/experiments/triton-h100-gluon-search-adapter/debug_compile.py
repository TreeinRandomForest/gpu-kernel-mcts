"""Print an uncaught traceback for a candidate's first GPU launch."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


def main() -> None:
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("gluon_debug_candidate", args.candidate)
    if spec is None or spec.loader is None:
        raise ValueError("cannot import candidate")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    a = torch.empty((4096, 4096), device="cuda", dtype=torch.bfloat16)
    b = torch.empty((4096, 4096), device="cuda", dtype=torch.bfloat16).T
    c = torch.empty((4096, 4096), device="cuda", dtype=torch.bfloat16)
    module.launch(a, b, c, 4096, 4096, 4096)
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
