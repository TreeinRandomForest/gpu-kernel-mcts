from __future__ import annotations

import os
from typing import Sequence

from . import cute_baseline_cli, worker_service


def main(argv: Sequence[str] | None = None) -> int:
    """Preserve standalone CuTe commands while supporting remote worker startup."""
    if os.environ.get("KERNEL_MCTS_WORKER_TOKEN"):
        worker_service.main()
        return 0
    return cute_baseline_cli.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
