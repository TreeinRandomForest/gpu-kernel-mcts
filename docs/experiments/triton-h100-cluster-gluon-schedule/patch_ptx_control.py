"""Replace one multicast B TMA with elected per-CTA B TMA in Gluon PTX.

This is a compiler-schedule probe only. A compiled cubin must be checked for
correctness before its performance can be interpreted.
"""

from __future__ import annotations

import argparse
from pathlib import Path


OLD = ("@%p17 cp.async.bulk.tensor.2d.shared::cluster.global."
       "mbarrier::complete_tx::bytes.multicast::cluster "
       "[%r115], [%rd6, {%r134, %r116}], [%r112], %rs1;")
NEW = ("@%p20 cp.async.bulk.tensor.2d.shared::cta.global."
       "mbarrier::complete_tx::bytes "
       "[%r115], [%rd6, {%r134, %r116}], [%r112];")


def patch(source: str) -> str:
    if source.count(OLD) != 1:
        raise ValueError("expected exactly one pinned multicast B instruction")
    if "elect.sync \t%r120|%p20, -1;" not in source:
        raise ValueError("expected per-CTA elected issuer predicate")
    result = source.replace(OLD, NEW)
    if "bytes.multicast::cluster" in result:
        raise ValueError("multicast instruction remained")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(patch(args.source.read_text()))


if __name__ == "__main__":
    main()
