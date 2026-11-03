"""Convert multicast B TMA to one elected ordinary B TMA per CTA.

The first patch omitted the first-warp guard. Keep that guard so each CTA
issues exactly one B request against its 49,152-byte transaction barrier.
"""

from __future__ import annotations

import argparse
from pathlib import Path


OLD_PREDICATE = "and.pred \t%p17, %p20, %p1;"
NEW_PREDICATE = "and.pred \t%p17, %p20, %p5;"
OLD_LOAD = ("@%p17 cp.async.bulk.tensor.2d.shared::cluster.global."
            "mbarrier::complete_tx::bytes.multicast::cluster "
            "[%r115], [%rd6, {%r134, %r116}], [%r112], %rs1;")
NEW_LOAD = ("@%p17 cp.async.bulk.tensor.2d.shared::cta.global."
            "mbarrier::complete_tx::bytes "
            "[%r115], [%rd6, {%r134, %r116}], [%r112];")


def patch(source: str) -> str:
    if source.count(OLD_PREDICATE) != 1 or source.count(OLD_LOAD) != 1:
        raise ValueError("pinned PTX predicate or B TMA instruction changed")
    return source.replace(OLD_PREDICATE, NEW_PREDICATE).replace(OLD_LOAD, NEW_LOAD)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(patch(args.source.read_text()))


if __name__ == "__main__":
    main()
