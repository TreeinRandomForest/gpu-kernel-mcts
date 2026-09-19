from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict, dataclass
from typing import Mapping


CTA_TILE_CHOICES = (
    (64, 128),
    (128, 128),
    (128, 256),
    (256, 128),
)
CLUSTER_SHAPE_CHOICES = ((1, 1), (1, 2), (2, 1))


@dataclass(frozen=True, slots=True)
class CuteSchedule:
    """Typed backend configuration for the pinned Hopper dense-GEMM kernel."""

    tile_m: int
    tile_n: int
    cluster_m: int
    cluster_n: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)

    @property
    def configuration_id(self) -> str:
        encoded = json.dumps(
            self.as_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


DEFAULT_CUTE_SCHEDULE = CuteSchedule(128, 256, 1, 1)


def validate_cute_schedule(
    schedule: CuteSchedule,
    *,
    m: int = 4096,
    n: int = 4096,
) -> tuple[str, ...]:
    """Return proven static violations without compiling or running a kernel."""

    reasons: list[str] = []
    tile = (schedule.tile_m, schedule.tile_n)
    cluster = (schedule.cluster_m, schedule.cluster_n)
    if tile not in CTA_TILE_CHOICES:
        reasons.append(f"unsupported CTA tile {tile}")
    if cluster not in CLUSTER_SHAPE_CHOICES:
        reasons.append(f"unsupported cluster shape {cluster}")
    if min(schedule.tile_m, schedule.tile_n) <= 0:
        reasons.append("CTA tile dimensions must be positive")
    if min(schedule.cluster_m, schedule.cluster_n) <= 0:
        reasons.append("cluster dimensions must be positive")
    if schedule.cluster_m * schedule.cluster_n > 8:
        reasons.append("cluster size exceeds the Hopper limit of 8 CTAs")
    if schedule.tile_m > 0 and m % schedule.tile_m:
        reasons.append("CTA tile M does not divide the fixed workload M")
    if schedule.tile_n > 0 and n % schedule.tile_n:
        reasons.append("CTA tile N does not divide the fixed workload N")
    if (
        schedule.tile_m > 0
        and schedule.cluster_m > 0
        and (m // schedule.tile_m) % schedule.cluster_m
    ):
        reasons.append("M tile grid is not divisible by cluster M")
    if (
        schedule.tile_n > 0
        and schedule.cluster_n > 0
        and (n // schedule.tile_n) % schedule.cluster_n
    ):
        reasons.append("N tile grid is not divisible by cluster N")
    return tuple(dict.fromkeys(reasons))


def enumerate_cute_schedules() -> tuple[CuteSchedule, ...]:
    schedules = (
        CuteSchedule(tile_m, tile_n, cluster_m, cluster_n)
        for (tile_m, tile_n), (cluster_m, cluster_n) in itertools.product(
            CTA_TILE_CHOICES, CLUSTER_SHAPE_CHOICES
        )
    )
    return tuple(schedule for schedule in schedules if not validate_cute_schedule(schedule))


def cute_schedule_from_mapping(value: Mapping[str, int]) -> CuteSchedule:
    expected = {"tile_m", "tile_n", "cluster_m", "cluster_n"}
    if set(value) != expected:
        raise ValueError(f"CuTe schedule fields must be exactly {sorted(expected)}")
    schedule = CuteSchedule(**{name: int(value[name]) for name in expected})
    reasons = validate_cute_schedule(schedule)
    if reasons:
        raise ValueError("; ".join(reasons))
    return schedule
