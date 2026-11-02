"""Validate dependent-read timing and profiler evidence without link claims."""

from __future__ import annotations

import json
import math
import re
import statistics
from pathlib import Path

from kernel_mcts.cuda_backend import _parse_ncu_csv


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"
CASES = {
    "l1_ca_4k": ("ca", 4 << 10, 8192),
    "l2_cg_4k": ("cg", 4 << 10, 8192),
    "l2_cg_1m": ("cg", 1 << 20, (1 << 20) // 128),
    "l2_cg_8m": ("cg", 8 << 20, (8 << 20) // 128),
    "hbm_cg_128m": ("cg", 128 << 20, (128 << 20) // 128),
    "shared_4k": ("shared", 4 << 10, 8192),
    "register_8k": ("register", 0, 8192),
    "register_64k": ("register", 0, (8 << 20) // 128),
    "register_1m": ("register", 0, (128 << 20) // 128),
}
PROFILES = {
    "latency_l1_ca_4k_l1": "l1tex__t_sector_hit_rate.pct",
    "latency_l1_ca_4k_dram": "dram__bytes_read.sum",
    "latency_l2_cg_1m_l2": "lts__t_sector_hit_rate.pct",
    "latency_l2_cg_1m_dram": "dram__bytes_read.sum",
    "latency_l2_cg_8m_dram": "dram__bytes_read.sum",
    "latency_hbm_cg_128m_dram": "dram__bytes_read.sum",
    "latency_hbm_cg_128m_l2": "lts__t_sector_hit_rate.pct",
    "latency_shared_4k_wavefronts": "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
}


def validate_case(item: dict, rounds: int) -> None:
    label = item["label"]
    if label not in CASES or (item["kind"], item["working_set_bytes"],
                               item["steps"]) != CASES[label]:
        raise ValueError("unexpected latency probe configuration")
    if item["measurement_scope"] != "dependent_read_chain" \
            or item["physical_link_attribution"] != "unverified":
        raise ValueError("latency probe overclaims physical-link attribution")
    for raw_name, median_name in (("raw_cycles_per_hop", "median_cycles_per_hop"),
                                  ("raw_nanoseconds_per_hop", "median_nanoseconds_per_hop")):
        samples = item[raw_name]
        if len(samples) != rounds or any(
            not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
            for value in samples
        ) or not math.isclose(statistics.median(samples), item[median_name],
                              rel_tol=1e-7):
            raise ValueError("invalid latency samples")


def summarize_sass(text: str) -> dict:
    names = {"global_chase_ca": "ca", "global_chase_cg": "cg",
             "shared_chase_probe": "shared", "register_chase_control": "register"}
    sections = {}
    for section in re.split(r"(?=\bFunction\s*:\s*)", text):
        first_line = section.splitlines()[0] if section.splitlines() else ""
        name = next((name for symbol, name in names.items() if symbol in first_line), None)
        if name is not None:
            sections[name] = {opcode: len(re.findall(rf"\b{opcode}\b", section))
                              for opcode in ("LDG", "LDS", "STG", "STS")}
            sections[name]["LDG_STRONG_SM"] = section.count("LDG.E.STRONG.SM")
            sections[name]["LDG_STRONG_GPU"] = section.count("LDG.E.STRONG.GPU")
    if set(sections) != set(names.values()):
        raise ValueError("SASS dump lacks latency probe kernels")
    if sections["shared"]["LDS"] == 0 or sections["register"]["LDS"] != 0 \
            or sections["register"]["LDG"] != 0 \
            or sections["ca"]["LDG_STRONG_SM"] == 0 \
            or sections["cg"]["LDG_STRONG_GPU"] == 0:
        raise ValueError("SASS does not support requested latency paths")
    return sections


def build_analysis(timings: dict, output_dir: Path) -> dict:
    if timings.get("schema_version") != 1 or timings.get("compute_capability") != "9.0" \
            or timings.get("sector_stride_bytes") != 128:
        raise ValueError("unsupported latency measurement schema or GPU")
    cases = timings.get("cases")
    if not isinstance(cases, list) or len(cases) != len(CASES):
        raise ValueError("expected nine latency cases")
    by_label = {}
    for item in cases:
        validate_case(item, timings["rounds"])
        if item["label"] in by_label:
            raise ValueError("duplicate latency case")
        by_label[item["label"]] = item
    profiles = {}
    for label, metric in PROFILES.items():
        path = output_dir / f"{label}.ncu.csv"
        if not path.is_file():
            profiles[label] = {"status": "unavailable"}
            continue
        parsed = _parse_ncu_csv(path.read_text(), (metric,))
        value = parsed[metric]["value"]
        warnings = []
        if metric.endswith("hit_rate.pct") and not 0 <= value <= 100:
            warnings.append("hit_rate_outside_0_100")
        profiles[label] = {"status": "measured", "metric": parsed[metric],
                           "counter_warnings": warnings,
                           "scope": "separate single-metric NCU application replay"}
    sass_path = output_dir / "latency.sass.txt"
    sass = ({"status": "measured", "static_opcode_sites": summarize_sass(
        sass_path.read_text())} if sass_path.is_file() else {"status": "unavailable"})
    return {
        "schema_version": 1,
        "device_name": timings["device_name"],
        "timer": timings["timer"],
        "cases": [by_label[label] for label in CASES],
        "profile_checks": profiles,
        "sass_evidence": sass,
        "physical_link_calibrations": {},
        "limitations": (
            "Per-hop time includes dependent load, address calculation, loop branch, "
            "and timer overhead. Random HBM traversal also includes address-translation "
            "effects. NCU runs separately from timing. Neither cycles/hop nor ns/hop "
            "is an isolated physical-link latency."
        ),
    }


def main() -> None:
    timings = json.loads((OUTPUT / "latency_timings.json").read_text())
    result = build_analysis(timings, OUTPUT)
    (OUTPUT / "latency_analysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    for item in result["cases"]:
        print(item["label"], round(item["median_cycles_per_hop"], 2),
              round(item["median_nanoseconds_per_hop"], 2))
    for label, profile in result["profile_checks"].items():
        print(label, profile["status"], profile.get("metric"))


if __name__ == "__main__":
    main()
