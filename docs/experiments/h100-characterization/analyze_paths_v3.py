"""Validate matched cache controls and explicit on-chip probe results."""

from __future__ import annotations

import json
import math
import re
import statistics
from pathlib import Path

from kernel_mcts.cuda_backend import _parse_ncu_csv


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"
READ_CASES = {(8 << 20, repeat, "cyclic") for repeat in (1, 4, 16, 64)}
READ_CASES.update({(256 << 20, repeat, "disjoint") for repeat in (1, 4)})
ONCHIP_CASES = {(kind, repeat) for kind in ("shared", "register")
                for repeat in (1, 16, 64, 256, 1024)}
PROFILES = {
    "clean_l2_8m_x64_dram": "dram__bytes_read.sum",
    "clean_l2_8m_x64_l2": "lts__t_sector_hit_rate.pct",
    "clean_hbm_256m_x4_dram": "dram__bytes_read.sum",
    "onchip_shared_x256_wavefronts": "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
    "onchip_register_x256_wavefronts": "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
}


def validate_samples(item: dict, rounds: int) -> None:
    samples = item["raw_us_per_launch"]
    if len(samples) != rounds or any(
        not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0
        for value in samples
    ) or not math.isclose(item["median_us_per_launch"], statistics.median(samples),
                          rel_tol=1e-9):
        raise ValueError("invalid timing samples")


def summarize_sass(text: str) -> dict:
    sections = {}
    for section in re.split(r"(?=\bFunction\s*:\s*)", text):
        if "Function :" not in section:
            continue
        name = "shared" if "shared_load_probe" in section.splitlines()[0] else (
            "register" if "register_control_probe" in section.splitlines()[0] else None
        )
        if name is None:
            continue
        sections[name] = {opcode: len(re.findall(rf"\b{opcode}\b", section))
                          for opcode in ("LDS", "STS", "LDG", "STG")}
    if set(sections) != {"shared", "register"}:
        raise ValueError("SASS dump lacks both on-chip probe kernels")
    if sections["shared"]["LDS"] == 0 or sections["register"]["LDS"] != 0 \
            or sections["shared"]["LDG"] != 0 or sections["register"]["LDG"] != 0:
        raise ValueError("SASS does not support shared/register probe distinction")
    return sections


def build_analysis(read: dict, onchip: dict, output_dir: Path) -> dict:
    if read.get("schema_version") != 1 or onchip.get("schema_version") != 1 \
            or read.get("compute_capability") != "9.0" \
            or onchip.get("compute_capability") != "9.0" \
            or read.get("device_name") != onchip.get("device_name"):
        raise ValueError("unsupported or mismatched GPU measurements")
    read_cases = {}
    observations = read.get("observations")
    if not isinstance(observations, list) or len(observations) != len(READ_CASES):
        raise ValueError("expected six matched read cases")
    for item in observations:
        key = (item["payload_bytes"], item["repeats"], item["working_set_policy"])
        if key not in READ_CASES or key in read_cases:
            raise ValueError("unexpected or duplicate read case")
        validate_samples(item, item["rounds"])
        if item["working_set_bytes"] != key[0] * (key[1] if key[2] == "disjoint" else 1) \
                or item["requested_read_bytes_per_launch"] != key[0] * key[1]:
            raise ValueError("incorrect working-set or requested byte count")
        if not any("ld.global.cg" in line for line in item["ptx_global_load_lines"]):
            raise ValueError("PTX does not verify .cg load")
        if item["physical_link_attribution"] != "unverified":
            raise ValueError("read case overclaims a physical link")
        read_cases[key] = item
    onchip_cases = {}
    observations = onchip.get("cases")
    if not isinstance(observations, list) or len(observations) != len(ONCHIP_CASES):
        raise ValueError("expected ten on-chip cases")
    for item in observations:
        key = (item["kind"], item["repeats"])
        if key not in ONCHIP_CASES or key in onchip_cases:
            raise ValueError("unexpected or duplicate on-chip case")
        validate_samples(item, onchip["rounds"])
        expected = (onchip["blocks"] * onchip["threads_per_block"] * 4 * key[1]
                    if key[0] == "shared" else 0)
        if item["requested_shared_read_bytes_per_launch"] != expected:
            raise ValueError("incorrect shared-read byte count")
        if item["physical_link_attribution"] != "unverified":
            raise ValueError("on-chip case overclaims a physical link")
        item = dict(item)
        if key[0] == "shared":
            item["effective_requested_shared_gb_per_s"] = (
                expected / (item["median_us_per_launch"] * 1000)
            )
        onchip_cases[key] = item
    profiles = {}
    for label, metric in PROFILES.items():
        path = output_dir / f"{label}.ncu.csv"
        if not path.is_file():
            profiles[label] = {"status": "unavailable"}
            continue
        parsed = _parse_ncu_csv(path.read_text(), (metric,))
        value = parsed[metric]["value"]
        warning = []
        if metric.endswith("hit_rate.pct") and not 0 <= value <= 100:
            warning.append("hit_rate_outside_0_100")
        profiles[label] = {"status": "measured", "metric": parsed[metric],
                           "counter_warnings": warning,
                           "scope": "separate one-metric NCU pass"}
    l2 = profiles["clean_l2_8m_x64_dram"]
    hbm = profiles["clean_hbm_256m_x4_dram"]
    regimes = {
        "8m_cyclic_x64": "unresolved",
        "256m_disjoint_x4": "unresolved",
    }
    if l2["status"] == "measured":
        fraction = l2["metric"]["value"] / ((8 << 20) * 64)
        regimes["8m_cyclic_x64"] = (
            "cache_resident_read_evidence" if fraction <= 0.05 else "mixed_or_unresolved"
        )
    if hbm["status"] == "measured":
        fraction = hbm["metric"]["value"] / ((256 << 20) * 4)
        regimes["256m_disjoint_x4"] = (
            "hbm_sourced_read_evidence" if fraction >= 0.8 else "mixed_or_unresolved"
        )
    sass_path = output_dir / "onchip.sass.txt"
    sass_evidence = ({"status": "measured", "static_opcode_sites": summarize_sass(
        sass_path.read_text())} if sass_path.is_file() else {"status": "unavailable"})
    expected_wavefronts = (onchip["blocks"] * (onchip["threads_per_block"] // 32)
                           * 256)
    shared_profile = profiles["onchip_shared_x256_wavefronts"]
    register_profile = profiles["onchip_register_x256_wavefronts"]
    onchip_path_evidence = {
        "status": "unresolved",
        "expected_shared_load_wavefronts_x256": expected_wavefronts,
    }
    if shared_profile["status"] == "measured" and register_profile["status"] == "measured":
        onchip_path_evidence["observed_shared_load_wavefronts_x256"] = (
            shared_profile["metric"]["value"]
        )
        onchip_path_evidence["observed_register_control_shared_wavefronts_x256"] = (
            register_profile["metric"]["value"]
        )
        if sass_evidence["status"] == "measured" \
                and shared_profile["metric"]["value"] == expected_wavefronts \
                and register_profile["metric"]["value"] == 0:
            onchip_path_evidence["status"] = "explicit_shared_load_path_verified"
    return {
        "schema_version": 1,
        "device_name": read["device_name"],
        "matched_read_cases": [read_cases[key] for key in sorted(read_cases)],
        "onchip_cases": [onchip_cases[key] for key in sorted(onchip_cases)],
        "profile_checks": profiles,
        "cache_regimes": regimes,
        "sass_evidence": sass_evidence,
        "onchip_path_evidence": onchip_path_evidence,
        "physical_link_calibrations": {},
        "limitations": (
            "Read timing includes loop/address arithmetic, reduction, output, and launch. "
            "Shared timing includes initialization, barrier, loads, arithmetic, and "
            "global output; register control includes arithmetic and output. NCU "
            "profiles are separate one-metric runs. No isolated physical-link rate "
            "or latency is inferred."
        ),
    }


def main() -> None:
    read = json.loads((OUTPUT / "clean_paths_timings.json").read_text())
    onchip = json.loads((OUTPUT / "onchip_timings.json").read_text())
    result = build_analysis(read, onchip, OUTPUT)
    (OUTPUT / "paths_v3_analysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    for item in result["matched_read_cases"]:
        print("read", item["payload_bytes"], item["repeats"],
              item["working_set_policy"], round(item["median_us_per_launch"], 3))
    for item in result["onchip_cases"]:
        print("onchip", item["kind"], item["repeats"],
              round(item["median_us_per_launch"], 3))
    for label, profile in result["profile_checks"].items():
        print("profile", label, profile["status"], profile.get("metric"))


if __name__ == "__main__":
    main()
