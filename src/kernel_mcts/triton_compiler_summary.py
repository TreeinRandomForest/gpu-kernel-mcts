"""Bounded static compilation evidence for Triton H100 prompts."""

from __future__ import annotations

import re
from collections import Counter
from typing import Mapping


SCHEMA_VERSION = 1
_INSTRUCTION = re.compile(r"^\s*(?:@!?%[A-Za-z_][\w$]*\s+)?([a-z][\w.]*)\b", re.MULTILINE)
_CATEGORIES = (
    "tensor", "global_load", "global_store", "shared_load", "shared_store",
    "local_load", "local_store", "async_copy", "barrier", "control", "other",
)
_RESOURCE_FIELDS = ("registers_per_thread", "static_shared_memory_bytes", "spills")


def ptx_instruction_sites(ptx: str) -> dict[str, object]:
    """Count static PTX instruction sites, excluding directives and labels."""
    counts: Counter[str] = Counter()
    for raw_line in ptx.splitlines():
        line = raw_line.split("//", 1)[0]
        match = _INSTRUCTION.match(line)
        if not match:
            continue
        if line[match.end():].lstrip().startswith(":"):
            continue
        opcode = match.group(1)
        if opcode.startswith(("mma.", "wgmma.")):
            category = "tensor"
        elif opcode.startswith("ld.global"):
            category = "global_load"
        elif opcode.startswith("st.global"):
            category = "global_store"
        elif opcode.startswith("ld.shared"):
            category = "shared_load"
        elif opcode.startswith("st.shared"):
            category = "shared_store"
        elif opcode.startswith("ld.local"):
            category = "local_load"
        elif opcode.startswith("st.local"):
            category = "local_store"
        elif opcode.startswith("cp.async"):
            category = "async_copy"
        elif opcode.startswith(("bar.", "mbarrier.")):
            category = "barrier"
        elif opcode.startswith(("bra", "brx", "ret", "call", "exit")):
            category = "control"
        else:
            category = "other"
        counts[category] += 1
    return {
        "total": sum(counts.values()),
        "categories": {name: counts[name] for name in _CATEGORIES},
    }


def prompt_triton_compiler_summary(context: Mapping[str, object] | None) -> dict[str, object]:
    """Select only bounded, versioned fields from the worker's compile record."""
    summary: dict[str, object] = {"schema_version": SCHEMA_VERSION, "status": "unavailable"}
    if not isinstance(context, Mapping) or context.get("backend") != "triton_h100" \
            or context.get("schema_version") != 1:
        return summary
    architecture = context.get("architecture")
    if isinstance(architecture, str) and re.fullmatch(r"sm_\d{2,3}", architecture):
        summary["architecture"] = architecture
    resources = context.get("resources")
    if isinstance(resources, Mapping):
        bounded = {key: value for key in _RESOURCE_FIELDS
                   if isinstance((value := resources.get(key)), int)
                   and not isinstance(value, bool) and value >= 0}
        if bounded:
            summary["compiler_resources"] = bounded
    ptx = context.get("ptx")
    if isinstance(ptx, Mapping) and ptx.get("status") == "captured":
        artifacts = ptx.get("artifacts")
        if isinstance(artifacts, list) and len(artifacts) == 1 \
                and isinstance(artifacts[0], Mapping):
            digest = artifacts[0].get("sha256")
            if isinstance(digest, str) and re.fullmatch(r"[a-f0-9]{64}", digest):
                summary["ptx_sha256"] = digest
        sites = ptx.get("static_instruction_sites")
        if isinstance(sites, Mapping) and isinstance(sites.get("total"), int) \
                and isinstance(sites.get("categories"), Mapping):
            categories = sites["categories"]
            bounded = {name: value for name in _CATEGORIES
                       if isinstance((value := categories.get(name)), int)
                       and not isinstance(value, bool) and value >= 0}
            if sum(bounded.values()) == sites["total"]:
                summary["ptx_static_instruction_sites"] = {
                    "total": sites["total"], "categories": bounded,
                }
    if "compiler_resources" in summary or "ptx_static_instruction_sites" in summary:
        summary["status"] = "available"
        summary["limitations"] = (
            "PTX counts are static sites, not executed instructions or measured traffic; "
            "Triton n_spills is a compiler-reported count, not spill bytes."
        )
    return summary
