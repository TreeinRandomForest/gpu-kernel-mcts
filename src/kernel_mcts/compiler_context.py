"""Versioned, GPU-independent evidence from one CUDA compilation."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Mapping


SCHEMA_VERSION = 2
PROMPT_SUMMARY_VERSION = 1
MAX_PTX_ARTIFACT_BYTES = 1_000_000
PROMPT_RESOURCE_FIELDS = (
    "registers_per_thread", "static_shared_memory_bytes", "stack_frame_bytes",
    "spill_store_bytes", "spill_load_bytes",
)
PROMPT_SASS_CATEGORIES = (
    "global_load", "global_store", "shared_load", "shared_store",
    "tensor", "arithmetic", "control", "other",
)
_FUNCTION = re.compile(r"(?:Compiling entry function\s+'|Function properties for\s+)([^'\s]+)")
_STACK = re.compile(r"(\d+) bytes stack frame,\s*(\d+) bytes spill stores,\s*(\d+) bytes spill loads")
_REGISTERS = re.compile(r"Used\s+(\d+) registers")
_SHARED = re.compile(r"(\d+) bytes smem")
_SASS_FUNCTION = re.compile(r"^\s*Function\s*:\s*(\S+)", re.MULTILINE)
_SASS_OPCODE = re.compile(r"/\*[0-9a-fA-F]+\*/\s*(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]*)")


def prompt_compiler_summary(context: Mapping[str, object] | None) -> dict[str, object]:
    """Bounded parent-compilation evidence suitable for an opt-in prompt."""
    summary: dict[str, object] = {
        "schema_version": PROMPT_SUMMARY_VERSION,
        "status": "unavailable",
    }
    if not isinstance(context, Mapping) or context.get("schema_version") != SCHEMA_VERSION:
        return summary
    architecture = context.get("architecture")
    if isinstance(architecture, str) and re.fullmatch(r"sm_\d{2,3}", architecture):
        summary["architecture"] = architecture
    ptxas = context.get("ptxas")
    if isinstance(ptxas, Mapping) and ptxas.get("status") == "parsed":
        resources = ptxas.get("resources")
        if isinstance(resources, Mapping):
            bounded = {
                key: value for key in PROMPT_RESOURCE_FIELDS
                if isinstance((value := resources.get(key)), int)
                and not isinstance(value, bool) and value >= 0
            }
            if bounded:
                summary["ptxas_resources"] = bounded
    sass = context.get("sass")
    if isinstance(sass, Mapping) and sass.get("status") == "parsed":
        categories = sass.get("categories")
        if isinstance(categories, Mapping):
            bounded = {
                key: value for key in PROMPT_SASS_CATEGORIES
                if isinstance((value := categories.get(key)), int)
                and not isinstance(value, bool) and value >= 0
            }
            if bounded:
                summary["sass_static_instruction_sites"] = bounded
    if "ptxas_resources" in summary or "sass_static_instruction_sites" in summary:
        summary["status"] = "available"
        summary["limitations"] = (
            "SASS counts are static instruction sites, possibly including inactive branches; "
            "they are not execution counts, tensor dependencies, or measured memory traffic."
        )
    return summary


def sass_instruction_categories(output: str, entry_point: str) -> dict[str, object] | None:
    """Count a small set of instruction families for the requested kernel."""
    if not _SASS_FUNCTION.search(output):
        return None
    in_entry = False
    counts = {name: 0 for name in (
        "global_load", "global_store", "shared_load", "shared_store",
        "tensor", "arithmetic", "control", "other",
    )}
    for line in output.splitlines():
        function = _SASS_FUNCTION.match(line)
        if function:
            in_entry = entry_point in function.group(1)
            continue
        if not in_entry:
            continue
        opcode_match = _SASS_OPCODE.search(line)
        if not opcode_match:
            continue
        opcode = opcode_match.group(1).split(".", 1)[0]
        if opcode.startswith("LDG"):
            category = "global_load"
        elif opcode.startswith("STG"):
            category = "global_store"
        elif opcode.startswith("LDS"):
            category = "shared_load"
        elif opcode.startswith("STS"):
            category = "shared_store"
        elif opcode.startswith(("HMMA", "MMA", "WGMMA", "IMMA")):
            category = "tensor"
        elif opcode.startswith(("BRA", "BRX", "JMP", "EXIT", "RET", "CALL")):
            category = "control"
        elif opcode.startswith(("F", "I", "D", "H")):
            category = "arithmetic"
        else:
            category = "other"
        counts[category] += 1
    total = sum(counts.values())
    return {"instruction_count": total, "categories": counts} if total else None


def parse_ptxas_resources(output: str, entry_point: str) -> Mapping[str, int] | None:
    """Read the entry kernel's resource report, ignoring other device functions."""
    current_function: str | None = None
    resources: dict[str, int] = {}
    for line in output.splitlines():
        function = _FUNCTION.search(line)
        if function:
            current_function = function.group(1)
        if current_function != entry_point:
            continue
        stack = _STACK.search(line)
        if stack:
            resources.update(
                stack_frame_bytes=int(stack.group(1)),
                spill_store_bytes=int(stack.group(2)),
                spill_load_bytes=int(stack.group(3)),
            )
        registers = _REGISTERS.search(line)
        if registers:
            resources["registers_per_thread"] = int(registers.group(1))
        shared = _SHARED.search(line)
        if shared:
            resources["static_shared_memory_bytes"] = int(shared.group(1))
    return resources or None


def cuda_compiler_context(
    *,
    artifact_dir: Path,
    artifact_id: str,
    architecture: str,
    entry_point: str,
    compiler_output: str,
) -> dict[str, object]:
    """Record provenance and parsed evidence without retaining raw PTX in metadata."""
    ptx_artifacts = []
    for path in sorted(artifact_dir.glob("candidate*.ptx")):
        if not path.is_file():
            continue
        raw = path.read_bytes()
        ptx_artifacts.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size_bytes": len(raw),
                "storage_status": (
                    "eligible" if len(raw) <= MAX_PTX_ARTIFACT_BYTES else "too_large"
                ),
            }
        )
    resources = parse_ptxas_resources(compiler_output, entry_point)
    return {
        "schema_version": SCHEMA_VERSION,
        "backend": "cuda_cpp",
        "architecture": architecture,
        "entry_point": entry_point,
        "artifact_id": artifact_id,
        "ptx": {
            "status": "captured" if ptx_artifacts else "missing",
            "artifacts": ptx_artifacts,
        },
        "ptxas": {
            "status": "parsed" if resources else "missing",
            "resources": resources,
            "diagnostic_sha256": hashlib.sha256(compiler_output.encode()).hexdigest(),
        },
        "sass": {"status": "pending"},
    }
