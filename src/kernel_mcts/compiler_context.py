"""Versioned, GPU-independent evidence from one CUDA compilation."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Mapping


SCHEMA_VERSION = 1
_FUNCTION = re.compile(r"(?:Compiling entry function\s+'|Function properties for\s+)([^'\s]+)")
_STACK = re.compile(r"(\d+) bytes stack frame,\s*(\d+) bytes spill stores,\s*(\d+) bytes spill loads")
_REGISTERS = re.compile(r"Used\s+(\d+) registers")
_SHARED = re.compile(r"(\d+) bytes smem")


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
    }
