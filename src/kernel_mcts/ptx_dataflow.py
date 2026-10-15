"""Conservative, operation-independent value dependencies from one PTX entry."""

from __future__ import annotations

import hashlib
import re
import sqlite3
from typing import Mapping


SCHEMA_VERSION = 1
MAX_BYTES = 256_000
MAX_INSTRUCTIONS = 4_000
MAX_BLOCKS = 256
MAX_EDGES = 8_000
MAX_PARAMETERS = 128

_ENTRY = re.compile(r"(?:^|\s)\.entry\s+([^\s(]+)\s*\(", re.M)
_PARAM = re.compile(r"\.param\b[^,\n]*?\s+([A-Za-z_$][\w$]*)(?:\[[^\]]+\])?\s*(?:,|$)")
_REG = re.compile(r"%[A-Za-z_$][\w.$]*")
_LABEL = re.compile(r"^([A-Za-z_$][\w.$]*):$")
_VERSION = re.compile(r"(?m)^\s*\.version\s+(\S+)")
_TARGET = re.compile(r"(?m)^\s*\.target\s+(\S+)")
_REGISTER_OPS = (
    "mov.", "cvt.", "cvta.", "add.", "addc.", "sub.", "subc.",
    "mul.", "mad.", "madc.", "fma.", "div.", "rem.", "neg.",
    "abs.", "min.", "max.", "setp.", "set.", "selp.", "slct.",
    "and.", "or.", "xor.", "not.", "shl.", "shr.", "bfe.",
    "bfi.", "lop3.", "prmt.", "popc.", "clz.", "brev.",
    "activemask.", "shfl.", "vote.", "sad.", "testp.",
    "sqrt.", "rsqrt.", "rcp.", "ex2.", "lg2.", "sin.",
    "cos.", "tanh.", "isnan.", "isspacep.",
)


def _limit(reason: str, sha256: str, entry: str) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION, "status": "unsupported",
        "ptx_sha256": sha256, "entry_point": entry, "reason": reason,
        "nodes": [], "edges": [], "uncertainty": [reason],
    }


def _matching(text: str, start: int, opening: str, closing: str) -> int | None:
    depth = 0
    for index in range(start, len(text)):
        if text[index] == opening:
            depth += 1
        elif text[index] == closing:
            depth -= 1
            if depth == 0:
                return index
    return None


def _operands(text: str) -> list[str]:
    parts: list[str] = []
    begin = 0
    depth = 0
    for index, char in enumerate(text):
        if char in "{[":
            depth += 1
        elif char in "}]":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[begin:index].strip())
            begin = index + 1
    parts.append(text[begin:].strip())
    return parts


def _family(opcode: str) -> tuple[str, str | None]:
    if opcode.startswith("ld.param"):
        return "parameter_load", None
    if opcode.startswith(("cp.async.commit_group", "cp.async.wait_group", "cp.async.wait_all")):
        return "synchronization", None
    if opcode.startswith("cp.async"):
        return "async_copy", "shared"
    if opcode.startswith(("ld.", "st.")):
        space = next((s for s in ("global", "shared", "local", "const") if f".{s}" in opcode), None)
        return ("load" if opcode.startswith("ld.") else "store"), space
    if opcode.startswith(("wmma.", "mma.", "wgmma.", "tcgen05.")):
        space = next((s for s in ("global", "shared") if f".{s}" in opcode), None)
        return "tensor", space
    if opcode.startswith(("bar.", "mbarrier.", "fence.", "membar.")):
        return "synchronization", None
    if opcode.startswith(("bra", "brx", "ret", "exit")):
        return "control", None
    if opcode.startswith(("call", "trap")):
        return "unsupported", None
    if opcode.startswith(("atom.", "red.")):
        return "atomic", None
    if opcode.startswith(_REGISTER_OPS):
        return "arithmetic", None
    return "unsupported", None


def linked_ptx_dataflow(
    connection: sqlite3.Connection,
    run_id: str,
    node_id: str,
    compiler_context: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Read only the one PTX artifact linked to this node, when available."""
    context = compiler_context if isinstance(compiler_context, Mapping) else {}
    ptx = context.get("ptx")
    artifacts = ptx.get("artifacts") if isinstance(ptx, Mapping) else None
    entry = context.get("entry_point")
    if not isinstance(artifacts, list) or not isinstance(entry, str):
        return None
    eligible = [a for a in artifacts if isinstance(a, Mapping)
                and a.get("storage_status") == "stored"
                and isinstance(a.get("sha256"), str)]
    if len(eligible) != 1:
        return None
    sha256 = eligible[0]["sha256"]
    try:
        row = connection.execute(
            """SELECT a.content FROM node_compiler_artifacts n
               JOIN compiler_artifacts a ON a.sha256 = n.sha256
               WHERE n.run_id = ? AND n.node_id = ? AND n.sha256 = ?
               AND a.kind = 'ptx'""",
            (run_id, node_id, sha256),
        ).fetchone()
    except sqlite3.OperationalError:
        return None
    if row is None:
        return None
    return extract_ptx_dataflow(row[0], sha256, entry)


def extract_ptx_dataflow(raw: bytes, sha256: str, entry_point: str) -> dict[str, object]:
    """Return proven within-block def-use edges, never inferred memory traffic.

    Unsupported syntax is reported as partial evidence. The parser does not
    connect memory writes to reads or carry register definitions across blocks.
    """
    if not isinstance(raw, bytes) or not isinstance(sha256, str) or not isinstance(entry_point, str):
        return _limit("invalid_input", str(sha256), str(entry_point))
    if len(raw) > MAX_BYTES:
        return _limit("ptx_size_limit", sha256, entry_point)
    if hashlib.sha256(raw).hexdigest() != sha256:
        return _limit("ptx_hash_mismatch", sha256, entry_point)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return _limit("ptx_not_utf8", sha256, entry_point)
    # NVIDIA-generated PTX uses line comments. Remove them before selecting an
    # entry so a commented-out declaration cannot be mistaken for code.
    clean = re.sub(r"/\*.*?\*/", lambda match: "\n" * match.group().count("\n"),
                   text, flags=re.S)
    clean = "\n".join(line.split("//", 1)[0] for line in clean.splitlines())
    entries = [match for match in _ENTRY.finditer(clean) if match.group(1) == entry_point]
    if len(entries) != 1:
        return _limit("entry_missing_or_ambiguous", sha256, entry_point)
    match = entries[0]
    close_params = _matching(clean, match.end() - 1, "(", ")")
    if close_params is None:
        return _limit("malformed_entry_parameters", sha256, entry_point)
    header_end = clean.find("{", close_params)
    if header_end < 0 or _ENTRY.search(clean, close_params, header_end):
        return _limit("malformed_entry_body", sha256, entry_point)
    body_end = _matching(clean, header_end, "{", "}")
    if body_end is None:
        return _limit("malformed_entry_body", sha256, entry_point)
    params_text = clean[match.end():close_params]
    params = _PARAM.findall(params_text)
    if len(params) > MAX_PARAMETERS or len(set(params)) != len(params):
        return _limit("parameter_limit_or_ambiguity", sha256, entry_point)
    param_ids = {name: f"param_{index}" for index, name in enumerate(params)}
    nodes: list[dict[str, object]] = [
        {"id": f"param_{index}", "kind": "parameter", "abi_index": index,
         "ptx_name": name, "line": None, "resource_class": None}
        for index, name in enumerate(params)
    ]
    edges: list[dict[str, object]] = []
    uncertainties: set[str] = set()
    definitions: dict[str, str] = {}
    scoped_registers: list[set[str]] = []
    block = 0
    fresh_segment = True
    instructions = 0
    body_start_line = clean.count("\n", 0, header_end) + 1
    pending = ""
    pending_line = body_start_line
    for line_number, raw_line in enumerate(clean[header_end + 1:body_end].splitlines(), body_start_line):
        line = raw_line.strip()
        if not line:
            continue
        opens_scope = bool(re.match(r"^\{(?:\s|$)", line))
        closes_scope = line.endswith("}")
        if opens_scope:
            scoped_registers.append(set())
            line = line[1:].strip()
        if closes_scope:
            line = line[:-1].strip()
        if line.startswith(".reg") and scoped_registers:
            scoped_registers[-1].update(_REG.findall(line))
        if not line or (line.startswith(".") and not pending):
            if closes_scope and scoped_registers:
                for reg in scoped_registers.pop():
                    definitions.pop(reg, None)
            continue
        if _LABEL.fullmatch(line):
            if not fresh_segment:
                block += 1
            if block > MAX_BLOCKS:
                return _limit("block_limit", sha256, entry_point)
            definitions.clear()
            uncertainties.add("control_flow_join_or_loop")
            fresh_segment = True
            continue
        if not pending:
            pending_line = line_number
        pending += " " + line
        while ";" in pending:
            statement, pending = pending.split(";", 1)
            statement = statement.strip()
            if not statement or statement.startswith("."):
                continue
            instructions += 1
            fresh_segment = False
            if instructions > MAX_INSTRUCTIONS:
                return _limit("instruction_limit", sha256, entry_point)
            predicated = statement.startswith("@")
            if predicated:
                predicated_parts = statement.split(None, 1)
                if len(predicated_parts) != 2:
                    uncertainties.add("unparsed_instruction")
                    definitions.clear()
                    continue
                statement = predicated_parts[1]
            pieces = statement.split(None, 1)
            if not pieces:
                uncertainties.add("unparsed_instruction")
                continue
            opcode = pieces[0]
            operand_text = pieces[1] if len(pieces) > 1 else ""
            operands = _operands(operand_text)
            family, space = _family(opcode)
            site_id = f"site_{instructions}"
            nodes.append({
                "id": site_id, "kind": family, "opcode": opcode,
                "address_space": space, "resource_class": space or (
                    "compute" if family in ("arithmetic", "tensor", "atomic") else
                    "synchronization" if family == "synchronization" else None),
                "line": pending_line, "block": block, "predicated": predicated,
            })
            if family == "unsupported":
                uncertainties.add("unsupported_instruction")
                definitions.clear()
                continue
            if family == "control":
                uncertainties.add("control_flow_join_or_loop")
                definitions.clear()
                block += 1
                fresh_segment = True
                if block > MAX_BLOCKS:
                    return _limit("block_limit", sha256, entry_point)
                continue
            if not operand_text and family not in ("synchronization",):
                uncertainties.add("unparsed_instruction")
                definitions.clear()
                continue
            if predicated:
                uncertainties.add("predicated_instruction")
                for reg in _REG.findall(operands[0] if operands else ""):
                    definitions.pop(reg, None)
                continue
            has_dest = family in ("parameter_load", "load", "arithmetic", "tensor", "atomic")
            # A tensor store has a pointer as its first operand, not a register destination.
            if family == "tensor" and ".store" in opcode:
                has_dest = False
            if opcode.startswith("red."):
                has_dest = False
            dest_regs = _REG.findall(operands[0]) if has_dest and operands else []
            use_text = ",".join(operands[1:] if has_dest else operands)
            if family == "tensor" and ".mma" in opcode:
                use_text += "," + (operands[0] if operands else "")
            for reg in dict.fromkeys(_REG.findall(use_text)):
                producer = definitions.get(reg)
                if producer is not None:
                    edges.append({"source": producer, "target": site_id,
                                  "kind": "ptx_def_use", "register": reg,
                                  "measured_link_bytes": None})
            if family == "parameter_load" and len(operands) > 1:
                param = operands[1].strip("[] ")
                if param in param_ids:
                    edges.append({"source": param_ids[param], "target": site_id,
                                  "kind": "parameter_provenance", "register": None,
                                  "measured_link_bytes": None})
                else:
                    uncertainties.add("unresolved_parameter")
            if (family in ("load", "store", "atomic", "async_copy")
                    or (family == "tensor" and (".load" in opcode or ".store" in opcode))):
                uncertainties.add("memory_alias_or_runtime_path_unknown")
            for reg in dest_regs:
                definitions[reg] = site_id
            if len(edges) > MAX_EDGES:
                return _limit("edge_limit", sha256, entry_point)
        if closes_scope and scoped_registers:
            for reg in scoped_registers.pop():
                definitions.pop(reg, None)
    if pending.strip():
        uncertainties.add("unterminated_statement")
    version = _VERSION.search(clean)
    target = _TARGET.search(clean)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "partial" if uncertainties else "bounded_straight_line",
        "ptx_sha256": sha256, "entry_point": entry_point,
        "ptx_version": version.group(1) if version else None,
        "target": target.group(1) if target else None,
        "nodes": nodes, "edges": edges,
        "instruction_count": instructions,
        "block_count": block + (0 if fresh_segment else 1),
        "uncertainty": sorted(uncertainties),
        "runtime_execution_counts": None, "physical_link_bytes": None,
    }
