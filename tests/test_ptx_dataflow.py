from __future__ import annotations

import hashlib

import pytest

from kernel_mcts.ptx_dataflow import (
    MAX_BLOCKS, MAX_BYTES, MAX_INSTRUCTIONS, MAX_PARAMETERS, extract_ptx_dataflow,
)


def graph(body: str, *, params: str = ".param .u64 p0, .param .u64 p1", entry: str = "kernel") -> dict:
    text = f".version 8.7\n.target sm_90\n.visible .entry {entry}({params}) {{\n{body}\n}}\n"
    raw = text.encode()
    return extract_ptx_dataflow(raw, hashlib.sha256(raw).hexdigest(), entry)


@pytest.mark.parametrize("body,kind", [
    ("ld.param.u64 %rd1, [p0];\nld.global.f32 %f1, [%rd1];\n"
     "add.f32 %f2, %f1, %f1;\nst.global.f32 [%rd1], %f2;\nret;", "arithmetic"),
    ("ld.param.u64 %rd1, [p0];\nld.global.f32 %f1, [%rd1];\n"
     "add.f32 %f2, %f1, 0f00000000;\natom.global.add.f32 %f3, [%rd1], %f2;\nret;", "atomic"),
    ("ld.param.u64 %rd1, [p0];\nld.param.u64 %rd2, [p1];\n"
     "mov.u32 %r1, %tid.x;\nmov.u32 %r2, %tid.y;\n"
     "mad.lo.u32 %r3, %r2, 32, %r1;\nmad.lo.u32 %r4, %r1, 32, %r2;\n"
     "cvt.u64.u32 %rd3, %r3;\ncvt.u64.u32 %rd4, %r4;\n"
     "add.u64 %rd5, %rd1, %rd3;\nadd.u64 %rd6, %rd2, %rd4;\n"
     "ld.global.u32 %r5, [%rd5];\nst.global.u32 [%rd6], %r5;\nret;", "store"),
    ("ld.param.u64 %rd1, [p0];\nwmma.load.a.sync.aligned.row.m16n16k16.bf16 "
     "{%r1, %r2}, [%rd1], 16;\nwmma.mma.sync.aligned.row.col.m16n16k16.f32.bf16.bf16.f32 "
     "{%f1}, {%r1}, {%r2}, {%f1};\nret;", "tensor"),
], ids=["elementwise", "reduction", "transpose", "tensor"])
def test_generic_kernels_have_neutral_parameters_and_local_def_use(body: str, kind: str) -> None:
    result = graph(body)
    assert result["status"] in ("partial", "bounded_straight_line")
    assert [n["id"] for n in result["nodes"][:2]] == ["param_0", "param_1"]
    assert any(n["kind"] == kind for n in result["nodes"])
    assert any(e["kind"] == "parameter_provenance" for e in result["edges"])
    assert any(e["kind"] == "ptx_def_use" for e in result["edges"])
    assert all(e["measured_link_bytes"] is None for e in result["edges"])
    assert result["physical_link_bytes"] is None
    assert "A" not in str(result["nodes"]) and "GEMM" not in str(result["nodes"])


def test_async_copy_is_direct_site_without_invented_register_value() -> None:
    result = graph("ld.param.u64 %rd1, [p0];\nld.param.u64 %rd2, [p1];\n"
                   "cp.async.ca.shared.global [%rd2], [%rd1], 16;\n"
                   "cp.async.commit_group;\nbar.sync 0;\nret;")
    copies = [n for n in result["nodes"] if n["kind"] == "async_copy"]
    assert len(copies) == 1
    assert any(n["kind"] == "synchronization" and n["opcode"] == "cp.async.commit_group"
               for n in result["nodes"])
    transfer = copies[0]
    incoming = [e for e in result["edges"] if e["target"] == transfer["id"]]
    assert {e["register"] for e in incoming} == {"%rd1", "%rd2"}
    assert not any(e["source"] == transfer["id"] for e in result["edges"])


def test_predication_and_control_flow_do_not_create_false_dependencies() -> None:
    result = graph("ld.param.u64 %rd1, [p0];\nmov.u32 %r1, 1;\n"
                   "@%p1\tmov.u32 %r1, 2;\nadd.u32 %r2, %r1, 1;\n"
                   "bra NEXT;\nNEXT:\nadd.u32 %r3, %r2, 1;\nret;")
    assert result["status"] == "partial"
    assert "predicated_instruction" in result["uncertainty"]
    assert "control_flow_join_or_loop" in result["uncertainty"]
    assert not any(e["register"] in {"%r1", "%r2"} and e["target"] in
                   {"site_4", "site_6"} for e in result["edges"])


def test_memory_alias_is_unknown_even_with_same_address_register() -> None:
    result = graph("ld.param.u64 %rd1, [p0];\n"
                   "st.global.u32 [%rd1], %r1;\nld.global.u32 %r2, [%rd1];\nret;")
    store = next(n for n in result["nodes"] if n["kind"] == "store")
    load = next(n for n in result["nodes"] if n["kind"] == "load")
    assert "memory_alias_or_runtime_path_unknown" in result["uncertainty"]
    assert not any(e["source"] == store["id"] and e["target"] == load["id"] for e in result["edges"])


def test_only_named_entry_is_parsed_and_hash_is_checked() -> None:
    text = ".version 8.7\n.target sm_90\n.visible .entry other() { mov.u32 %r1, 1; ret; }\n"
    text += ".visible .entry kernel(.param .u64 p0) { ld.param.u64 %rd1, [p0]; ret; }\n"
    raw = text.encode()
    result = extract_ptx_dataflow(raw, hashlib.sha256(raw).hexdigest(), "kernel")
    assert result["entry_point"] == "kernel"
    assert not any(n.get("opcode") == "mov.u32" for n in result["nodes"])
    assert extract_ptx_dataflow(raw, "0" * 64, "kernel")["reason"] == "ptx_hash_mismatch"
    assert extract_ptx_dataflow(raw, hashlib.sha256(raw).hexdigest(), "absent")["reason"] == "entry_missing_or_ambiguous"


def test_malformed_and_bounded_inputs_report_unsupported() -> None:
    assert graph("ld.param.u64 %rd1, [p0]", params=".param .u64 p0")["status"] == "partial"
    raw = b".visible .entry kernel(.param .u64 p0) { ld.param.u64 %rd1, [p0];"
    assert extract_ptx_dataflow(raw, hashlib.sha256(raw).hexdigest(), "kernel")["reason"] == "malformed_entry_body"
    huge = b"x" * (MAX_BYTES + 1)
    assert extract_ptx_dataflow(huge, hashlib.sha256(huge).hexdigest(), "kernel")["reason"] == "ptx_size_limit"


def test_inline_scope_does_not_leak_temporary_registers() -> None:
    result = graph("ld.param.u64 %rd1, [p0];\n"
                   "{ .reg .b64 %tmp;\nmov.b64 %tmp, %rd1;\n"
                   "add.u64 %rd2, %tmp, 8; }\n"
                   "add.u64 %rd3, %tmp, 8;\nld.global.u32 %r1, [%rd2];\nret;")
    scope_add, later_add = [n for n in result["nodes"] if n.get("opcode") == "add.u64"]
    assert any(e["register"] == "%tmp" and e["target"] == scope_add["id"] for e in result["edges"])
    assert not any(e["register"] == "%tmp" and e["target"] == later_add["id"] for e in result["edges"])


def test_unknown_instruction_stops_reaching_definition_chain() -> None:
    result = graph("mov.u32 %r1, 1;\nexotic.op %r2, %r1;\n"
                   "add.u32 %r3, %r2, 1;\nret;")
    assert "unsupported_instruction" in result["uncertainty"]
    assert any(n["kind"] == "unsupported" and n["opcode"] == "exotic.op" for n in result["nodes"])
    assert not any(e["register"] == "%r2" for e in result["edges"])


def test_all_parser_limits_are_explicit() -> None:
    params = ", ".join(f".param .u64 p{i}" for i in range(MAX_PARAMETERS + 1))
    assert graph("ret;", params=params)["reason"] == "parameter_limit_or_ambiguity"
    body = "\n".join("mov.u32 %r1, 1;" for _ in range(MAX_INSTRUCTIONS + 1))
    assert graph(body)["reason"] == "instruction_limit"
    body = "\n".join(f"L{i}:\nmov.u32 %r1, 1;" for i in range(MAX_BLOCKS + 2))
    assert graph(body)["reason"] == "block_limit"
    body = "mov.u32 %r1, 1;\nmov.u32 %r2, 2;\nmov.u32 %r3, 3;\n"
    body += "\n".join("mad.lo.u32 %r4, %r1, %r2, %r3;" for _ in range(2700))
    assert graph(body)["reason"] == "edge_limit"
