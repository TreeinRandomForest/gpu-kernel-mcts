"""Paired one-/two-CTA CuTe GEMM with retained JIT binary evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import struct
import subprocess
from pathlib import Path

from kernel_mcts import cute_baseline
from kernel_mcts.cute_schedule import CuteSchedule


SCHEDULES = {"single": CuteSchedule(128, 256, 1, 1),
             "cluster_b_multicast": CuteSchedule(128, 256, 2, 1)}


def embedded_binaries(ir_text: str) -> tuple[bytes, ...]:
    pattern = r'llvm\.mlir\.global[^\n]*@kernels_binary\("((?:\\.|[^"\\])*)"\)'
    return tuple(decode_mlir_blob(match.group(1))
                 for match in re.finditer(pattern, ir_text))


def decode_mlir_blob(value: str) -> bytes:
    result = bytearray()
    index = 0
    while index < len(value):
        if value[index] != "\\":
            result.extend(value[index].encode())
            index += 1
        elif index + 2 < len(value) and all(
            character in "0123456789abcdefABCDEF" for character in value[index + 1:index + 3]
        ):
            result.append(int(value[index + 1:index + 3], 16))
            index += 3
        elif index + 1 < len(value) and value[index + 1] in ('\\', '"'):
            result.append(ord(value[index + 1]))
            index += 2
        else:
            raise ValueError("unexpected MLIR string escape in JIT binary")
    data = bytes(result)
    if len(data) < 16 or data[:4] != b"\x50\xed\x55\xba":
        raise ValueError("JIT binary is not a CUDA fatbinary")
    payload_length = struct.unpack_from("<Q", data, 8)[0]
    if len(data) != 16 + payload_length:
        raise ValueError("decoded CUDA fatbinary length mismatch")
    return data


def save(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "experiment": "cute_2cta_b_multicast",
              "orders": {}, "observations": [], "complete": False}
    original_describe = cute_baseline.describe_kernel_callable
    current = {"name": "", "round": -1, "artifacts": []}

    def capture(callable, keywords):
        evidence = original_describe(callable, keywords)
        ir = getattr(callable, "ir_module", None)
        raw_ir = str(ir) if ir is not None else ""
        for index, data in enumerate(embedded_binaries(raw_ir)):
            path = args.output.parent / f"{current['name']}-r{current['round']}-{index}.fatbin"
            path.write_bytes(data)
            record = {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(),
                      "size_bytes": len(data)}
            elf_offset = data.find(b"\x7fELF")
            if elf_offset < 0:
                raise ValueError("CUDA fatbinary has no embedded cubin")
            cubin_path = path.with_suffix(".cubin")
            cubin = data[elf_offset:]
            cubin_path.write_bytes(cubin)
            record["cubin_sha256"] = hashlib.sha256(cubin).hexdigest()
            record["cubin_size_bytes"] = len(cubin)
            disassembly = subprocess.run(
                ["nvdisasm", "-c", str(cubin_path)], capture_output=True, text=True,
                timeout=120,
            )
            if disassembly.returncode == 0:
                sass_path = path.with_suffix(".sass")
                sass_path.write_text(disassembly.stdout)
                record["sass_path"] = str(sass_path)
                record["sass_sha256"] = hashlib.sha256(disassembly.stdout.encode()).hexdigest()
                record["utmaldg_mentions"] = disassembly.stdout.upper().count("UTMALDG")
                record["multicast_mentions"] = disassembly.stdout.upper().count("UTMALDG.3D.MULTICAST")
            else:
                record["nvdisasm_error"] = disassembly.stderr[-400:]
            current["artifacts"].append(record)
        return evidence

    cute_baseline.describe_kernel_callable = capture
    try:
        for round_index in range(3):
            order = list(SCHEDULES)
            random.Random(4127 + round_index).shuffle(order)
            report["orders"][str(round_index)] = order
            for name in order:
                current.update(name=name, round=round_index, artifacts=[])
                print(f"round {round_index} {name}", flush=True)
                result = cute_baseline.run_hopper_bf16_comparable(
                    schedule=SCHEDULES[name], capture_jit_diagnostics=True,
                )
                if not result["correctness"]["success"] or len(result["benchmark"]["timings_us"]) != 30:
                    raise RuntimeError("CuTe evaluation contract failed")
                report["observations"].append({
                    "round": round_index, "name": name,
                    "schedule": SCHEDULES[name].as_dict(),
                    "correctness": result["correctness"],
                    "benchmark": result["benchmark"],
                    "example_sha256": result["example_sha256"],
                    "jit_diagnostics": result["jit_diagnostics"],
                    "artifacts": current["artifacts"],
                })
                save(args.output, report)
                print(name, result["benchmark"]["median_us"], flush=True)
        report["complete"] = True
        save(args.output, report)
    finally:
        cute_baseline.describe_kernel_callable = original_describe


if __name__ == "__main__":
    main()
