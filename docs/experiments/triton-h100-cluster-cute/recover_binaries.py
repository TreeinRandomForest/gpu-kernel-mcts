"""Recover exact CUDA fatbinaries from retained CuTe MLIR string escapes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from paired import embedded_binaries


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"


def main() -> None:
    report = json.loads((OUTPUT / "paired.json").read_text())
    if not report.get("complete") or len(report.get("observations", [])) != 6:
        raise ValueError("paired report incomplete")
    records = []
    for item in report["observations"]:
        blobs = embedded_binaries(item["jit_diagnostics"]["mlir"]["normalized_text"])
        if len(blobs) != 1:
            raise ValueError("expected one embedded CUDA binary")
        data = blobs[0]
        name = f"{item['name']}-r{item['round']}"
        (OUTPUT / f"{name}.recovered.fatbin").write_bytes(data)
        elf_offset = data.find(b"\x7fELF")
        if elf_offset < 0:
            raise ValueError("CUDA fatbinary has no ELF payload")
        cubin = data[elf_offset:]
        (OUTPUT / f"{name}.cubin").write_bytes(cubin)
        records.append({"name": item["name"], "round": item["round"],
                        "fatbin_sha256": hashlib.sha256(data).hexdigest(),
                        "fatbin_size_bytes": len(data),
                        "cubin_sha256": hashlib.sha256(cubin).hexdigest(),
                        "cubin_size_bytes": len(cubin),
                        "elf_offset": elf_offset})
    (HERE / "binary_audit.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n"
    )
    for item in records:
        print(item["name"], item["round"], item["cubin_sha256"])


if __name__ == "__main__":
    main()
