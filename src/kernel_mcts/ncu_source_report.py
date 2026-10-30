"""Collect one selected NVIDIA kernel's NCU Source report outside search."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Callable, Sequence


_LOCAL_LOAD = re.compile(r"\bLDL(?:\.|\b)")
_LOCAL_STORE = re.compile(r"\bSTL(?:\.|\b)")


def summarize_correlated_source(csv_text: str, *, limit: int = 8) -> list[dict[str, object]]:
    """Rank Triton source lines by executed local-memory SASS instructions."""
    rows = csv.reader(csv_text.splitlines())
    header = next((row for row in rows if row and row[0] == "Line No"), None)
    if header is None:
        return []
    source_column = header.index("Source", 3)
    instructions_column = header.index("Instructions Executed")
    line: tuple[int, str] | None = None
    totals: dict[tuple[int, str], dict[str, int]] = {}
    for row in rows:
        if row and row[0].isdigit():
            line = (int(row[0]), row[1].strip())
        if line is None or len(row) <= max(source_column, instructions_column):
            continue
        instruction = row[source_column]
        is_load = bool(_LOCAL_LOAD.search(instruction))
        is_store = bool(_LOCAL_STORE.search(instruction))
        if not (is_load or is_store):
            continue
        record = totals.setdefault(line, {"load_sites": 0, "store_sites": 0,
                                          "executed_local_instructions": 0})
        record["load_sites"] += int(is_load)
        record["store_sites"] += int(is_store)
        count = row[instructions_column]
        if count.isdigit():
            record["executed_local_instructions"] += int(count)
    ranked = sorted(totals.items(), key=lambda item: (
        -item[1]["executed_local_instructions"], item[0][0]
    ))[:limit]
    return [{"line": number, "source": source, **counts}
            for (number, source), counts in ranked]


def has_source_correlation(csv_text: str) -> bool:
    return any(row and row[0].isdigit() and len(row) > 1 and bool(row[1].strip())
               for row in csv.reader(csv_text.splitlines()))


def collect_source_report(
    output_prefix: Path,
    target_command: Sequence[str],
    *,
    kernel_regex: str,
    ncu: str = "ncu",
    timeout_seconds: float = 900.0,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> dict[str, object]:
    """Save the binary report and source exports; return bounded provenance."""
    if not target_command or not kernel_regex:
        raise ValueError("target command and kernel regex are required")
    report = output_prefix.with_suffix(".ncu-rep")
    summary_path = output_prefix.with_suffix(".json")
    exports = {view: output_prefix.with_suffix(f".{view}.csv")
               for view in ("sass", "ptx", "cuda_sass")}
    for path in (report, summary_path, *exports.values()):
        if path.exists():
            raise FileExistsError(path)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ncu, "--set", "full", "--import-source", "yes",
        "--kernel-name-base", "function", "--kernel-name", f"regex:{kernel_regex}",
        "--launch-count", "1", "--export", str(output_prefix),
        *target_command,
    ]
    collected = runner(command, capture_output=True, text=True, timeout=timeout_seconds)
    if collected.returncode != 0 or not report.is_file():
        raise RuntimeError("NCU Source report collection failed: "
                           + (collected.stderr or collected.stdout)[-1000:])
    statuses: dict[str, str] = {}
    for view, path in exports.items():
        source_view = "cuda,sass" if view == "cuda_sass" else view
        result = runner(
            [ncu, "--import", str(report), "--page", "source",
             "--print-source", source_view, "--csv"],
            capture_output=True, text=True, timeout=timeout_seconds,
        )
        if result.returncode == 0:
            path.write_text(result.stdout, encoding="utf-8")
            statuses[view] = "captured"
        else:
            statuses[view] = "unavailable"
    sass = exports["sass"].read_text(encoding="utf-8") if statuses["sass"] == "captured" else ""
    correlated = (exports["cuda_sass"].read_text(encoding="utf-8")
                  if statuses["cuda_sass"] == "captured" else "")
    source_lines = summarize_correlated_source(correlated)
    result = {
        "schema_version": 1,
        "kernel_regex": kernel_regex,
        "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
        "report_size_bytes": report.stat().st_size,
        "source_views": statuses,
        "sass_local_load_sites": len(_LOCAL_LOAD.findall(sass)),
        "sass_local_store_sites": len(_LOCAL_STORE.findall(sass)),
        "source_correlated": has_source_correlation(correlated),
        "top_local_source_lines": source_lines,
    }
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--kernel-regex", required=True)
    parser.add_argument("--ncu", default="ncu")
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("target_command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    target = args.target_command
    if target and target[0] == "--":
        target = target[1:]
    result = collect_source_report(
        args.output_prefix, target, kernel_regex=args.kernel_regex,
        ncu=args.ncu, timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
