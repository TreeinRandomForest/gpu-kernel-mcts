# Selected-node NCU Source report validation — 2026-10-30

One H100 SXM worker evaluated two exact saved Triton crossover sources, then
collected an NCU `full` Source report and a separate `diagnostic_v2` counter
pass for each. Both kernels passed correctness. The VM and disk were deleted.
The report collector is reusable as `python -m kernel_mcts.ncu_source_report`
on a NVIDIA worker; [run.py](run.py) shows how this experiment invoked it for
selected cached candidate sources. It does not run on every MCTS visit.

| Grouped `BN=256` | Four warps | Eight warps |
| --- | ---: | ---: |
| Candidate median, µs | 1,768.528 | 201.536 |
| Registers/thread | 255 | 213 |
| Triton `n_spills` | 256 | 0 |
| SASS `LDL` sites | 213 | 0 |
| SASS `STL` sites | 262 | 0 |
| Local-load sectors, diagnostic replay | 134,389,760 | 0 |
| Local-store sectors, diagnostic replay | 134,791,168 | 0 |
| Total DRAM read bytes, diagnostic replay | 209,474,560 | 186,661,376 |
| Total DRAM write bytes, diagnostic replay | 764,534,272 | 28,752,640 |
| Overall L2 sector hit rate, diagnostic replay | 97.11% | 70.98% |

The four-warp kernel's register-pressure result is corroborated by actual
local-memory instructions and very large local traffic; the eight-warp kernel
has none of either. The NCU Source export includes Triton `candidate.py` lines.
The largest clusters of *executed local instructions* map to line 38
(`acc = tl.dot(a, b, acc)`, 16,777,216) and line 33 (the A-load mask,
16,666,624). These are correlation labels from the optimized binary, not
proof that either high-level statement alone caused the spills. The compiler
may hoist, combine, and schedule operations across source lines.

NCU 2025.1 on this H100 lists local-load L1 lookup-hit and lookup-miss sector
counters as available. The new opt-in `spill_v1` profile set includes them;
their values were not collected in this run, so that set still needs an
end-to-end worker-image validation. The diagnostic pass's L2 hit rate and
DRAM bytes are **whole-kernel totals**, not cache hits or HBM bytes attributed
to spills. The local sector counts include any thread-private local storage.
The separate full and diagnostic passes can differ due to profiler replay.
The subsequent [H100 `spill_v1` validation](../triton-h100-spill-v1-validation/README.md)
collected those local L1 hit/miss values and verified SQLite and prompt delivery
using the current package overlaid on the pinned worker container.

[analysis.json](analysis.json) is the bounded, evidence-labeled summary that a
future prompt experiment can use. [results.json](results.json) records source
hashes, worker manifest, compiler resources, timings, report hashes, and
metric availability. The `.ncu-rep` and exported `.csv` files are kept locally
and Git-ignored; they can be opened with `ncu-ui` or inspected with `ncu
--import`. Their report hashes are in `results.json`. The worker image was
`docker.io/saarora/gpu-kernel-mcts@sha256:d0cd8e7b784ac8828691aa0add0e41d1802e22ef11745780de018afaa12c0469`.
