# Ordinary versus multicast B under the wait-one schedule

## Question and method

Does B multicast close the remaining roughly 10 µs gap to cuBLAS in the
three-slot, wait-one Gluon GEMM? The validated Gluon source contains both
B-load paths behind a runtime-uniform `B_MODE` branch. `paired.py` launches
both modes from the **same cubin**, with the same tile, grid, barriers,
prefetch ring, WGMMA schedule, and seed-0 `4096³` BF16 inputs. It also runs
optimized Triton B and cuBLAS in randomized arm order for three CUDA Graph
rounds, 30 samples per arm per round.

The run additionally checked exact output after each timing round and
attempted five launches per B mode for seeds 1 and 2. The selected seed-0
kernel identity and exactness were checked in the ordinary NCU pass.

## Timing and correctness

All four arms passed the initial seed-0 exact check and all three post-round
checks. Ordinary and multicast B had identical cubin SHA-256. The within-round
median latencies were:

| Arm | Round 0 | Round 1 | Round 2 | Median of rounds |
| --- | ---: | ---: | ---: | ---: |
| Wait-one, ordinary B | 184.640 µs | 187.296 µs | 192.496 µs | 187.296 µs |
| Wait-one, multicast B | 187.232 µs | 189.264 µs | 194.448 µs | 189.264 µs |
| Optimized Triton B | 183.280 µs | 185.280 µs | 189.040 µs | 185.280 µs |
| cuBLAS | 174.384 µs | 176.672 µs | 181.232 µs | 176.672 µs |

Multicast was slower by **2.592, 1.968, and 1.952 µs** in the paired rounds.
Seed 1 passed both modes. Ordinary B passed seed 2, but one multicast launch
in the five-launch seed-2 check failed exact comparison. The harness stopped
at that point. A fresh 12-launch seed-2 diagnostic in each mode passed all
comparisons, so the original mismatch is **observed but not reproduced**.
Multicast correctness remains unresolved, and the timing evidence already
argues against selecting it for this fixed workload.

## Profiling limit

The ordinary-mode NCU pass completed and matched the timed cubin. It recorded
174.102 MB DRAM reads, 28.689 MB DRAM writes, and 84.31% tensor-pipe
activity; its Source report executed ordinary `UTMALDG.2D` and no multicast
TMA sites. The multicast NCU replay remained active for more than two minutes
and was stopped. **No multicast traffic number was obtained**, so this study
cannot say whether multicast reduced DRAM or L2 traffic. NCU durations are
not used as latency measurements.

## Operational record and artifacts

The original `run.py` provisioned a worker but its post-readiness instance-IP
lookup timed out during Nebius CLI reauthentication. Its cleanup also failed
until authentication was restored. The prepared harness was then uploaded and
run directly on that same worker. The failed runner record is
`auth-failure.json`; `results.json` records the partial manual recovery,
source and harness hashes, correctness limit, profile status, and verified
cleanup. The instance and boot disk were subsequently deleted, and both
resource lists were empty.

- `manual_output/paired.json`: all 12 timing observations, exact checks, and
  the seed-2 multicast failure.
- `seed2-diagnostic.log`: fresh 12-launch check per B mode.
- `manual_output/ordinary.raw.csv`, `ordinary.sass.csv`, and compiler
  artifacts: completed ordinary NCU evidence. The large `.ncu-rep` is local
  and Git-ignored.
- `analyze.py` regenerates `analysis.json` and marks the multicast profile
  incomplete.

## Decision

Keep ordinary B as the validated fixed-workload choice. Do not add multicast
as a performance search action from this result. If multicast is revisited,
first isolate the intermittent correctness observation and bound NCU replay
time, then remeasure both modes on the shapes where B sharing may matter.
