# Triton H100 compiler × NCU prompt ablation

The first context ablation crosses visibility of measured `spill_v1` NCU
metrics and a versioned compiler summary. Each arm profiles the same way and
stores full profiles in SQLite. The NCU prompt view masks compiler-derived
fields, so the compiler factor is distinct.

| Arm | NCU metrics in prompt | Compiler summary in prompt |
| --- | --- | --- |
| A | No | No |
| B | Yes | No |
| C | No | Yes |
| D | Yes | Yes |

The compiler summary has registers per thread, static shared-memory bytes,
Triton `n_spills`, and static PTX instruction-site counts. It includes a PTX
hash, limitations, and availability status. It excludes SASS, Source-line
attribution, and graph edges. The count categories are tensor, global/shared/
local loads and stores, asynchronous copy, barrier, control, and other.

Use the packaged Triton root, fixed BF16 GEMM workload, versioned Triton
strategies, GLM-5.3 Messages with thinking disabled, seeds 0/1/2, and
`B_gen=10`. Each arm uses one Nebius H100 image. The run script overlays the
current Python package into the worker container, restarts that worker, and
records the source-bundle hash; it does not publish an image. The GLM key
remains on the local controller.

Run from the repository root with the local EnMaaS environment configured:

```bash
systemd-inhibit --what=sleep:idle --mode=block --who=kernel-mcts \
  --why='Keep H100 search controller active' \
  .venv/bin/python docs/experiments/triton-h100-context-factorial/run.py \
  --resume --label corrected
```

Results are written to `output/`, with one SQLite trace and one bounded JSON
summary per seed and arm. The SQLite traces are Git-ignored. `--resume` skips
closed traces with the complete budget and archives incomplete traces before
rerunning them. The experiment generator counts a transient gateway 502/503/504
or a 120-second hard request deadline as one invalid generation; repeated
failures stop the run and trigger VM cleanup. Credit errors also stop the run.

## Corrected 3-seed results

Best verified CUDA-event median, µs, after `B_gen=10`:

| Seed | A: source | B: NCU | C: compiler | D: NCU + compiler |
| --- | ---: | ---: | ---: | ---: |
| 0 | 201.856 | 202.928 | 203.168 | 228.784 |
| 1 | 201.664 | 201.232 | **200.224** | 234.800 |
| 2 | 201.808 | 202.576 | 203.152 | 202.592 |
| Median across seeds | **201.808** | 202.576 | 203.152 | 228.784 |
| Valid generations / 30 | 29 | 27 | 27 | 27 |

Every arm collected `spill_v1`, with five persisted node profiles per run.
The [machine-readable audit](output/analysis.json) checks all 120 recorded
generation calls, prompt factors, model and thinking mode, metric set, and
run completion. The root timing ranged from 816.14 to 823.22 µs across the
12 runs. No completed run recorded a gateway error. Nebius listed no remaining
VM or disk after the run.

The combined arm was poor in seeds 0 and 1 but close to the others in seed 2.
At three seeds and this short budget, this is a search-behavior clue rather
than a general claim that more context hurts. Seeds share the same initial
RNG seed across arms, but each arm's generated candidates alter its later
tree. The restarts also used multiple fresh H100 VMs with the same pinned
image and current source overlay; exact bundle hashes are recorded per run.

Two observed differences help locate the poor D paths. In seed 1, D's best
source used a `128 × 128 × 64` tile with eight warps and four stages; A/B/C
reached `BN=256` and about 200–201 µs. D's best had 82.76% tensor-pipe
activity versus 91.47% for C, with zero compiler spills and zero local-load
sectors in both. In seed 0, B and D both used `128 × 256 × 64`, eight warps,
and three stages, but their loop and mask source differed. D's best had
75.26% tensor-pipe activity versus 91.62% for B, again with no spills or
local-load sectors. Those are whole-kernel profile observations. The
[corrected seed-0 B/D source crossover](../triton-h100-bd-crossover/README.md)
finds that changing the K-loop form reproduces nearly all of that kernel
latency gap on one worker; mask hoisting alone does not. This does not explain
why either prompt selected its source.

## Interrupted and exploratory traces

The first 12-arm attempt exposed that `launch__registers_per_thread` remained
in the NCU prompt view; those traces are exploratory and excluded from the
corrected table. The run attempts also encountered gateway 502/504 and a
stalled response. Incomplete traces were retained under `output/interrupted/`
or the earlier `factorial-*` names and excluded from the audit. The runner now
has resume, bounded gateway handling, and a hard request deadline. No partial
trace is counted as a completed arm.
