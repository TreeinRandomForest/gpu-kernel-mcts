# Exact B/D source crossover (corrected seed 0)

This experiment follows the [Triton compiler × NCU context factorial](../triton-h100-context-factorial/README.md).
It extracts the **best verified nodes** of corrected seed-0 arm B (NCU prompt)
and arm D (NCU + compiler prompt) from their read-only SQLite traces. The
extracted [`trace_manifest.json`](trace_manifest.json) records run/node IDs,
source and PTX SHA-256, binary fingerprints, original timing samples,
compiler resources, and whole-kernel profiles. [`extract.py`](extract.py)
checks the trace hashes before writing exact source and PTX artifacts; it
refuses to overwrite changed artifacts.

Both traced sources use a `128 × 256 × 64` tile, eight warps, three stages,
and grouped scheduling. The source changes are the K-loop form and row/column
mask construction, alongside syntactic changes to zero initialization,
output conversion, comments, and launcher variable names. The saved PTX has
904 sites/1,439 proven edges in B and 901/1,413 in D. Both have eight WGMMA
sites, 44 `cp.async` sites, and no textual local-memory loads/stores. D has
more static `add.s64` sites (116 vs 105) and fewer `or.b64` sites (2 vs 14).
Static counts are not dynamic execution counts or a causal explanation.

[`variants.py`](variants.py) constructs a two-factor crossover from exact B:

| Variant | K loop | Row/column masks |
| --- | --- | --- |
| `B_exact` | Tile index, `k * BK` | Recomputed in loads/store |
| `B_loop_only` | Element offset, `k0 + kk` | Recomputed |
| `B_mask_only` | Tile index, `k * BK` | Hoisted outside loop |
| `B_both` | Element offset, `k0 + kk` | Hoisted |
| `D_exact` | Original D source | Original D source |

`D_exact` is a separate control because it also changes superficial source
details. The runner verifies correctness and measures every source in three
randomized rounds on one pinned H100 worker, capturing PTX/resources each
round and a separate `spill_v1` NCU profile in the first round. It records
raw timing samples, order, and worker provenance in `results.json`; then
deletes the VM and disk. [`analyze.py`](analyze.py) validates those records,
compares paired round medians, and reports bounded PTX/overlay and profile
summaries in `analysis.json`. It does not infer per-instruction traffic.

Run from the repository root only when the source traces are present:

```bash
.venv/bin/python docs/experiments/triton-h100-bd-crossover/extract.py
.venv/bin/python docs/experiments/triton-h100-bd-crossover/variants.py
.venv/bin/python docs/experiments/triton-h100-bd-crossover/run.py
.venv/bin/python docs/experiments/triton-h100-bd-crossover/analyze.py
```

The experiment compares *kernel source effects* on a controlled worker. It
cannot determine why the LLM selected B or D in the original trees, nor can
three rounds establish a general prompt-factor effect.

## Result

All 15 evaluations passed correctness. The three rounds randomized variant
order on the same H100 worker. Times below are medians of the three per-round
benchmark medians; the delta is the median of the three *paired* round
differences from `B_exact`.

| Variant | Round medians (µs) | Summary (µs) | Paired Δ vs B (µs) | Registers/thread | Tensor pipe (%) | Executed instructions |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `B_exact` | 199.120, 201.104, 199.552 | 199.552 | — | 213 | 91.89 | 39,968,768 |
| `B_mask_only` | 199.328, 200.400, 199.296 | 199.328 | −0.256 | 213 | 91.87 | 39,968,768 |
| `B_loop_only` | 228.016, 227.904, 229.440 | 228.016 | +28.896 | 197 | 75.28 | 43,999,232 |
| `B_both` | 227.808, 228.288, 227.504 | 227.808 | +27.952 | 197 | 75.27 | 43,999,232 |
| `D_exact` | 228.656, 228.688, 229.088 | 228.688 | +29.536 | 197 | 75.25 | 43,999,232 |

Replacing B's K-loop form reproduces nearly all of the B/D latency gap in
this source/compiler/workload combination. Hoisting the masks alone leaves
timing and executed instructions essentially unchanged. The loop-form
variants also have lower tensor-pipe utilization and more executed
instructions, even though all five use the same tile, warps, stages, and
147,456 bytes of shared memory. Every variant has zero compiled spills and
zero NCU local-load/store sectors. Each first-round profile recorded 1,048,576
tensor instructions, and the math-pipe throttle counter rises from 0.20 for
the B-loop form to 0.42 for the D-loop form. These are whole-kernel
correlations; the counters do not isolate an instruction-level bottleneck.

The original tree's B/D best-node timings were 202.928/228.784 µs. The
same-worker replay reproduced their ordering and a similar gap. Replay PTX
files have different full SHA-256 values from the original trace artifacts
because their temporary source paths appear in debug metadata. For each exact
control, the PTX text *before* the debug-file boundary matches the original
byte for byte. `analysis.json` records both comparisons. This bounded check
does not prove binary identity; the replay's correctness, timing, compiler
resources, and NCU profiles provide the independent validation.

The result explains a concrete codegen/performance difference between these
two selected kernels. It does not show that the combined prompt caused the
slower K-loop choice, or that loop syntax will have this effect in other
shapes, Triton versions, or workloads. The run's Nebius VM and disk were
deleted and independently listed as empty after completion.
