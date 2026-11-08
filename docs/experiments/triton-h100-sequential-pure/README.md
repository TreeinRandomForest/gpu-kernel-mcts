# Pure-Triton sequential H100 pilot

This pilot starts from the packaged BF16 GEMM Triton root and uses the
Section 50 H100 correctness and CUDA-event timing contract. The two seed-0
arms used GLM-5.3 Messages with thinking disabled, the same pinned H100
worker image plus local package overlay, a configured `B_gen=6`, up to three
repairs, zero session-history turns, and no profile, compiler, dataflow, or
hardware-graph prompt context. Lightweight profiles were still collected
after accepted nodes. Each arm used a separate worker.

| Sequential action policy | Best median, µs | `B_gen` used | Valid generations | Unique nodes | Best at call |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pure v20 strategy menu | 200.464 | 2/6 | 2/2 | 3 | 2 |
| Pure-Triton freeform | 201.040 | 6/6 | 6/6 | 6 | 3 |

The strategy-menu run selected the applicable pointer-loop action at call 1
(224.528 µs), then the BN=256 action at call 2 (200.464 µs). Its first
generated source already used `tl.constexpr` for M/N/K, so the separate
specialization action was inapplicable; the finite menu path ended after two
calls. The trace is `strategy-bgen6-seed0.sqlite` and its best source is
`strategy-bgen6-seed0-best.py`. This first smoke predates the change that
persists sequential worker calibration, so its trace has no cuBLAS record.

Freeform call 1 introduced grouped CTA traversal (241.440 µs). Call 2
widened BN to 256 and raised stages from 3 to 4 (201.328 µs). Call 3 loaded
the physical column-major B tile with K-contiguous addresses and transposed
that tile for `tl.dot` (201.040 µs). Call 4 added an even-K fast path
(203.344 µs). Call 5 increased BM to 256 and slowed to 2062.832 µs, yet
remained a valid accepted state. Call 6 returned the exact call-4 source and
reused that node. The trace is `freeform-bgen6-seed0.sqlite`, with best source
`freeform-bgen6-seed0-best.py`; its same-worker correct cuBLAS calibration
was 179.584 µs.

The seed-0 results are implementation smoke tests, not an estimate that either
action policy wins. They differ in actual model calls, source trajectories,
and worker, and the menu arm lacks calibration in its trace. A pure-Triton
freeform output-stage gate rejects Gluon candidates before GPU evaluation; a
selected menu action must pass its stage and source guard before normal
compilation and correctness checks. No Triton dataflow view was used or
inferred.

## Three-seed GLM follow-up

Seeds 1–2 repeated the same configured settings and now persist worker
calibration. A separate-worker seed-1 GPT-6 Sol menu run used the same root,
catalog, H100 image, context settings, and cap, with medium reasoning. Seeds
label independent runs; the model gateway does not promise deterministic
sampling from these seed values.

| Model | Seed | Policy | Best, µs | Correct cuBLAS, µs | `B_gen` used | Valid calls | First call reaching best |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| GLM-5.3 | 0 | Menu | 200.464 | unavailable | 2/6 | 2/2 | 2 |
| GLM-5.3 | 0 | Freeform | 201.040 | 179.584 | 6/6 | 6/6 | 3 |
| GLM-5.3 | 1 | Menu | 243.280 | 178.688 | 6/6 | 2/6 | 5 |
| GLM-5.3 | 1 | Freeform | 198.864 | 177.184 | 6/6 | 5/6 | 4 |
| GLM-5.3 | 2 | Menu | 234.048 | 177.968 | 6/6 | 2/6 | 6 |
| GLM-5.3 | 2 | Freeform | 238.928 | 178.560 | 6/6 | 6/6 | 1 |
| GPT-6 Sol | 1 | Menu | 201.376 | 178.624 | 2/6 | 2/2 | 2 |

In GLM menu seed 1, call 1 returned invalid JSON with a `\ ` escape inside
the JSON `source` string, so no Python source could be extracted or compiled.
Calls 3–4 changed source beyond the BN-only action, and call 6 changed source
beyond the M/N/K-only specialization action; these were rejected by the
source guards before GPU evaluation. Seed 2 had two responses without usable
source and two BN-only guard mismatches before its valid BN child. By
contrast, seed 1 freeform had one actual compile failure and reached its best
on call 4. The Sol seed-1 menu arm produced two valid, guard-compliant edits
without repairs. Its pointer-loop source already marked M/N/K `tl.constexpr`,
so no specialization action remained after BN widening.

The one Sol run suggests better action adherence in this specific comparison;
it does not establish a model ranking. Freeform reached approximately 200 µs
in two GLM seeds, while menu did so in one. Seed 2 missed that range in both
policies. The pure menu can exhaust its finite action path before its cap, so
equal configured `B_gen` does not imply equal calls, GPU work, or model cost.
The exact prompts, outputs, rejected candidates, and timing samples are in
the SQLite traces named by policy, model when Sol, budget, and seed.

The first seed-2 freeform attempt was interrupted during worker startup,
before any generation. Its local process stopped but the Nebius VM and disk
remained running; both were explicitly deleted. The incomplete trace is
`freeform-bgen6-seed2-interrupted.sqlite`. The completed seed-2 result above
comes from a fresh run, and final Nebius instance and disk inventories were
empty.
