# Wait-one Gluon source under the Triton H100 search contract

**Historical timing cohort:** The runs below used per-call CUDA events around
the Python launcher. `spec.md` Section 50 now defines
`triton_h100_cuda_graph_v1` replay as the Triton H100 search reward. Compare
latencies within the recorded cohort; see the
[timing reconciliation](../triton-h100-gluon-timing-reconcile/README.md) and
[Graph contract validation](../triton-h100-graph-contract-validation/README.md).

This integrates the validated source with the search backend and records one
H100 worker smoke. The scripts are `run.py` and `smoke.py`; the measurement and
environment manifest are in `results.json`.

The adapter at
`src/kernel_mcts/benchmarks/kernels/bf16_gemm_gluon_wait1.py` retains the
validated device `gemm` function from
`docs/experiments/triton-h100-gluon-wait1/kernel.py`. Its host launcher now
has the search ABI `launch(A,B,C,M,N,K)`: A and C are row-major BF16 views;
B is a column-major BF16 `[K,N]` view with strides `(1,K)`. `B.T` supplies
a zero-copy, contiguous physical `[N,K]` view to the existing TMA descriptor.
The launcher checks shape, dtype, strides, full tiles, and the three-stage K
minimum, then returns the compiled Gluon kernel. The packaged ordinary
Triton root remains the default search root.

The standalone Gluon results used a Triton 3.9 worker image. Earlier
Triton search ablations used a different pinned image. This smoke uses the
Gluon-capable image for the adapted search candidate.

`TritonH100Backend` records the compiled kernel name in the artifact and
compiler context. For a lazy profile, NCU selects that exact function name
inside the harness's `KERNEL_MCTS_TARGET` NVTX range. The backend rejects a
CSV report with a different or ambiguous kernel name. This replaces the
fixed `bf16_gemm_triton` selector and lets a valid Gluon source keep its
`gemm` name. The generation prompt now permits `@triton.jit` or `@gluon.jit`
while requiring the same launcher and returned-kernel contract.

Local tests check device-function AST identity, launcher signature and B
transpose, NVTX range lifetime, profile selection and reported identity,
and the updated generation prompt. Relevant tests passed with 67 tests total.

## H100 smoke

On one NVIDIA H100 80GB HBM3 worker with the Gluon-capable Triton 3.9.0 image,
the runner overlaid the local package source and called `TritonH100Backend`
for each candidate. Both compiled, passed the backend correctness check, and
produced cubin and PTX artifacts. The adapted Gluon candidate also matched
`torch.mm` exactly on a direct same-input check. The runner collected 30
CUDA-event measurements after 10 warmups for each implementation:

| Implementation | Median latency | Registers/thread | Static shared memory | Compiler spills |
| --- | ---: | ---: | ---: | ---: |
| Packaged Triton root | 832.784 µs | 48 | 12,288 B | 0 |
| Adapted Gluon wait-one | 269.120 µs | 154 | 147,480 B | 0 |
| PyTorch/cuBLAS | 183.568 µs | — | — | — |

The lazy `lightweight_v1` NCU profile selected function `gemm` in the
`KERNEL_MCTS_TARGET` NVTX range and verified the reported kernel name. Its
summary recorded 89.25% tensor-pipe utilization and 12.5% achieved occupancy.
The profile is a separate replay, not part of the timing measurements.

This is one worker and one timing round per implementation, executed in
sequence. These CUDA-event latencies should be compared under this harness;
the earlier standalone CUDA Graph medians use a different timing contract.
The run used an overlaid local source tree with uncommitted changes, identified
by the bundle SHA-256 in `results.json`. The provisioning context released the
worker; subsequent Nebius instance and disk lists were empty.

The next gate is a model-generated, scaffold-free structural realization that
passes compile, correctness, and timing before search ablations.

## Strategy v2 probes

`configs/triton_h100_strategies_v2.json` defines six stage-aware actions. The
baseline five-action catalog remains separate. The source-stage classifier
follows the JIT kernel returned by `launch`, and MCTS offers only actions
eligible for that parent before computing priors or collecting an expansion
profile. Local tests route the H100-validated Gluon source as one structural
realization under its v2 action, without using a model.

Two bounded GLM-5.3 prompt probes tested the structural action. The first
returned syntactically valid ordinary Triton pointer-load code, so it did not
apply the selected action. The second supplied the validated Gluon source as
an explicit scaffold and returned Gluon code with the same parsed AST as the
scaffold. It was not re-run on H100 because it is the already validated
device and launcher code. Prompts, outputs, source, and token counts are in
`v2-model-probe/` and `v2-model-probe-scaffold/`. The main v2 catalog does
not supply the scaffold; its prompt-alone structural-generation gate remains
open. A scaffold-assisted search arm would need to be labeled separately.
The three v2 Gluon actions now reject a generated kernel whose returned JIT
function is not statically classified as clustered Gluon. The first probe's
pointer-load Triton source would be recorded as `STRATEGY_MISMATCH` before GPU
evaluation, with `B_gen` charged and repair allowed. The second probe passes
this source-stage guard. The guard alone does not certify its pipeline or
persistent scheduling details.

## Scaffold-free repair and smaller actions

The direct v2 clustered-Gluon action received one repair attempt. The first
GLM-5.3 candidate was pointer-load Triton; the repair used Gluon-like syntax
but had no two-CTA launch or WGMMA/barrier structure and did not pass the
source-stage guard. Both generation calls are recorded in `v2-model-repair/`.

The separate `triton_h100_staged_v3` catalog breaks the transition into
descriptor-load Triton, a Gluon MMA-v2 layout port, single-CTA Gluon
TMA/WGMMA, and a two-CTA cluster action. Each structural action declares its
eligible parent stage and required candidate stage. These are source-stage
checks; hardware code generation and numerical behavior still require the
ordinary backend checks.

The first staged action succeeded on its repair call. The generated
descriptor-load Triton source in `v3-descriptor-repair/` compiled and passed
the H100 backend correctness check with reported maximum error 0.0. On the
same worker, 30 CUDA-event samples after 10 warmups gave these medians:

| Implementation | Median latency | Registers/thread | Static shared memory | Spills |
| --- | ---: | ---: | ---: | ---: |
| Packaged Triton root | 834.864 µs | 48 | 12,288 B | 0 |
| Generated descriptor Triton | 300.448 µs | 58 | 73,752 B | 0 |
| PyTorch/cuBLAS | 181.536 µs | — | — | — |

The candidate's lazy `lightweight_v1` NCU profile selected its
`bf16_gemm_tma` kernel. The measurement and environment are in
`v3-descriptor-repair-results.json`. This confirms a descriptor-based source
and a valid measured candidate; the saved PTX summary alone does not prove a
particular TMA SASS instruction. Nebius instance and disk lists were empty
after the runner exited.

The next v3 action, descriptor Triton to single-CTA Gluon TMA/WGMMA, returned
pointer Triton on its first call and no extractable source on repair despite
reporting the full 8,192 output-token budget. The still smaller Gluon MMA-v2
language/layout action returned no source on the first call and descriptor
Triton on repair. Those artifacts are in `v3-single-cta-repair/` and
`v3-mma-v2-repair/`. Neither passed its output-stage guard, so neither was
run on a GPU. A Gluon intermediate remains an open generation and validation
gate before staged MCTS ablations.

## Diagnostic single-CTA bridge and cluster attempt

`single_cta_gluon_fixture.py` is a manually derived diagnostic bridge from the
previously validated Gluon wait-one source. Its host launch uses one CTA and
its shared layouts omit the cluster mappings; the device computation otherwise
comes from that known source. This is evidence that the intermediate is
feasible, not evidence that GLM-5.3 generated it. The H100 backend compiled
it, passed correctness with maximum error 0.0, and selected `gemm` for its
lazy NCU profile. On one worker, 30 CUDA-event samples after 10 warmups gave:

| Implementation | Median latency | Registers/thread | Static shared memory | Spills |
| --- | ---: | ---: | ---: | ---: |
| Packaged Triton root | 835.616 µs | 48 | 12,288 B | 0 |
| Manual single-CTA Gluon fixture | 249.632 µs | 154 | 147,480 B | 0 |
| PyTorch/cuBLAS | 181.424 µs | — | — | — |

The measurements and environment are in `v3-single-cta-fixture-results.json`.
The earlier two-CTA Gluon result came from a different worker, so its median
is not a paired performance comparison with this fixture.

From that validated single-CTA parent, a bounded GLM-5.3 two-CTA cluster
action plus one repair produced two candidates still classified as single-CTA
Gluon, both retaining `num_ctas=1`. The prompts and responses are in
`v3-two-cta-repair/`. Neither candidate was GPU evaluated. The next gate is
a model-reachable cluster edit with explicit ownership and launch changes,
followed by H100 compile, correctness, timing, and artifact inspection.
After these probes, the staged catalog clarified that B uses unicast per-CTA
TMA descriptor loads. The saved prompt JSON and catalog hash in each probe
record identify the exact pre-clarification instructions used for those calls.

## Same-worker CTA crossover and bounded tile action

`run.py --paired-crossover` compiled and checked the one-CTA fixture and the
packaged two-CTA reference, then alternated their benchmark order for five
rounds on one H100. Each round used 10 warmups and 30 CUDA-event samples per
candidate. Both passed correctness with maximum error 0.0. The medians of
round medians were 249.568 µs (one CTA) and 269.168 µs (two CTAs). The five
paired two-minus-one deltas were 21.920, 20.624, 22.256, 13.632, and
19.600 µs; median 20.624 µs. The full samples, compiler contexts, source
hashes, and environment are in `paired-crossover-results.json`. This resolves
the earlier cross-worker comparison for this fixed workload and timing method.

`typed_shapes.py` is an opt-in, pinned-template renderer for two smaller
one-CTA actions: `CTA_N=128` with its matching WGMMA N shape, or `BLOCK_K=32`.
It checks the validated template hash and modifies only the declared source
sites. `run.py --typed-sweep` compiled, checked, and measured both against the
unchanged one-CTA kernel on one H100, again in five randomized-order rounds:

| Candidate | Median of round medians | Median paired delta vs base | Registers/thread | Static shared memory |
| --- | ---: | ---: | ---: | ---: |
| One-CTA base (N=256, BK=64) | 249.392 µs | — | 154 | 147,480 B |
| N=128, BK=64 | 262.704 µs | +13.088 µs | 90 | 98,328 B |
| N=256, BK=32 | 346.944 µs | +97.920 µs | 154 | 73,752 B |

All three passed compile and correctness with maximum error 0.0 and no
reported spills. The samples and source hashes are in `typed-sweep-results.json`.
The smaller resource footprints did not improve latency in this measurement.

The typed renderer exposes these variants through a `B_mut`-charged generator
under the `single_cta_tile_shape` action. `replay_typed_mcts.py` ran a two-step
MCTS smoke using the already measured sweep medians: `B_gen=0`, `B_mut=2`,
three valid nodes, both slower children retained, and the one-CTA root best at
249.392 µs. `typed-mcts-replay.json` records that result. This is a routing
and budget replay without GPU evaluation, not a live MCTS performance result.
The action is a template-backed typed experiment, distinct from the
scaffold-free GLM generation probes. Both Nebius worker and disk lists were
empty after each crossover.

## Local model-written traversal from an alternate Gluon root

`configs/triton_h100_gluon_local_v1.json` contains one source-edit action for
the validated one-CTA Gluon kernel. `--triton-root-source` records that source
as an explicit alternate root. The static guard accepts only a grouped
program-ID mapping and matching one-dimensional launch grid, preserves the
rest of the parsed source, and rejects a stage or edit mismatch before GPU
evaluation while charging the generation call to `B_gen`. It does not prove
coverage; H100 correctness is still mandatory.

The first scaffold-free GLM-5.3 probe generated the grouped `GROUP_M=8` edit
on its first call. `local-traversal-probe-v1/` preserves the prompt, response,
candidate, and token counts. A pure index check covered all 512 output tiles
once. On one H100, both parent and candidate compiled and passed correctness
with maximum error 0.0. Five same-worker paired rounds measured 248.912 µs
for the parent and 249.456 µs for the candidate by median of round medians;
the median paired candidate-minus-parent delta was +0.464 µs. Full samples
and resources are in `local-traversal-paired-v1-results.json`.

The first `B_gen=4` MCTS attempt stopped before generation: the older worker
package in the pinned Gluon image failed to select `gemm` in NCU. Its SQLite
trace, `local-traversal-mcts-bgen4.sqlite`, records `B_gen=0` and the profile
failure. The opt-in local package overlay now runs before root evaluation and
records its bundle hash. It made the same image's root profile succeed.

The first completed alternate-root smoke, in
`local-traversal-mcts-bgen4-overlay.sqlite`, spent all four generation calls.
`GROUP_M=8` compiled and passed correctness at 249.168 µs versus 248.304 µs
for its same-worker root. Three later calls repeated that mapping while
dropping the module docstring, and the edit guard rejected them before GPU
evaluation. The repair feedback and widening prompt were then made more
specific: name the docstring omission, list accepted sibling group sizes and
latencies, and ask for an untried size.

The revised smoke in `local-traversal-mcts-bgen4-context.sqlite` completed
with `B_gen=4`, one root NCU profile, and three valid nodes:

| Source | Same-run median | Status |
| --- | ---: | --- |
| One-CTA Gluon root | 253.536 µs | Valid, best |
| Model `GROUP_M=8` | 254.752 µs | Compiled, correctness passed |
| Model `GROUP_M=16` | 254.352 µs | Compiled, correctness passed |

The second call chose a distinct `GROUP_M=4` after seeing the accepted `8`,
but failed compilation because `/` produced floating-point tile counts. Its
repair chose `16` but omitted the module docstring and was rejected by the
edit guard. The fourth call generated a valid `16` source. The final catalog
prompt now requires integer `//` division, and the guard rejects `/` in the
mapping before GPU evaluation; this final prompt revision has local tests but
has not been GPU rerun. The two valid children were slightly slower than their
root on this worker. Neither smoke demonstrates reaching Gluon from the
packaged Triton root or a latency improvement. Nebius instance and disk lists
were empty after the runs.

## Persistent factorial arm under the search timing contract

`persistent_factorial_fixture.py` preserves the device `gemm` AST from the
256-wide, M-fast, 66-cluster persistent arm of the factorial experiment. Its
launcher converts the search ABI's column-major `B[K,N]` view to a contiguous
physical `B.T[N,K]` descriptor and uses the factorial arm's tile, layout,
worker count, order, and cluster settings. The paired harness can include a
cuBLAS arm with the same CUDA-event timing method and warmup/sample counts as
the search backend. The run was opt-in; the packaged root was not changed.

On one H100 worker, both sources compiled, passed the search correctness check
with maximum error 0, and matched `torch.mm` exactly on the seeded input.
Five randomized-order rounds gave:

| Arm | Median of round medians | Median paired delta vs one CTA |
| --- | ---: | ---: |
| cuBLAS | 182.272 µs | -68.128 µs |
| One-CTA Gluon | 250.800 µs | — |
| Persistent factorial Gluon | 267.760 µs | +17.104 µs |

The persistent arm was slower than the one-CTA arm in all five rounds. The raw
samples, source hashes, compiler contexts, and worker manifest are in
`persistent-factorial-crossover-v2-results.json`. The worker and boot disk
were released after the run.

The earlier 183.728 µs factorial result used CUDA Graph replay; this crossover
uses the search backend's per-call CUDA events. These results establish which
source is faster under the current search timing contract, but do not isolate
kernel execution from launcher and descriptor setup. A paired Graph-versus-
event measurement of these exact adapted sources is the next diagnostic if
the search objective is intended to track replayed device-kernel latency.

## Model-generated Gluon MMA-v2 bridge

The opt-in `triton_h100_gluon_bridge_v4` action narrows the descriptor-Triton
to Gluon transition to a one-CTA pointer-load MMA-v2 kernel. It supplies API
and layout constraints, but no complete Gluon source scaffold. GLM-5.3 first
returned an invalid mixed-language candidate, then a source classified as
Gluon MMA-v2. A source-review repair removed obvious API mistakes. The first
H100 compile identified missing `gl.constexpr` layout annotations; the next
compile identified a 16-threads-per-warp layout where 32 were required. Both
compiler errors were fed back in separate generation calls. All five calls
are counted as `B_gen`; saved prompts, responses, source hashes, and summary
records are in the `v4-mma-v2-*-repair/` directories.

The fifth generated source compiled and passed the fixed-workload correctness
check with maximum error 0. Under the search CUDA-event harness it measured
3276.432 µs, versus 832.224 µs for the packaged Triton root and 182.816 µs
for cuBLAS on that worker. Its compiler context reports 255 registers per
thread and 46 spills; lightweight NCU reports 6.69% tensor-pipe utilization.
This establishes a real model-generated Gluon intermediate, not a performance
improvement. `v4-mma-v2-thread-layout-validation-results.json` contains the
H100 record. The worker and disk were released.

Two local warp-count probes were evaluated on the separate validated one-CTA
WGMMA source. Four warps passed exact correctness but took 1278.800 µs versus
251.216 µs for eight warps in five paired CUDA-event rounds. The 16-warp
variant failed compilation. Neither is a supported search action. Their
records are `v4-bridge-four-warp-results.json` and
`v4-bridge-sixteen-warp-debug-results.json`.

## A measured local Gluon action

Shrinking the generated MMA-v2 tile from 64×128 to 32×64 eliminated compiler
spills (164 registers/thread, zero spills) but raised the median paired
CUDA-event latency from 3296.912 to 5428.752 µs. The five paired differences
had median +2128.096 µs. Both variants compiled and matched `torch.mm`
exactly. This rules out tile shrinking as a performance action for this source;
see `v4-mma-v2-small-tile-results.json`.

The generated MMA-v2 source used the wrong fastest-axis order in its A, B,
and output `BlockedLayout` definitions. A second bounded candidate changed
only those three orders to match the contiguous K axis of row-major A, the
contiguous K axis of column-major B, and the contiguous N axis of row-major C.
Both variants compiled and matched `torch.mm` exactly. Five paired rounds
measured 3302.400 µs for the parent and 2225.216 µs for the corrected source
by median of round medians. The median paired candidate-minus-parent delta
was -1077.184 µs, with the candidate faster in every round. Compiler spills
were 46 and 48 respectively, so the gain is not explained by spill removal.
The full record is `v4-mma-v2-coalesced-layout-results.json`.

`triton_h100_gluon_bridge_local_v5` now combines descriptor staging, the
pointer-load Gluon bridge, and a coalesced-layout action. The last action is
offered only for source with the expected named layout sites and at least one
wrong order. A static AST guard admits only the three order edits before GPU
evaluation; it does not replace correctness. GLM-5.3 produced the corrected
source on its first standalone generation call. The candidate AST matches
the H100-validated fixture; the probe is in `v5-coalesced-layout-repair/`.

A live alternate-root MCTS smoke then used the valid, misordered Gluon source
and the v5 catalog. In one generation call, GLM produced the guarded layout
edit. It compiled, passed correctness, became a node, and improved measured
CUDA-event latency from 3306.896 to 2231.024 µs on the same H100 worker.
The run recorded `B_gen=1`, two nodes, two lightweight profiles, and best
root-normalized reward 0.393549. The best source's parsed AST matches the
standalone validated fixture. The local SQLite trace is
`v5-coalesced-mcts-bgen1-authretry.sqlite` and the exported best is
`v5-coalesced-mcts-bgen1-authretry-best.py`. A first attempt stopped during
Nebius authentication before VM creation and was superseded; its separate
trace is `v5-coalesced-mcts-bgen1.sqlite`. The successful worker and disk were
released.

This MCTS smoke starts from the Gluon intermediate. It proves that the local
action is model-reachable and performance-positive under the current harness;
it does not prove the full packaged-Triton-to-Gluon path in one search run.
The bridge intermediate is still much slower than the Triton root, so a later
Gluon TMA/WGMMA transition is needed before the chain is competitive.

The first v5 bridge-prompt request failed at the Messages transport before a
response (`v5-mma-v2-bridge-repair/summary.json`). A fresh request returned a
source classified as Gluon MMA-v2 on its first call, but static review found
`import gluon`, `tl.program_id` inside `@gluon.jit`, and a missing MMA layout
version. It was not sent to H100. Its prompt, response, and source are in
`v5-mma-v2-bridge-retry/`. The v5 prompt has therefore not yet established a
compile/correctness-valid bridge within the normal one-repair limit.
One bounded static-feedback repair corrected the imports and `gl.program_id`
calls but still omitted required layouts on `gl.arange` and mixed tensors
before `gl.convert_layout`. It was not sent to H100. Its source and response
are in `v5-mma-v2-bridge-static-repair/`. The validated v4 bridge remains the
only GPU-proven model-generated intermediate in this family.

## Scaffold-free packaged-root bridge validation

The separate v6 bridge prompt made the imports and the six layout-bearing
offset vectors explicit without providing a complete Gluon source. GLM-5.3
produced a stage-matching candidate in one call. The H100 search-backend
smoke compiled it, passed correctness, and measured 2230.576 µs; the packaged
root was 834.336 µs and cuBLAS 181.040 µs in that run. The generated source,
prompt, response, and accounting are in `v6-mma-v2-bridge-retry/`; the H100
record is `v6-mma-v2-bridge-validation-results.json`. The preceding sandboxed
Messages request failed at transport before a response and is recorded in
`v6-mma-v2-bridge-probe/`.

The v7 catalog combined that bridge with descriptor staging and the local
coalesced-layout action. Its first packaged-root MCTS test spent three calls
on invalid descriptor candidates, creating no child. The trace is
`v7-packaged-root-mcts-bgen3.sqlite`. Stage review found that the checker
missed descriptor `.load` calls when kernel parameters were named `A` and
`B`; it now traces launch-time descriptor bindings into kernel arguments.
Another candidate had a syntax error. A v8 descriptor prompt specifies the
constructor and load pattern of the previously validated descriptor source.
Its standalone first response matched the descriptor stage; probe records
are in `v8-descriptor-probe/`.

The first v8 packaged-root MCTS run produced a correct descriptor node at
294.896 µs from an 832.272 µs root, then a correct Gluon MMA-v2 node at
1762.656 µs after one compile-failed bridge call and one repair. It was
interrupted after discovering a termination bug: with `max_depth=2`, `k_max=1`,
and one generation call left, MCTS repeated depth-limit visits. Its trace is
`v8-packaged-root-mcts-bgen4.sqlite`; it has valid node and generation records
but no `run_completed` event. The VM and disk were released. Search now tracks
depth-specific exhausted positions, respects the progressive-widening cap
when checking whether an action can generate, and stops when no reachable
proposal remains. Tests cover this case and a transposed node reached at a
shallower depth.

The clean v8 repeat used the same catalog, packaged root, GLM-5.3 Messages
configuration, `B_gen=4`, `k_max=1`, `max_depth=2`, and hidden parent profile.
It admitted the descriptor child on call 1 at 294.928 µs and the Gluon child
on call 2 at 1718.960 µs, both compile/correctness passing. The root was
834.336 µs. Search ended after five iterations with `B_gen=2`, two profiles,
three nodes, and a `run_completed` event; the remaining two calls were
unspent because the depth-limited proposal space was exhausted. The best
kernel remained the descriptor node. The trace is
`v8-packaged-root-mcts-bgen4-complete.sqlite`; its exported best source is
`v8-packaged-root-mcts-bgen4-complete-best.py`. Nebius VM and disk lists were
empty afterward. This establishes model reachability of the full staged path
in one real MCTS run, but the pointer-load Gluon intermediate remains much
slower than the descriptor kernel.

## Serial TMA/WGMMA strategy from the generated Gluon node

The next opt-in action changes the correct pointer-load MMA-v2 source into a
single-CTA, one-buffer TMA/WGMMA kernel. It permits a coordinated 128×256×64
tile and eight warps, with a barrier wait and a WGMMA drain before each buffer
reuse. The prompt gives API call patterns, not a complete source scaffold.
The parent source was exported unchanged from the completed v8 MCTS Gluon
node to `v8-packaged-root-mcts-bgen4-complete-gluon.py` (SHA-256 prefix
`3ade911c03eb`).

The v9 and v10 two-call GLM probes exposed wrong shared-memory, descriptor,
and TMA API forms. A v9 repaired candidate reached H100 compilation but
failed because assigning `warpgroup_mma_wait` inside the K loop changed the
loop-carried accumulator type. A diagnostic copy of the v10 response changed
only three host layout constructor calls to the validated API. It compiled,
passed correctness, and measured 492.544 µs versus an 833.216 µs packaged
Triton root and 181.856 µs cuBLAS on its worker. This diagnostic is
`v10_wgmma_serial_api_fixture.py` with result
`v10-wgmma-serial-api-diagnostic-results.json`; it is a manual API repair,
not a model-only success.

The v11 standalone probe supplied bounded static feedback to GLM and produced
a complete correct source in two generation calls. H100 validation measured
496.336 µs, 154 registers/thread, zero spills, and 114,696 B static shared
memory, with 39.70% tensor-pipe activity. Records are in
`v11-wgmma-serial-probe/`, `v11-wgmma-serial-static-repair/`, and
`v11-wgmma-serial-validation-results.json`. The first live v11 MCTS
alternate-root attempt did not reproduce that result: both generation calls
failed compilation. The first compiler error exposed only an invalid
`triton.bfloat16` reference, so the normal repair fixed that first error but
left other API errors. Its trace is `v11-wgmma-alternate-root-mcts-bgen2.sqlite`.

The opt-in v12 API guard was added to reject known invalid forms before GPU
evaluation and report them together. Its first live run found a prompt-plumbing
bug: the guard detail was logged but omitted from the repair prompt, and GLM
repeated the first source exactly. The bounded detail now reaches the repair
prompt. A second v12 run used that detail and fixed several API calls but
still failed two positional signatures. These separate traces are
`v12-wgmma-guarded-mcts-bgen2.sqlite` and
`v12-wgmma-guard-detail-mcts-bgen2.sqlite`.

Subsequent model-only probes and bounded live checks exposed missing
accumulator assignment, an invalid conversion to a shared layout,
`mbarrier.inval` instead of `mbarrier.invalidate`, and shared-memory `.nbytes`
instead of descriptor `nbytes_per_cta`. The v13–v16 catalogs and probe
directories retain each prompt/response and call count. The static guard now
checks these known API forms; passing it still requires GPU compilation and
correctness. The v14 live run passed the guard after repair but failed
compilation at `mbarrier.inval`. Its trace is
`v14-wgmma-guarded-mcts-bgen2.sqlite`. The v15 probe passed the then-current
guard but was withheld from GPU because source review found `.nbytes` on
shared memory.

The completed v16 live MCTS run used the measured MMA-v2 alternate root,
`B_gen=2`, one repair, `k_max=1`, `max_depth=1`, a hidden parent profile, and
the GLM-5.3 Messages endpoint. Call 1 was rejected before GPU evaluation with
`STRATEGY_MISMATCH` and detailed API feedback. Call 2 compiled, passed
correctness, became a node, and improved same-worker CUDA-event latency from
1719.840 to 500.352 µs (root-normalized reward 1.234675). The child used
154 registers/thread, zero spills, and 114,696 B static shared memory;
lightweight NCU recorded 40.09% tensor-pipe activity. The run has two nodes,
two profiles, a `run_completed` event, and no remaining Nebius VM or disk.
The trace is `v16-wgmma-guarded-mcts-bgen2.sqlite`; the exported best source
is `v16-wgmma-guarded-mcts-bgen2-best.py`. This establishes a
performance-positive model-reachable serial TMA/WGMMA action from the Gluon
alternate root. It does not yet demonstrate the entire packaged-root chain
through WGMMA in one run, or the faster pipelined schedule.

The first full packaged-root v16 run (`v16-packaged-root-full-chain-bgen6-retry.sqlite`)
used seed 0, `B_gen=6`, one repair per proposal, `k_max=1`, and depth 3.
Descriptor staging and the pointer-load Gluon port both compiled, passed
correctness, and became nodes: 832.752 → 294.656 → 2226.688 µs on the
same worker. The serial TMA/WGMMA action consumed the remaining four calls
across two proposals and two repairs. All four failed the static stage/API
guard before GPU evaluation; the final repair's sole reported mismatch was
the mbarrier expected-byte expression. The best remained the descriptor node.
The run completed and its Nebius VM and disk were released. This is evidence
that the first two stages are reachable together, not that the full chain is.

A second seed-0 full-chain run used the same v16 catalog, `B_gen=10`, up to
three repairs per proposal, `k_max=1`, depth 3, and the GLM-5.3 Messages
endpoint. It completed with four valid nodes in order: packaged Triton root
834.016 µs → descriptor Triton 305.584 µs → pointer-load Gluon MMA-v2
1782.160 µs → serial single-CTA Gluon TMA/WGMMA 502.368 µs. The WGMMA
transition used two generation calls (one static mismatch, then a valid
repair), so the entire chain used `B_gen=4` of 10. MCTS stopped after seven
iterations because the depth-three, finite action space was exhausted; the
descriptor node remained the best. The run had three profile calls, a
`run_completed` event, and no remaining Nebius VM or disk. The trace is
`v16-packaged-root-full-chain-bgen10.sqlite`; the serial WGMMA node source
is exported as `v16-packaged-root-full-chain-bgen10-serial-wgmma.py`.
This establishes model reachability of all three transitions in one search,
not a competitive end-to-end latency or a multi-seed success rate.

## Three-slot wait-one follow-up

The opt-in v17 catalog adds a `single_cta_gluon_tma_wgmma_wait_one` action
from a valid single-CTA Gluon source. It asks the model to retain the parent
ABI, tile, descriptors, unicast B, and epilogue while changing the serial K
loop to the three-slot prefetch/refill schedule seen in the measured one-CTA
reference. A narrow static guard rejects outputs that retain the serial
one-buffer loop or omit the ring/barrier/wait-one structure, charging `B_gen`
and passing the rejection details to the ordinary repair prompt. The guard
does not establish safe synchronization or performance. The v17 action has
not yet passed a live GPU search.

The first v17 alternate-root live search used the generated serial node,
seed 0, `B_gen=3`, two repairs, `k_max=1`, and depth 1. Its root measured
492.880 µs on that worker. Calls 1 and 3 passed the source guard but failed
Triton compilation because the model wrote `a_smem.index([pre])` instead of
scalar `a_smem.index(pre)`; call 2 was rejected by an overstrict barrier-init
shape check. No pipelined child was admitted. The trace is
`v17-wait-one-alternate-root-bgen3.sqlite`; the VM and disk were released.
The v18 catalog adds explicit scalar-index guidance, and the guard now rejects
list indices while accepting three explicit constant-index barrier calls.

The completed v18 alternate-root MCTS run used the exported serial WGMMA
source from the full-chain run, seed 0, `B_gen=3`, two allowed repairs,
`k_max=1`, and depth 1. The model's first generation passed the static guard,
compiled, passed correctness, and became a valid node. Same-worker CUDA-event
median improved from 499.296 to 251.984 µs (root-normalized reward 0.683834).
MCTS stopped after three iterations with `B_gen=1` of 3 because the finite
action space and depth were exhausted. The run has two profiles and a
`run_completed` event; Nebius VM and disk were released. Its trace is
`v18-wait-one-alternate-root-bgen3.sqlite` and its exported best source is
`v18-wait-one-alternate-root-bgen3-best.py`. The latency is close to the
separately measured 249.632 µs one-CTA diagnostic reference, but those two
numbers are from different workers and are not a paired performance claim.

The completed packaged-root v18 MCTS run tested all four staged transitions
in one tree with seed 0, `B_gen=10`, at most three repairs, `k_max=1`, and
depth 4. Five valid nodes were admitted in order: pointer Triton root
834.080 µs → descriptor Triton 297.536 µs → pointer-load Gluon MMA-v2
1749.296 µs → serial single-CTA TMA/WGMMA 495.040 µs → three-slot wait-one
Gluon 249.856 µs. The serial transition used three calls: a stage/API
rejection, a compile failure, and a valid repair. The pipeline transition
succeeded in one call. The full path used `B_gen=6` of 10, completed nine
MCTS iterations with five lazy profiles, and ended when depth and finite
action space were exhausted. The wait-one node was the run's best, with a
root-normalized reward of 1.205445. This is one sampled tree, not a success
rate. The same worker's correctness-passing cuBLAS calibration was 177.216 µs,
so the generated wait-one kernel still has a substantial measured gap. The
trace is `v18-packaged-root-full-chain-bgen10.sqlite`; the best
source is `v18-packaged-root-full-chain-bgen10-best.py`. No Nebius VM or disk
remained after completion.

Two more packaged-root v18 runs used the same catalog, model settings,
`B_gen=10`, repair allowance, depth, and `k_max`, with declared seeds 1 and 2.
Each ran on a fresh H100 worker under the same CUDA-event contract. All five
stages compiled, passed correctness, and became nodes in each run:

| Seed | Root | Descriptor | MMA-v2 | Serial WGMMA | Wait-one WGMMA | `B_gen` used |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 834.080 | 297.536 | 1749.296 | 495.040 | **249.856** | 6/10 |
| 1 | 832.576 | 295.456 | 2208.224 | 493.840 | **252.992** | 7/10 |
| 2 | 833.312 | 295.728 | 1722.496 | 497.872 | **251.984** | 7/10 |

All latencies are CUDA-event medians in µs. Each row is a same-worker path;
the rows are not paired measurements against one another. The serial action
required two or three repairs after static mismatch or compilation failure;
the wait-one action succeeded on its first call in all three runs. The three
completed traces are `v18-packaged-root-full-chain-bgen10.sqlite` and its
`-seed1`/`-seed2` counterparts, with best-source exports beside them. All
three workers and disks were released. This small sample establishes repeated
reachability under this prompt and budget; it does not measure general
strategy success probability, compare search policies, or establish a
competitive result against the same-worker cuBLAS calibration.
