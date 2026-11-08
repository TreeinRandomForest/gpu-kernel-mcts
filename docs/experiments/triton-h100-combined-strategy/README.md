# H100 combined pure-Triton and Triton/Gluon strategy catalog

The opt-in `configs/triton_h100_combined_v19.json` offers two pure-Triton
actions alongside the v18 staged Gluon path. All branches start from the
packaged Triton root and keep the Section 50 BF16 GEMM correctness and
CUDA-event timing contract.

The first pure-Triton action asks for an advancing A/B pointer loop with a
128×128×64 tile, eight warps, three stages, and a two-dimensional CTA grid.
The second changes only the launch `BN` literal from 128 to 256. That local
edit is eligible only for the measured eight-warp pointer-loop family and is
checked by AST identity outside the literal. Prior paired H100 evidence found
about 200 µs for this family at BN=256/eight warps, and a large spill cliff at
BN=256/four warps. These are fixed-workload observations, not universal tile
rules. The source guards do not prove correctness or speed; normal JIT,
correctness, and timing gates remain mandatory.

The v18 Gluon actions remain available as a separate branch. This catalog
tests whether MCTS can make and compare both source families in one global
tree. A first bounded packaged-root run should verify routing, rejection
details, valid nodes, source stages, and exact budget use. Matched multi-seed
comparisons against the original pure-Triton and v18 Gluon catalogs are later
strategy-set ablations; they must keep the model, root, worker image, prompt
context, and `B_gen` fixed.

## v19 packaged-root smoke

A seed-0 GLM-5.3 MCTS run used `B_gen=14`, `k_max=1`, depth 4, up to three
repairs, and hidden parent NCU profile context while still collecting lazy
profiles. It admitted valid nodes on both branches. The pure-Triton
pointer-loop/tile action produced 258.736 µs; its exact BN-only child reached
245.600 µs. The same tree reached a valid Gluon wait-one node at 252.128 µs.
The best was the pure-Triton wide child; the run used all 14 generation
calls, admitted 11 nodes, and completed 16 iterations. Its trace is
`v19-packaged-root-bgen14-seed0.sqlite` and its best source is
`v19-packaged-root-bgen14-seed0-best.py`. The worker VM and disk were released.
This establishes branch routing and action fidelity in one tree, not a
matched comparison of the catalogs.

## Isolating the remaining pure-Triton gap

The generated 245.600 µs source used runtime `M`, `N`, and `K` kernel
parameters; the earlier roughly 200 µs source used `tl.constexpr` for those
dimensions. A candidate changed only those three annotations. The
`measure_constexpr.py` runner compiled and correctness-checked both sources
on one H100 worker, then alternated timing order over three rounds:

| Round | Runtime dimensions, µs | `tl.constexpr` dimensions, µs | Paired delta, µs |
| --- | ---: | ---: | ---: |
| 0 | 243.584 | 198.656 | -44.928 |
| 1 | 243.856 | 198.448 | -45.408 |
| 2 | 244.064 | 198.720 | -45.344 |

The median paired improvement was **45.344 µs**. Exact source hashes,
correctness, compiler output, timings, worker manifest, and cuBLAS startup
calibration are in `constexpr-paired-results.json`; the VM and disk were
released. This isolates a large local specialization effect under this fixed
4096³ workload. It does not establish the effect for dynamic shapes or a
different source family.

The opt-in `configs/triton_h100_combined_v20.json` adds
`triton_specialize_gemm_dims` after the BN-widening action. Its static guard
requires AST identity outside the three parameter annotations. Ordinary
compile, correctness, and timing gates remain mandatory; model reachability
was tested in the following bounded searches.

## v20 model and packaged-root checks

The first live v20 check started from the explicit v19 wide pure-Triton source
with `B_gen=2`, depth 1, and one repair allowance. MCTS selected a descriptor
action first, then selected `triton_specialize_gemm_dims`. The model's first
specialization response passed the AST guard, compiled, passed correctness,
and became a node. Same-worker CUDA-event median improved from 242.960 to
200.144 µs. The trace is `v20-constexpr-alternate-root-bgen2.sqlite` and
its best source is exported beside it. This proves local model reachability,
not packaged-root reachability.

A seed-0 packaged-root v20 MCTS run then used `B_gen=16`, `k_max=1`, depth 4,
up to three repairs, and the same hidden-parent-profile prompt setting as
v19. It reached the pure-Triton path in seven generation calls:

| Node | Same-worker median, µs | Generation call |
| --- | ---: | ---: |
| Packaged Triton root | 841.456 | — |
| Advancing-pointer 128×128×64/eight-warps child | 256.208 | 3 |
| BN=256 local child | 244.112 | 6 |
| M/N/K `tl.constexpr` local child | **199.072** | 7 |

The correctness-passing same-worker cuBLAS calibration was 178.112 µs. The
v20 run exhausted all 16 calls, admitted 11 nodes, and completed 15 MCTS
iterations; six late serial-Gluon attempts were invalid and did not displace
the pure-Triton best. Its trace is `v20-packaged-root-bgen16-seed0.sqlite`,
with the best source exported beside it. The worker VM and disk were released.
The run demonstrates both branches can be offered while the new pure-Triton
path reaches roughly 200 µs. It is one sampled tree with a larger budget than
the v19 smoke, so these two runs are not a matched catalog ablation.

## Strategy-set control design

`configs/triton_h100_pure_local_v20.json` is an exact subset of the first
three actions in the combined v20 catalog, with identical IDs, prompts,
applicability, and source guards. The v18 staged catalog supplies the
Gluon-only control. Compare those three catalogs from the same packaged root
at the same configured `B_gen`, model, worker image, seed, MCTS parameters,
and hidden-parent-profile prompt setting. Record actual generation calls as
well as the cap: the pure arm may exhaust its finite action space early at
`k_max=1`. The existing combined seed-0 v20 trace is one arm of this first
control check; additional runs must not be described as paired same-worker
measurements unless they share a worker.

## Seed-0 strategy-set control

All three arms used the packaged root, GLM-5.3 Messages with thinking
disabled, the same H100 image and local package overlay, seed 0, configured
`B_gen=16`, `k_max=1`, depth 4, up to three repairs, and hidden parent NCU
profile context. Each arm used a separate H100 worker:

| Catalog | Best CUDA-event median, µs | Correct cuBLAS calibration, µs | `B_gen` used | Valid nodes |
| --- | ---: | ---: | ---: | ---: |
| Pure subset v20 | 200.832 | 177.968 | 2/16 | 3 |
| Gluon-only v18 | 251.920 | 178.432 | 10/16 | 5 |
| Combined v20 | 199.072 | 178.112 | 16/16 | 11 |

The pure subset's first model-generated pointer-loop child already annotated
M/N/K as `tl.constexpr`; BN widening then reached 200.832 µs. The separate
specialization action was correctly inapplicable. The Gluon-only arm needed
seven serial-WGMMA calls before a valid serial child, then one wait-one call;
it exhausted its finite path at ten calls. The combined arm spent additional
calls on both branches after finding its pure-Triton best. The traces are
`control-pure-v20-bgen16-seed0.sqlite`,
`control-gluon-v18-bgen16-seed0.sqlite`, and
`v20-packaged-root-bgen16-seed0.sqlite`. Every run completed and released its
VM and disk. These are separate-worker, single-seed observations. The
1.760 µs pure-versus-combined difference cannot establish a winner; report
best-so-far by actual generation call and repeat seeds before interpreting
search efficiency.

## Three-seed strategy-set control

The seed-1 and seed-2 controls completed with the same configured settings as
seed 0: packaged root, GLM-5.3 Messages with thinking disabled, H100 worker
image and local package overlay, `B_gen=16`, `k_max=1`, depth 4, up to three
repairs, and hidden parent NCU profile context. Each arm used a fresh worker;
all workers and disks were released. CUDA-event medians and correct cuBLAS
calibrations are from each arm's own worker.

| Seed | Catalog | Best, µs | cuBLAS, µs | `B_gen` used | Valid nodes | First call reaching best |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | Pure subset v20 | 200.832 | 177.968 | 2/16 | 3 | 2 |
| 0 | Gluon-only v18 | 251.920 | 178.432 | 10/16 | 5 | 10 |
| 0 | Combined v20 | 199.072 | 178.112 | 16/16 | 11 | 7 |
| 1 | Pure subset v20 | 197.792 | 177.776 | 3/16 | 4 | 3 |
| 1 | Gluon-only v18 | 242.848 | 177.616 | 6/16 | 5 | 6 |
| 1 | Combined v20 | 246.064 | 180.336 | 16/16 | 9 | 6 |
| 2 | Pure subset v20 | 201.872 | 181.344 | 3/16 | 4 | 3 |
| 2 | Gluon-only v18 | 253.920 | 178.064 | 6/16 | 5 | 6 |
| 2 | Combined v20 | 198.192 | 176.336 | 16/16 | 13 | 13 |

The pure subset reached the approximately 200 µs family in every seed. Its
seed-0 pointer-loop child already specialized M/N/K, so the BN action reached
the best in two calls; seeds 1–2 used the explicit specialization action at
call 3. The Gluon-only catalog reached a valid wait-one node in every seed,
but its best was 243–254 µs. The combined catalog reached the approximately
200 µs pure family in seeds 0 and 2; seed 1 stopped at the 246.064 µs
BN-widened node without selecting specialization.

The combined tree's best-so-far paths show the search cost of offering both
branches. In seed 0, calls 3, 6, and 7 improved the pure branch to 256.208,
244.112, and 199.072 µs. In seed 1, calls 1 and 6 reached 258.336 and
246.064 µs; calls 9–16 were serial-Gluon `STRATEGY_MISMATCH` failures, leaving
no call for pure specialization. In seed 2, calls 1, 2, and 13 reached
252.576, 242.736, and 198.192 µs. These failures count against `B_gen` and
create no nodes. The pure and Gluon controls exhausted their finite staged
paths before the 16-call cap, so equal *configured* budget does not mean equal
model calls or GPU time.

This small sample supports a narrow conclusion: the pure three-action path is
reachable consistently, while adding Gluon actions can delay or miss its
final specialization within 16 calls. It does not establish a general search
policy ranking, a paired latency difference, or usefulness across other
workloads. Before broadening the catalog, inspect why the serial rewrite
causes repeated `STRATEGY_MISMATCH` attempts and consider a way to keep one
failing action from consuming most of the remaining generation budget. Such
a change would require a separate spec and search-invariant review.

Seed-1 and seed-2 traces are named
`control-{pure-v20,gluon-v18,combined-v20}-bgen16-seed{1,2}.sqlite`, with
corresponding `-best.py` exports in this directory. The seed-0 combined trace
keeps its earlier `v20-packaged-root-bgen16-seed0.sqlite` name.

## Seed-1 serial guard investigation and rerun

The original combined seed-1 trace's eight late failures were not eight
independent API mistakes. In calls 10–12 and 14–16, the model used kernel
descriptor parameters `A` and `B` consistently in both `tma.async_load` and
`mbarrier.expect`. The serial static guard required the literal names
`a_desc` and `b_desc`, even though the host passed those descriptor objects
to the kernel as `A` and `B`. Calls 11–12 repeated the exact same candidate;
calls 15–16 did likewise. Replaying the stored sources through a guard that
matches the two descriptor parameters used by the TMA loads clears those six
false rejections. Calls 9 and 13 still fail for other malformed API forms.
The corrected guard continues to reject shared-memory byte counts and
nonmatching descriptors; compile and correctness gates are unchanged.

A fresh seed-1 combined rerun used the same catalog, GLM model, image, root,
prompt-context setting, and configured `B_gen=16`, with only the local guard
correction. It is a new sampled run, not deterministic replay of the original
model outputs. The trace is
`control-combined-v20-guard-alias-bgen16-seed1.sqlite` and the best source is
exported beside it. The run admitted a correct serial Gluon node at call 15
with a 494.896 µs CUDA-event median. This demonstrates that a serial proposal
can now reach and pass the GPU gates. The best was a pure-Triton BN-widened
node at call 6, 199.136 µs; its model-generated pointer parent had already
specialized dimensions. The same-worker cuBLAS calibration was 178.496 µs.

The rerun still spent seven calls before its valid serial node: four other
static mismatches, one generation without a usable program, and one compile
failure. Its final wait-one call failed the separate pipeline guard. The
guard correction resolves a false rejection, but does not by itself make the
large serial rewrite reliable or establish a search-policy improvement.
No MCTS selection, retry, or budget-accounting rule changed. The worker VM
and disk were released.

## B_gen=25 image compatibility failure

A later seed-0 combined v20 run configured `B_gen=25`, `k_max=3`, depth 10,
one repair, GLM-5.3, and hidden parent profile context. It accidentally used
the older Triton 3.3.1 worker image from the pure-Triton freeform comparison,
not the Triton 3.9 image used for the descriptor/Gluon validations above.
Its best was 253.968 µs, with four valid children. The descriptor action
consumed 21 of 25 generation calls: all 11 compilation failures reported
`No module named 'triton.tools.tensor_descriptor'`, and the ten repair
responses were classified as pointer Triton rather than descriptor Triton.
No Gluon node was reached. The trace is
`../triton-h100-freeform-compare/output/combined-glm53-s0-b25-catalogv20.sqlite`.
This is an image compatibility failure, not a valid test of the combined
catalog against the Gluon-capable image.

The comparison runner now pins the resolved Triton 3.9 Gluon image digest
and checks the worker manifest's Triton version before generation. The v20
catalog and earlier traces remain unchanged.

A seed-0 four-call smoke on that corrected image used the same combined v20
catalog, GLM-5.3, `k_max=3`, depth 10, and one repair allowance. All four
generation calls produced valid nodes: descriptor Triton at calls 1 and 2,
Gluon MMA-v2 at call 3, and the pure-Triton pointer action at call 4. Its best CUDA-event
median was 256.144 µs. The trace is
`../triton-h100-freeform-compare/output/combined-glm53-s0-b4-catalogv20triton39smoke.sqlite`.
This confirms the catalog can reach Gluon on the image used for its API
validation; it is only a bounded reachability check.

## Corrected-image B_gen=25 seed-0 run

The full seed-0 combined v20 MCTS run used the same corrected Triton 3.9
image, packaged root, GLM-5.3 Messages with thinking disabled, hidden parent
profile, `k_max=3`, depth 10, one repair allowance, and `B_gen=25`. All 25
generation calls returned valid candidates. Nine reused an existing state;
the tree ended with 17 nodes including the root. The action allocation was
10 BN-widening calls, nine descriptor calls, three pointer-rewrite calls,
and three Gluon MMA-v2 port calls. No serial TMA/WGMMA or wait-one child was
generated in this sampled tree.

The best node appeared at call 10 via the pure-Triton pointer rewrite followed
by BN widening. Its CUDA-event median was **198.944 µs**, versus a correct
same-worker cuBLAS calibration of **178.256 µs**. The generated pointer parent
had already specialized M/N/K as `tl.constexpr`, so the separate specialization
action was inapplicable on that path. The trace and exported best source are
`../triton-h100-freeform-compare/output/combined-glm53-s0-b25-catalogv20triton39.sqlite`
and `../triton-h100-freeform-compare/output/combined-glm53-s0-b25-catalogv20triton39-best.py`.
The VM and disk were released.

This run establishes that the combined catalog can reach a roughly 200 µs
kernel with the larger `k_max=3` search configuration. The earlier freeform
seed-0 run used Triton 3.3.1, so its 201.488 µs best is not a matched-image
strategy comparison. A matched policy ablation needs the same image, worker
calibration, prompt protocol, and budget for both arms.

The subsequent three-seed same-worker Triton 3.9 comparison against
pure-Triton freeform MCTS is in
`../triton-h100-freeform-compare/strategy-vs-freeform-triton39.md`.
