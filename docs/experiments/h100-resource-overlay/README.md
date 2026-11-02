# Generic PTX dataflow → H100 resource overlay, first pass

The maintained inventory of graph resources, routes, evidence, and unknowns
is [H100 hardware model: current state](../../h100-hardware-model.md).

This is a diagnostic prototype built from saved artifacts. It does not alter
MCTS, candidate verification, prompts, or priors. `build.py` hash-verifies the
exact Triton source and PTX for three grouped-scheduling crossover kernels,
extracts the existing conservative PTX def-use graph, and annotates each
instruction with **candidate** H100 resources. It retains every def-use edge
and every parser uncertainty. A whole-kernel NCU profile stays at whole-kernel
scope; it is never divided among PTX sites. The large per-variant overlay JSON
files are local and Git-ignored, while [summary.json](output/summary.json) is
the bounded, reviewable result.

For MCTS integration, **each valid node needs its own static overlay** tied
to its source hash, PTX hash, and binary fingerprint. The H100 resource graph
and standalone calibrations are shared across nodes. Create the static
overlay after compilation/correctness/timing, alongside compiler and PTX
dataflow artifacts; add the node's NCU evidence only when lazy expansion
profiles it. An unchanged MCTS visit does not rerun the GPU. The current
saved-kernel overlays have `binding=standalone_artifact`; per-node trace
persistence and prompt use require a separate spec-and-implementation change.

The annotation vocabulary is independent of GEMM dimensions and dtype:
global/shared/local reads and writes, asynchronous copies, tensor operations,
scalar/vector operations, warp exchanges, synchronization, and control.
`ldmatrix` is parsed as a shared-memory read with register results that can
feed `mma.sync`; shared-memory aliases and loop-carried values remain unknown. A
synthetic attention-style test covers global reads, exponentials, warp
exchange/reduction, shared reads, tensor operations, and output stores. A
real compiled causal-attention case below validates this vocabulary, while
held-out profile and timing checks remain before search use.

| Variant | Median | Registers/thread | Spills | PTX nodes / proven edges | Static shared writes | Source-report local-load sectors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `BN=128`, 4 warps | 224.416 µs | 232 | 0 | 1,078 / 1,440 | 64 | unavailable |
| `BN=256`, 4 warps | 1,777.296 µs | 255 | 256 | 1,825 / 2,485 | 128 | 134,389,760 |
| `BN=256`, 8 warps | 201.280 µs | 213 | 0 | 904 / 1,439 | 16 | 0 |

The slow four-warp kernel's saved PTX has no textual `ld.local` sites, while
the separate-worker NCU Source report records 213 SASS local-load sites and
262 local-store sites, along with heavy local-sector traffic. This is why the
overlay keeps compiler PTX, SASS observations, and measured traffic as
different evidence layers. The PTX graph is `partial`: control-flow joins,
memory aliases, predication, and some opcodes prevent complete value paths.
The larger static graph and spill evidence do not, by themselves, assign a
causal fraction of the 1.78 ms latency to any one hardware link.

Current hardware mappings are conservative. A PTX shared load names the
shared→register path; a `.cg` global load names a logical L2→registers
L1-bypass route, and `cp.async.cg` names a distinct L2→shared route rather
than the L1 or TMA route. Register-fed `mma.sync` and warp shuffles have
separate candidate routes. All of their bandwidths and latencies remain
unknown. Global loads without a verified cache outcome retain L1/L2/HBM
candidates. Tensor fence/wait instructions are synchronization rather than matrix arithmetic.
Whole-kernel NCU percentages and bytes remain attached to the kernel, not
individual instructions. No calibrated path measurement is multiplied by a
static PTX site count to predict runtime.

Run locally from the repository root:

```bash
.venv/bin/python docs/experiments/h100-resource-overlay/build.py
.venv/bin/python docs/experiments/h100-resource-overlay/visualize.py
```

`visualize.py` creates two smaller views per variant in `output/`:

- `*.overview.svg` groups PTX sites by operation class, labels each class with
  its static site count, and labels class-to-class links with counts of **actual
  recorded PTX edges**. It also shows classes with no proven edges. The links
  must not be chained into a new path: two sites in one class may be unrelated.
- `*.view.json` records the same counts, candidate hardware-link site counts,
  sites without proven edges, path gaps, and parser uncertainty.
- `*.neighborhood.svg` and `*.neighborhood.json` show the exact one-hop PTX
  edges around the first tensor instruction. The SVG collapses parallel edges
  to one labeled arrow; the JSON retains all original edges. This is an
  illustrative local neighborhood, not a complete tensor-input path.

The overview can shrink thousands of instruction sites into roughly a dozen
classes without treating static site counts as executed instructions or
measured bytes. For the slow `BN=256`, four-warp case, 481 of 1,825 PTX sites
have no proven incident def-use edge in this partial parser; the overview
intentionally leaves their classes visible.

For analysis, use [viewer.html](viewer.html) instead of inferring paths from
the class-count SVG. Open the HTML locally and choose one of the generated
`*.overlay.json` files. It keeps the complete overlay in browser memory, lets
you find a site by class/opcode/line, follows exact incoming and outgoing
edges, and traces a directed path to another site if the partial PTX graph
proves one. Each selected site exposes its original annotation and incident
edge records. The viewer does not modify MCTS traces or send data over a
network. A missing proven path is explicitly inconclusive.

Next: compare the fast B/slow D generated kernels when their exact PTX and NCU
artifacts are linked; then add symbolic byte counts and value lifetimes where
the compiler provides enough evidence. Only after that should we test this
overlay as optional prompt context. Hardware-route rates and latencies remain
unknown unless a probe truly isolates them.

## Real attention validation

`attention_inside.py` is a small causal BF16 attention kernel with tiled QK,
online softmax (row max/sum and exponential), and a second dot with V. It has
an independent higher-precision PyTorch correctness reference. On one
disposable H100, `run_attention.py` compiles and verifies it, retains exact
PTX and resource metadata, and attempts a separate whole-kernel NCU traffic
profile. `analyze_attention.py` hash-checks the source and PTX, applies the
same generic overlay, and reports unsupported PTX sites and missing hardware
paths. It leaves physical-link calibration empty. Run from the repository root:

```bash
.venv/bin/python docs/experiments/h100-resource-overlay/run_attention.py
.venv/bin/python docs/experiments/h100-resource-overlay/analyze_attention.py
```

This case is for representation validation, not a performance benchmark or
an MCTS search. It uses one 256-token, 64-dimensional causal head and does
not imply coverage of paged attention, fused multi-head attention, or
production shapes.

The H100 run passed the higher-precision reference at max absolute error
0.00518 (tolerance 0.025 absolute / 0.03 relative), with 119 registers/thread,
24 KiB static shared memory, and zero compiled spills. The saved 46,443-byte
PTX has 750 sites and 1,039 proven def-use/provenance edges after recognizing
24 `ldmatrix` sites; it includes 16 `cp.async.cg` copies, 32 `mma.sync` tensor
sites, 20 warp shuffles, and 20 exponentials. The updated hardware graph has
candidate routes for these sites and no remaining *topology* path gaps in this
case. The dataflow is still partial across shared-memory aliases, loop joins,
and predication; 69 sites have no proven incident edge. See
[attention_analysis.json](output/attention_analysis.json),
[attention PTX](output/attention.ptx), and the
[overview](output/attention.overview.svg).

The separate NCU application replay recorded zero local-memory sectors and
256 DRAM read bytes on a warm, tiny workload; neither is an instruction-level
or link-bandwidth measurement. Its L2 hit-rate field was 101.2%, outside the
valid range, and is flagged rather than interpreted. The physical link rates
and latencies remain uncalibrated.
