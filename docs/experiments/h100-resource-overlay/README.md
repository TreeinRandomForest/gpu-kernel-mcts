# Generic PTX dataflow → H100 resource overlay, first pass

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
`stmatrix` and `ldmatrix` are recognized as shared-memory operations even
when the underlying PTX def-use parser cannot connect their operands. A
synthetic attention-style test covers global reads, exponentials, warp
exchange/reduction, shared reads, tensor operations, and output stores. A
**real compiled attention kernel remains a required validation case** before
this representation is used to guide search.

| Variant | Median | Registers/thread | Spills | PTX nodes / proven edges | Static shared writes | Source-report local-load sectors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `BN=128`, 4 warps | 224.416 µs | 232 | 0 | 1,078 / 1,440 | 64 | unavailable |
| `BN=256`, 4 warps | 1,777.296 µs | 255 | 256 | 1,825 / 2,485 | 128 | 134,389,760 |
| `BN=256`, 8 warps | 201.280 µs | 213 | 0 | 904 / 1,312 | 16 | 0 |

The slow four-warp kernel's saved PTX has no textual `ld.local` sites, while
the separate-worker NCU Source report records 213 SASS local-load sites and
262 local-store sites, along with heavy local-sector traffic. This is why the
overlay keeps compiler PTX, SASS observations, and measured traffic as
different evidence layers. The PTX graph is `partial`: control-flow joins,
memory aliases, predication, and some opcodes prevent complete value paths.
The larger static graph and spill evidence do not, by themselves, assign a
causal fraction of the 1.78 ms latency to any one hardware link.

Current hardware mappings are conservative. A PTX shared load names the
shared→register path; a `.cg` global load requests L1 bypass but the nominal
resource graph lacks a direct bypass route, so that gap is explicit. Global
loads without a verified cache outcome retain L1/L2/HBM candidates. Tensor
fence/wait instructions are synchronization rather than matrix arithmetic.
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

Next: validate the generic annotation against a real attention kernel;
compare the fast B/slow D generated kernels when their exact PTX and NCU
artifacts are linked; then add symbolic byte counts and value lifetimes where
the compiler provides enough evidence. Only after that should we test this
overlay as optional prompt context. The four hardware link rates and latencies
remain unknown unless a probe truly isolates them.
