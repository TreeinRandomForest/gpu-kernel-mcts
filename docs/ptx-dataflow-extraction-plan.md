# Bounded PTX dataflow extraction plan

This is an operation- and dtype-independent diagnostic design for generated
CUDA nodes. It does not change MCTS selection, validity, reward, `B_gen`,
profiling policy, or prompt contents.
`spec.md` remains authoritative for those behaviors.

## Current evidence and scope

Generated nodes currently store a version-2 graph with workload-level A/B/GEMM/C
relationships and disconnected static SASS site counts. It does not assert
compiled value dependencies or traffic. The SQLite `compiler_artifacts` table
retains hash-checked PTX blobs separately from node metadata, and
`node_compiler_artifacts` links blobs to valid nodes. In the completed Sol
compiler-only seed-0 trace, 42 PTX artifacts range from 3.3 to 74.8 KB; all
contain a branch and 13 contain `cp.async`. Branches and asynchronous copies
therefore belong in the initial uncertainty model.

The first extractor should analyze only the named `.entry` in a hash-verified
PTX blob, with explicit limits on input bytes, instructions, blocks, and emitted
edges. It should record PTX and extractor versions alongside each derived graph.
Its output must remain meaningful for an elementwise kernel, a reduction, or a
transpose without assuming GEMM, BF16, tensor cores, or three tensor arguments.

## Evidence layers

Keep three layers distinct rather than merging them into one asserted path:

1. **Compiled PTX evidence:** parameter slots, register values, basic blocks,
   memory-operation sites and address spaces, compute sites, synchronization,
   and only locally proven value dependencies. This is the reusable core. Use
   operation-neutral IDs such as `param_0` and `global_load_12`.
2. **Optional workload interpretation:** a separately identified adapter may
   map parameter slots to named tensors and add logical operation relationships
   when the workload contract and kernel ABI have been verified. The present
   BF16 GEMM A/B/C model belongs here. If the adapter is absent or the contract
   does not match, retain neutral parameter names and unknown semantics. Never
   infer tensor identity, shape, dtype, or output ownership from parameter
   position alone.
3. **Hardware and runtime overlay:** map proven operation sites to possible
   resource classes (global, shared, register, compute, synchronization), with
   the target hardware graph identified separately. Keep measured NCU counters
   at their documented aggregation level. A possible resource route is not
   proof of physical link traffic or per-tensor bytes.

The graph should expose the provenance and confidence of each relationship:
`ptx_def_use`, `verified_workload_contract`, `static_sass_site`, or
`measured_ncu_aggregate`. A viewer or prompt summary must not render logical
contract edges as if they were compiled dependencies.

## Conservative dependency rules

1. Split the entry into basic blocks at labels, branches, and exits. Parse PTX
   statements, predicates, opcode, register definitions and uses, address space,
   and pointer operands. Preserve each instruction's PTX location for audit.
2. Track register reaching definitions within a straight-line block. Trace
   parameter provenance by ABI slot, including scalar values and pointer-like
   values when supported by the PTX. Do not assume which slots are inputs,
   outputs, or pointers from their positions. Apply A/B/C labels only in the
   separate, verified BF16 GEMM workload adapter. A parameter label is
   provenance, not proof of runtime tensor traffic.
3. Emit a value edge only when its producer and consumer are unambiguous.
   Recognize global/shared/local loads and stores, `cp.async`, barriers,
   ordinary arithmetic, and tensor instructions as operation sites when their
   syntax is supported. `cp.async` is a direct global-to-shared transfer; do
   not invent an intermediate register value. Unsupported instruction families
   remain explicit unknown sites rather than being forced into GEMM stages.
4. Mark dependencies crossing control-flow joins, loops, predicated writes,
   indirect calls, ambiguous pointer arithmetic, or memory aliasing as
   `partial` or `unsupported` with a reason. Do not connect a shared store to a
   shared load unless the address and ordering proof is implemented and tested.
5. Keep static sites, source-level demand, and measured NCU traffic in separate
   fields. Unknown bytes, dynamic execution counts, cache behavior, and physical
   hardware-link traffic remain null. SASS categories can corroborate
   instruction families but cannot prove PTX value edges.

## Integration and acceptance

- Add GPU-independent parser tests for straight-line def-use, scalar and
  pointer-parameter provenance, `cp.async`, predication, loops, alias ambiguity,
  entry selection, malformed PTX, and all extraction limits. Include small
  elementwise, reduction, transpose, and GEMM fixtures with differing dtypes
  and argument counts. Test that the core works without a workload adapter,
  that a mismatched adapter adds no tensor names, and that unsupported cases
  produce explicit status instead of a plausible complete graph.
- Version the node graph schema and derive it once from the stored PTX artifact
  when a valid node is created. Preserve exact-source BF16 models as optional
  workload-specific views, separate from the generic PTX dependency graph.
  Read-only derivation for old traces may use the same artifact and must not
  write to the trace or trigger recompilation.
- Manually inspect several generated kernels with different global/shared and
  compute paths, including kernels without tensor-core instructions. Compare
  PTX operation sites with SASS categories; use same-worker NCU byte/sector
  observations only for runtime reconciliation.
  Record mismatches and unsupported regions before displaying path claims or
  enabling an off-by-default prompt summary.
- Run the dataflow prompt ablation only after that inspection supports the
  summary. Keep prompt provenance and equal-budget, matched-seed settings so
  it can be compared with the compiler-only arm.

Implement this after the active Sol prompt comparisons, or in an isolated
worktree, because later experiment arms launch from the current working tree.
