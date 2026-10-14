# Bounded PTX dataflow extraction plan

This is a diagnostic design for generated CUDA nodes. It does not change MCTS
selection, validity, reward, `B_gen`, profiling policy, or prompt contents.
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

## Conservative dependency rules

1. Split the entry into basic blocks at labels, branches, and exits. Parse PTX
   statements, predicates, opcode, register definitions and uses, address space,
   and pointer operands. Preserve each instruction's PTX location for audit.
2. Track register reaching definitions within a straight-line block. Trace
   kernel pointer parameters by ABI index; label indices 0/1/2 as A/B/C only
   under the verified BF16 GEMM workload contract. A parameter label is
   provenance, not proof of runtime tensor traffic.
3. Emit a value edge only when its producer and consumer are unambiguous.
   Recognize global/shared loads and stores, `cp.async`, barriers, and
   `wmma`/`mma` instructions as operation sites. `cp.async` is a direct
   global-to-shared transfer; do not invent an intermediate register value.
4. Mark dependencies crossing control-flow joins, loops, predicated writes,
   indirect calls, ambiguous pointer arithmetic, or memory aliasing as
   `partial` or `unsupported` with a reason. Do not connect a shared store to a
   shared load unless the address and ordering proof is implemented and tested.
5. Keep static sites, source-level demand, and measured NCU traffic in separate
   fields. Unknown bytes, dynamic execution counts, cache behavior, and physical
   H100 link traffic remain null. SASS categories can corroborate instruction
   families but cannot prove PTX value edges.

## Integration and acceptance

- Add GPU-independent parser tests for straight-line def-use, parameter
  provenance, `cp.async`, predication, loops, alias ambiguity, entry selection,
  malformed PTX, and all extraction limits. Test that unsupported cases produce
  explicit status instead of a plausible complete graph.
- Version the node graph schema and derive it once from the stored PTX artifact
  when a valid node is created. Preserve exact-source BF16 models. Read-only
  derivation for old traces may use the same artifact and must not write to the
  trace or trigger recompilation.
- Manually inspect several generated kernels with different global/shared and
  tensor-core paths. Compare PTX operation sites with SASS categories; use
  same-worker NCU byte/sector observations only for runtime reconciliation.
  Record mismatches and unsupported regions before displaying path claims or
  enabling an off-by-default prompt summary.
- Run the dataflow prompt ablation only after that inspection supports the
  summary. Keep prompt provenance and equal-budget, matched-seed settings so
  it can be compared with the compiler-only arm.

Implement this after the active Sol prompt comparisons, or in an isolated
worktree, because later experiment arms launch from the current working tree.
