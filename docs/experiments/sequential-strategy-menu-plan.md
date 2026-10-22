# Sequential strategy-choice experiment and hybrid-MCTS handoff

## Question and existing evidence

Does requiring the model to choose a strategy from the current catalog change
the quality of **sequential** CUDA refinement when the rest of its context and
verification loop are held fixed? The single-seed, two-prior-turn freeform Sol
run reached 655.856 µs at call 44 of 50. A raw Sol MCTS run reached
734.448 µs; a later compiler-only MCTS run reached 724.4 µs. Those runs
used different prompts, profiling schedules, search policies, and token
budgets, so their best latencies do not isolate a strategy effect.

The [50-call sequential trace](sequential-sol-bgen50-seed0.md) already suggests
catalog gaps at tensor-core bootstrapping and CTA grid remapping. Most other
productive edits fit existing themes, often with coupled changes. Use that
trace for transition labeling now; reserve causal claims for matched runs.

## Controlled arms

Run a 2 × 2 × 2 design: `history_turns ∈ {2, 50}` ×
`hardware_graph ∈ {off, on}` × `strategy_choice ∈ {freeform, catalog}`.
With `B_gen=50`, a 50-turn bound retains the full visible history accumulated
within each run. The first matched pair is **two turns, no H100 graph**,
freeform versus catalog. Repeat that pair over declared seeds before ranking
it, then extend to the graph and full-history cells. Each cell needs its own
fresh freeform control; the historical 655.856 µs run is context, not a
matched replicate of a new menu arm.
Matching controller seeds does not pair model samples or subsequent parent
states; repeat arms and compare distributions and trajectories.

Hold fixed within each pair: `gpt-5.6-sol` at the selected reasoning setting
if still available, root source and workload, `B_gen=50`, one repair, the
pinned worker image, tier-0 correctness/timing contract, `lightweight_v1`
NCU collection and full visibility, bounded `ptxas`/SASS compiler summary,
PTX dataflow `paths` with limits, and no best-so-far source. The graph toggle
adds only the versioned H100 architecture graph. The current compiler
summary does **not** include raw PTX or SASS; `paths` contains selected
static PTX dependencies, not measured traffic. Keep this representation
fixed throughout the strategy experiment.

Use one model call to **choose and apply** a catalog strategy. Repairs consume
their ordinary `B_gen` calls and retain the original chosen ID. Changing to a
separate strategy-selection call would change the budget and belongs in a
different experiment. The catalog arm uses a frozen, hashed copy of
`configs/strategies.yaml`; do not revise its wording between matched arms.
Valid candidates become current even when locally slower, and every newly
accepted distinct node is profiled under the existing sequential policy.

## Implementation work for the first pair

1. In `sequential_generation.py`, make the chosen action explicit in the JSON
   contract: `selected_strategy_id`, `hypothesis`, and complete `source`.
   The freeform arm uses the reserved ID `freeform`; the menu arm requests
   exactly one ID from the catalog. Keep the shared prompt scaffold identical
   where possible. Log a missing or out-of-catalog ID as a **prompt-adherence
   violation**, not as a compile/correctness failure or a reason to discard an
   otherwise valid candidate. Count all such model calls in `B_gen`.
2. In `search_cli.py`, freeze the ordered catalog IDs, rendered menu, catalog
   file hash, and prompt/schema version in run configuration. Record whether
   the menu was enabled. In `sequential.py`, persist each declared ID, its
   adherence status, and the original ID on repair attempts. Keep the node's
   validity, acceptance, evaluation, and profile behavior unchanged.
3. Add GPU-free tests in `tests/test_sequential.py` and
   `tests/test_search_cli.py` for menu rendering, valid/invalid/missing IDs,
   one-call selection plus application, repair ID retention, trace metadata,
   budget accounting, and an unchanged freeform path. Verify the exact prompt
   and run-config fingerprints in a small local fixture before spending GPU or
   model calls.
4. Run a small end-to-end smoke under the normal compile/correctness/timing
   gates, then the first two-turn/no-graph matched pair at `B_gen=50`.
   Record exact prompts, output JSON, validity, benchmark samples, profiles,
   tokens including cache counts, wall time, and worker occupancy. Do not
   infer significance from the first seed.

The existing `--sequential-strategy-menu --strategies PATH` flag already
renders the menu, but today it asks for the choice only in free-text
`hypothesis` and does not parse or persist a selected strategy ID. The first
three items above make that existing path auditable rather than changing the
sequential search rule.

## Analysis and decision gates

For each matched cell and seed, compare best-so-far latency versus `B_gen`,
best latency, validity and repair rates, selected-strategy frequencies,
adherence violations, accepted-chain depth, profiles, input/output/cache
tokens, and GPU time. Classify source transitions independently of the
declared ID: primary mechanism, supporting mechanisms, restored earlier
states, and genuinely unclassified changes. A worse menu arm can reflect
catalog coverage, restrictive strategy wording, model choice quality, or
additional prompt load; the trace analysis should distinguish these before
revising the catalog.

If recurring productive freeform edits are absent or prohibited by the
catalog, propose narrowly specified strategy additions and rerun the
sequential menu comparison with a **new versioned catalog**. Only after this
study, test the separately enabled hybrid CUDA MCTS defined in Section 48
of `spec.md`: PUCT receives one `freeform_cuda` action alongside the catalog
actions, while progressive widening, UCB, valid-only nodes, and budgets
remain intact. Compare it with strategy-only MCTS under matched MCTS context;
sequential results alone cannot explain the full algorithm gap.
