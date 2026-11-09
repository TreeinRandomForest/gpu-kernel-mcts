# Pure-Triton freeform MCTS versus sequential search

The results below use the historical per-call CUDA-event reward. A separate
[Graph-replay cohort](../triton-h100-graph-search/README.md) reruns paired
arms under `triton_h100_cuda_graph_v1`; do not pool the two latency curves.

Six completed GLM-5.3 H100 runs compare freeform MCTS with freeform sequential
refinement at configured and consumed `B_gen=25`, seeds 0–2. Both use the
packaged 4096³ BF16 GEMM root, the same pinned worker image and local package
overlay mechanism, one repair allowance, Messages thinking disabled, zero conversation
history, hidden parent NCU, no compiler summary, no dataflow view, and no H100
resource graph in prompts. Lightweight NCU was still collected. MCTS used
`k_max=3` and depth cap 10; sequential advanced from every valid candidate,
including slowdowns. The exact prompts, outputs, timings, and failures are in
the linked SQLite traces.

| Seed | Sequential best, µs | Best at call | MCTS best, µs | Best at call | Same-worker cuBLAS, µs |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | [200.368](output/seq-glm53-s0-b25.sqlite) | 1 | [201.488](../triton-h100-freeform-mcts/output/glm53-seed0-bgen25-retry.sqlite) | 24 | 178.848 / 178.464, separate workers |
| 1 | [202.320](output/seq-glm53-s1-b25-retry1.sqlite) | 2 | [201.984](output/mcts-glm53-s1-b25-retry1.sqlite) | 8 | 178.912 |
| 2 | [218.224](output/seq-glm53-s2-b25-retry1.sqlite) | 3 | [226.960](output/mcts-glm53-s2-b25-retry1.sqlite) | 25 | 176.624 |

Seeds 1 and 2 each ran both arms on one worker and have identical environment
manifest IDs and overlay hashes within the pair. Seed 1 ran sequential first;
seed 2 ran MCTS first. Seed 0 used separate workers and different overlay
bundle hashes because the sequential prompt fix landed after its MCTS run.
The seed controls search RNG, not a
guaranteed GLM sampling seed. These three samples do not establish a winning
policy. The paired final difference was 0.336 µs in seed 1, with MCTS lower,
and 8.736 µs in seed 2, with sequential lower.

## Best latency by generation budget

Values are the best verified CUDA-event median available by that `B_gen`
call, including failed and repair calls in the denominator.

| Seed | Arm | Call 5, µs | Call 10, µs | Call 25, µs | Valid / 25 | Unique nodes |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | Sequential | 200.368 | 200.368 | 200.368 | 18 | 13 |
| 0 | MCTS | 221.536 | 221.536 | 201.488 | 19 | 20 |
| 1 | Sequential | 202.320 | 202.320 | 202.320 | 22 | 19 |
| 1 | MCTS | 202.544 | 201.984 | 201.984 | 19 | 17 |
| 2 | Sequential | 218.224 | 218.224 | 218.224 | 20 | 13 |
| 2 | MCTS | 236.064 | 227.312 | 226.960 | 22 | 22 |

Sequential found its eventual best by calls 1–3 in each sample. MCTS found
its best at calls 24, 8, and 25. This is a useful efficiency clue, but the
small sample and different prompt protocols prevent a causal policy claim.
MCTS seed 0's late call-24 child jointly widened `BN` and increased
`num_stages`; see the [freeform MCTS run report](../triton-h100-freeform-mcts/README.md).

## Prompt audit

The actual first-generation prompts in all six traces have the same packaged
root source hash, BF16 workload, Triton `launch(A,B,C,M,N,K)` ABI, H100 target,
measured parent latency, and correct same-worker cuBLAS reference. NCU,
compiler, dataflow, and H100 graph views are hidden from the model. The
sequential Triton prompt previously carried the CUDA root ABI; this was
corrected before the runs above and its prompt schema was raised to version 3.

The prompts are still **not byte-identical**. MCTS asks for raw source under
the reserved `freeform_triton` action and uses a compact parent-timing field.
Sequential asks for JSON containing a `freeform` ID, hypothesis, and source;
its benchmark field includes timing samples, and later prompts include
recent acceptance/failure feedback. MCTS generation calls are fresh requests;
sequential was configured with zero history turns. Both protocols preserve
the same candidate admission and `B_gen` accounting rules. A stricter
search-policy comparison would align response format, timing view, and
feedback visibility, then repeat across more samples.

## Interrupted attempt and resource cleanup

An initial seed-1 sequential attempt reached 17 recorded calls, then a
generated candidate's worker evaluation did not return. It was interrupted
and is preserved as `output/seq-glm53-s1-b25.sqlite`; it is excluded above.
The paired runner now has a tested six-minute no-progress watchdog that
interrupts a stalled arm and releases its worker. The completed seed-1 and
seed-2 pairs used fresh `retry1` traces. Nebius showed no remaining VM or
boot disk after the runs.

The later same-worker Triton 3.9 comparison of combined-catalog versus
pure-Triton freeform **MCTS** is reported separately in
[strategy-vs-freeform-triton39.md](strategy-vs-freeform-triton39.md).
