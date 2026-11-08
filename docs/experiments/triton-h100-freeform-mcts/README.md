# Pure-Triton freeform MCTS H100 smoke

One seed-0 GLM-5.3 Messages run exercised the opt-in `freeform_triton` MCTS
action on the packaged 4096³ BF16 GEMM root. The pinned H100 worker image
received a local package overlay. The run used `B_gen=6`, `k_max=3`, depth 3,
one repair allowance, and lightweight NCU profiling. Parent NCU profiles,
compiler summaries, dataflow, and the H100 resource graph were absent from
generation prompts. Messages thinking was disabled. The exact prompts and
outputs are in [the SQLite trace](output/glm53-seed0-bgen6-retry.sqlite); the
[best source](output/glm53-seed0-bgen6-retry-best.py) is saved separately.

| Measure | Result |
| --- | ---: |
| Packaged root CUDA-event median | 814.640 µs |
| Best verified kernel CUDA-event median | 219.824 µs |
| Same-worker correct cuBLAS calibration | 178.144 µs |
| Generation calls / configured `B_gen` | 6 / 6 |
| Valid, compilation/correctness-passing calls | 6 / 6 |
| Unique nodes including root | 7 |
| NCU profile calls | 4 |

| Call | Parent selection | Candidate median, µs | Result |
| ---: | --- | ---: | --- |
| 1 | Expand root | 238.064 | Valid |
| 2 | Expand root | 253.216 | Valid |
| 3 | UCB first root child, then expand | 229.680 | Valid |
| 4 | UCB second root child, then expand | 227.872 | Valid |
| 5 | Expand root again | **219.824** | Valid, new best |
| 6 | UCB best root child, then expand | 225.408 | Valid |

Root expansions at calls 1, 2, and 5 had the same prompt hash, confirming
that this initial arm gives no sibling feedback to the model. UCB controlled
which existing child was revisited; progressive widening permitted the third
root sample at call 5. The best source combines grouped CTA traversal,
`128×256×64` tiles, eight warps, four stages, pointer increments through K,
and unmasked loads/stores for the fixed divisible workload. Correctness was
checked by the worker under the Section 50 contract. This one run validates
the search plumbing; it is not an action-policy comparison with the earlier
sequential or catalog runs, which used other workers and prompts.

An initial attempt failed before VM creation because the local sandbox denied
Nebius DNS access; its separate `glm53-seed0-bgen6.sqlite` trace has zero
generation calls. The retried run completed successfully. The Nebius VM and
boot disk inventories were empty afterward.

## Twenty-five-call depth run

A second seed-0 GLM-5.3 freeform MCTS run kept the packaged root, pinned image,
local package overlay, hidden parent profiles, lightweight NCU collection,
`k_max=3`, and one repair allowance. It raised `B_gen` to 25 and the depth cap
to 10. The completed retry used a 300-second Messages request timeout. Its
[SQLite trace](output/glm53-seed0-bgen25-retry.sqlite) and
[best source](output/glm53-seed0-bgen25-retry-best.py) retain the full evidence.

| Measure | Result |
| --- | ---: |
| Packaged root CUDA-event median | 815.600 µs |
| Best verified kernel CUDA-event median | **201.488 µs** |
| Same-worker correct cuBLAS calibration | 178.464 µs |
| Generation calls / configured `B_gen` | 25 / 25 |
| Valid candidates / compilation failures | 19 / 6 |
| Unique nodes including root | 20 |
| MCTS iterations / UCB descents | 20 / 25 |
| NCU profile calls | 13 |

The best-so-far reached 221.536 µs at call 1, 221.008 at call 12,
219.104 at call 14, then 201.488 at call 24. The call-24 child was two
edges from the root. Relative to its 241.280 µs parent, it jointly widened
`BN` from 128 to 256 and increased `num_stages` from 3 to 4. All six invalid
calls failed compilation. Five were followed by a valid repair; call 25 was
invalid after the budget was exhausted. The tree had three root children,
nine depth-two children, and seven depth-three children. Root calls 1, 2,
and 6 used the same initial prompt hash; call 3 was a repair prompt.

The first 25-call attempt stopped at `B_gen=9` when the Messages transport
failed during call 9. It had reached 230.224 µs by call 8 and is preserved as
`output/glm53-seed0-bgen25.sqlite`; it is not included in the completed-run
figures above. The second attempt started a fresh tree rather than resuming
that trace. Nebius listed no VM or boot disk after either attempt. The
25-call result is a useful depth observation, not a matched comparison with
sequential search or a claim about model sampling variance.
