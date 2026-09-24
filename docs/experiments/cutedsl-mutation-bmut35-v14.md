# CuTe mutation search B_mut=35 v14

Run `fa5f89da-9801-4f37-8ef9-00ddafc830f8` searched all four currently validated
typed mutation strategies around NVIDIA's pinned Hopper dense-GEMM CuTe template:
CTA tile, cluster shape, mainloop pipeline depth, and epilogue pipeline depth.

The run used `B_mut=35`, `B_gen=0`, `k_max=4`, and `max_depth=10`. It completed 42
iterations, created 24 unique nodes, reused 12 transposed nodes, performed 11 profile
calls, and encountered no invalid proposal. Seven iterations terminated a selected
DAG traversal cycle without consuming another mutation, explaining why iteration
count exceeded `B_mut`.

## Result

| State | Median (us) | Reward |
| --- | ---: | ---: |
| Root: `(128,256)`, cluster `(1,1)`, pinned stages | 193.136 | 0.000000 |
| Best: `(128,256)`, cluster `(2,1)`, pinned stages | 186.816 | 0.033270 |
| Same schedule, epilogue stage 3 | 186.928 | 0.032671 |
| Same schedule, epilogue stage 2 | 188.432 | 0.024657 |
| Same schedule, mainloop stage 3 | 189.104 | 0.021097 |

The best node was found at iteration 4 by `change_cluster_shape`. Its 3.38% speedup
is relative to this run's root. Absolute latency is the more useful cross-run measure:
the result is consistent with, but does not improve upon, the earlier standalone grid
optimum near 184.9 us. The difference is plausibly run-to-run measurement and GPU
operating-state variation.

## Search behavior

All 35 mutation proposals were valid. They comprised 11 cluster, 9 CTA-tile, 11
mainloop-pipeline, and 4 epilogue attempts. Twelve proposals transposed to existing
canonical nodes, leaving the root plus 23 newly measured unique states. Cluster-shape
changes received the most traversal traffic and contained the global best. CTA-tile
changes were substantially worse in this run.

The experiment therefore shows that MCTS correctly navigates and caches the typed
space, but it does not show an advantage over direct grid tuning on this small expert-
template parameter space. Mainloop and epilogue staging did not produce a meaningful
improvement over NVIDIA's pinned heuristics. The next useful expansion should test a
genuinely structural transformation before promotion to the canonical search schema.
