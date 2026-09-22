# CuTe single-warp-group validation v10

## Purpose

This standalone H100 experiment tested whether the pinned `(128,256)` Hopper GEMM
benefits from replacing its automatic two-warp-group decomposition with one
128-thread warp group. Tile, cluster, pipeline, layouts, workload, and measurement
policy were held fixed. This was a pre-search capability test, not an MCTS run.

## Results

| WGMMA configuration | Correctness | Median latency |
|---|---:|---:|
| `pinned_default` | exact pass | `192.976 us` |
| `single_warp_group` | exact pass | `3015.728 us` |

The configurations produced distinct runtime artifacts, confirming that the typed
field reached JIT execution. The single-warp-group configuration was approximately
15.6 times slower, consistent with the pinned implementation's comment that its
two-group decomposition reduces register spilling for this tile.

## Decision

Keep `single_warp_group` representable as diagnostic evidence, but do not add it to
the default MCTS mutation neighborhood. Passing legality and correctness does not
make this configuration a useful search action for the fixed workload.
