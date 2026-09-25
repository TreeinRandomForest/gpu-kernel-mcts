# Independent CuTe TMA copy diagnostic v23

This H100 run validated the first executable dynamic phase generated from the
independent typed representation. It used CUTLASS 4.5.1, tile `(128,256,64)`, cluster
`(2,1)`, three shared-memory stages, and SW128 layouts. It is a standalone diagnostic,
not an MCTS state.

## Failure and correction

The initial v20 kernel hung during device synchronization even with one CTA and only
operand A enabled. Staged diagnostics established that JIT and empty launch were
healthy, while the A-load/barrier stage deadlocked. The generated kernel had entered
the TMA `cute.copy` operation from only thread zero. Pinned CuTe examples enter TMA
copy warp-uniformly and use `elect_one()` only for the explicit barrier arrival.

After changing load and store issuance to `warp_idx == 0` and electing the arrival
lane, the A-load stage and every subsequent v23 stage passed.

## Staged results

| Stage | Cluster | Purpose | Result |
| --- | --- | --- | --- |
| `compile_only` | `(2,1)` | CuTe JIT without launch | Passed |
| `launch_empty` | `(1,1)` | Launch geometry without memory movement | Passed |
| `single_cta_a_load` | `(1,1)` | A TMA load and transaction-barrier wait | Passed |
| `single_cta_a` | `(1,1)` | A TMA load/store round trip | Exact |
| `single_cta_ab` | `(1,1)` | Independent A/B round trips | Exact |
| `cluster_ab_no_multicast` | `(2,1)` | Cluster execution with redundant B loads | Exact |
| `cluster_ab_multicast` | `(2,1)` | Cluster execution with B multicast | Exact |

The final multicast report recorded:

```text
configuration_hash = affe6ba57e0d372295897831a517f5b6b8bd8882d664a6b1bbea90781baf37ed
source_hash        = 2edc4003741ccb0c91fccb25d934e43f260cef17161d0c27e2908eb8d2fd3cf3
a_exact            = true
b_exact            = true
a_max_error        = 0.0
b_max_error        = 0.0
```

## Decision

The typed TMA/shared-memory contract has now produced a correct executable dynamic
phase, including cluster multicast. The next bounded step is to add WGMMA consumption
of the validated shared A/B tiles while retaining the same staged failure isolation.
