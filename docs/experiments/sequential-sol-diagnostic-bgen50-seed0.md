# Sequential CUDA refinement: diagnostic NCU, Sol seed 0

This is a matched pilot against the [lightweight NCU seed-0 run](sequential-sol-bgen50-seed0.md). It used `gpt-5.6-sol` at medium reasoning, `B_gen=50`, two prior conversation turns, full bounded compiler summary, PTX `paths`, one repair, the same 4096³ BF16 GEMM root and tier-0 timing contract, and the same pinned H100 worker image (`sha256:f90da644bcabc3a07756ecea933f4e1bc2d7faa5785e1c5d39c851eecedb0429`). The changed factor was NCU collection and full prompt visibility: `diagnostic_v2` instead of `lightweight_v1`. Neither run supplied a fixed strategy menu.

| Measure | Diagnostic pilot | Lightweight pilot |
| --- | ---: | ---: |
| Best saved-node median | **797.360 µs** | **655.856 µs** |
| Call that created best node | 44 | 44 |
| Valid / invalid / infrastructure-failed calls | 40 / 8 / 2 | 47 / 3 / 0 |
| Distinct nodes and profiles, including root | 33 / 33 | 41 / 41 |
| Input / output tokens | 1,728,370 / 209,160 | 1,368,533 / 197,229 |
| Worker occupancy | 2,618.9 s | 2,459.7 s |

The diagnostic best was 141.504 µs (21.6%) slower than the lightweight best. A repeated evaluation at call 46 measured the **same** diagnostic best source at 790.992 µs; the node retains its original 797.360 µs median. The best node's original 30 samples had mean 795.211 µs, standard deviation 5.426 µs, and range 783.808–800.352 µs. The result does not isolate metric usefulness from stochastic generation across one seed. It provides no evidence that the larger metric bundle improved this particular sequential search.

Run ID: `fae2eb94-ac96-4f87-b6c4-75e9fe912893`. Local artifacts: [`sequential-sol-diagnostic-bgen50-seed0.sqlite`](../../sequential-sol-diagnostic-bgen50-seed0.sqlite) and [best CUDA source](../../sequential-sol-diagnostic-bgen50-seed0-best.cu). The trace passed SQLite integrity checking; all 50 recorded prompt hashes matched their prompt text, and all 33 stored profiles identified `diagnostic_v2`. The runner completed and Nebius listed no instance or disk afterward. Worker occupancy is a wall-time proxy, not a provider bill.
