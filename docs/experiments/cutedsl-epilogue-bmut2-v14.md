# CuTe epilogue mutation smoke v14

Run `d884b5c3-bc5c-4648-a3a6-d1c68d16b5a9` tested the promoted epilogue control
through the normal MCTS and SQLite trace path. It began from tile `(128,256)`, cluster
`(2,1)`, with pinned epilogue depth 4 and restricted the semantic action set to
`change_epilogue_stages`.

The run used `B_mut=2`, `B_gen=0`, `k_max=2`, and `max_depth=1`. It completed two
iterations and produced exactly three nodes:

| State | Median (us) | Reward | Result |
| --- | ---: | ---: | --- |
| Root, pinned stage 4 | 185.216 | 0.000000 | Valid baseline |
| Epilogue stage 2 | 187.232 | -0.010826 | Valid, 1.09% slower |
| Epilogue stage 3 | 186.560 | -0.007230 | Valid, 0.73% slower |

Both proposals were logged as `typed_mutation`, compiled successfully, passed
correctness, created distinct canonical nodes, and consumed one `B_mut` each. No LLM
generation or repair occurred. The sole strategy edge recorded two visits, two
proposals, two mutation attempts, two valid proposals, zero invalid proposals,
`Q_mean=-0.009028`, and `Q_max=-0.007230`.

Each valid leaf reward was backed up. Since both candidates were slower than the
fixed root measurement, the root correctly remained the global best with reward
zero. This is expected behavior and preserves the rule that valid locally slower
kernels remain searchable nodes rather than being pruned.

The trace recorded three node snapshots, two realization-edge snapshots, two backup
events, and complete schema-v2 representations. `profiles=1` means only the root was
profiled in this depth-one smoke; it is not evidence of missing candidate evaluation.
