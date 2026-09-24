# CuTe epilogue-stage validation v13

## Purpose

This standalone H100 experiment tested epilogue pipeline depths 2, 3, and the pinned
depth 4 while holding CTA tile `(128,256)`, cluster `(2,1)`, and the heuristic A/B
mainloop depth fixed. It was the promotion gate for a typed CuTe structural field and
consumed no MCTS or tuning budget.

## Results

| Epilogue stages | Median latency | Normalized IR SHA-256 | Fatbin SHA-256 |
|---:|---:|---|---|
| 2 | `190.000 us` | `e625141f...` | `3c696361...` |
| 3 | `189.728 us` | `ac08ca92...` | `64e57298...` |
| pinned 4 | `188.448 us` | `671224e5...` | `4be702f4...` |

All configurations passed the repository correctness contract with zero observed
maximum and mean error. Their means were `190.923`, `190.931`, and `189.587 us`;
population standard deviations were `2.592`, `3.737`, and `2.674 us` respectively.
Every depth produced distinct normalized IR and an embedded fatbin with distinct
identity, proving that the control reaches effective code generation.

## Interpretation

Pinned depth 4 was fastest in this run, but all median differences were below 1% and
small relative to the observed distributions. The experiment therefore validates a
real, non-pathological structural dimension; it does not establish that changing
depth improves the fixed kernel. Locally slower valid kernels remain eligible because
interactions with future epilogue layouts or ownership policies may differ.

## Promotion decision

Promote epilogue depth to `CuteGemmProgram` schema v2 with strict initial legality:
explicit depths 2 and 3 apply only to tile `(128,256)`, cluster `(2,1)`. The pinned
depth 4 is represented by `None`. Explicit 4 is excluded because the v13 pinned run
already represents that effective kernel, and duplicate canonical identities would
violate transposition semantics.
