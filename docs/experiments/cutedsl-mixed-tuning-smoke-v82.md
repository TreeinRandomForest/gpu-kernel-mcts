# Independent CuTe mixed-generation and tuning smoke v82

This bounded H100 SXM run validated the first combined independent CuTe proposal
and post-search tuning path. It used image
`docker.io/saarora/gpu-kernel-mcts:cutedsl-ablation-v82`, the serial independent
root, only `change_mainloop_schedule`, and explicit budgets `B_mut=1`, `B_gen=1`,
and `B_tune=2`.

The deterministic proposal changed the serial mainloop to the admitted software-
prefetch state. It was valid and produced reward `0.8921356954`. After `B_mut` was
exhausted, one fresh LLM call returned a valid typed representation for the same
strategy. Canonicalization mapped it to the existing serial root, so the trace
recorded a transposition (`node_reused`) rather than GPU re-evaluation or a third
node. Search ended after two iterations with two nodes, two profiles, `B_mut=1`,
and `B_gen=1`.

The final-result tuner started only after MCTS completed. It evaluated two grid
configurations while preserving the prefetched algorithm:

- pipeline stages 2, SW64: reward `0.6297175790`;
- pipeline stages 2, SW128: reward `0.6729894053`.

Both trials were valid but slower than the untuned search winner. They were recorded
as two `tuning_trial` rows and did not change MCTS iterations, nodes, profiles,
visits, Q values, or backups.

The run exposed a result-selection bug: the tuner reported the best trial as its
output even when that trial did not beat the untuned baseline. The implementation
now initializes the post-search best result to the untuned winner and replaces it
only after a strictly better valid trial. Regression coverage proves that a
non-improving tuning sweep retains the original program and reports
`improved=False`. The v82 trace predates that correction, so a later bounded run
must confirm the corrected persisted `best_reward` and tuned output on H100.

Local artifacts such as the SQLite trace and rendered `.py` outputs remain
untracked.
