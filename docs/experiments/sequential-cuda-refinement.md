# Sequential CUDA refinement experiment

This is a standalone linear baseline for CUDA C++. It uses the existing worker's
tier-0 compile, correctness, and timing contract. It does not invoke MCTS
selection, progressive widening, priors, backup, or re-rooting.

## One iteration

1. Start from the current verified kernel. The root is evaluated on the worker.
2. Profile the current kernel with the selected NCU metric set. The initial root
   and each newly accepted kernel are profiled once. The NCU prompt view may
   hide or filter metrics without changing collection.
3. Read the current node's recorded PTX dataflow graph and bounded compiler
   summary, if selected. The prompt includes the measured benchmark, source,
   selected profile/compiler/dataflow views, and one recent verification result.
4. Ask the model for JSON containing a short optimization hypothesis and the
   complete replacement CUDA source. A fixed strategy menu is optional; the
   default prompt has no strategy field or menu.
5. Compile, check correctness, and measure performance through the existing
   worker evaluator. Each generation or repair call consumes one `B_gen`.
   Infrastructure retries repeat worker evaluation without another model call.
6. An invalid candidate leaves the current kernel unchanged. Every valid
   candidate becomes current, including slower ones. The best verified node is
   tracked separately. A duplicate verified state reuses its recorded node.

The OpenAI client keeps a bounded local text conversation: the most recent
`--session-history-turns` user/assistant pairs are replayed with each request,
while `store=False` remains in effect. The trace records the exact request
message array and instructions for each generation. This replays visible text;
it does not carry hidden model reasoning state across calls. Current source,
measurement, context, and recent verification feedback are sent afresh on each
iteration, even if older turns fall out of the window. Fix the history bound
before comparing runs.

## CLI factors

Use `python -m kernel_mcts.search_cli --algorithm sequential` with the normal
worker, image, model, and trace flags. This mode requires
`--generator openai --backend cuda_cpp` and a positive `--generation-budget`.
The existing `--confirm-create-and-terminate` gate applies when launching a
billable worker.

| Factor | Flag | Values |
| --- | --- | --- |
| Generation-call depth | `--generation-budget` | positive integer, including repairs |
| Session memory | `--session-history-turns` | 0 or more prior text turns; default 2 |
| Compiler summary | `--sequential-compiler-detail` | `none`, `resources`, `full` |
| PTX dataflow | `--sequential-dataflow-view` | `none`, `counts`, `paths`, `subgraph`, `full` |
| Dataflow explanation | `--sequential-dataflow-description` | `none`, `limits` |
| Collected NCU metrics | `--profile-metric-set` | `lightweight_v1`, `diagnostic_v2` |
| Visible NCU metrics | `--sequential-profile-visibility` | `none`, `lightweight`, `full` |
| Fixed strategy menu | `--sequential-strategy-menu --strategies PATH` | off by default |

The `full` compiler choice is the existing bounded ptxas resource and SASS
static-site summary. The PTX graph views remain static evidence: they do not
claim dynamic memory traffic or paths across control-flow joins.

SQLite `search_runs.algorithm` is `sequential`. Standard `generations`, `nodes`,
`node_dataflow_graphs`, and profiling records remain available to the trace
browser. `sequential_step` events capture lineage, acceptance, current and best
node IDs, and feedback. `worker_usage` records worker occupancy wall time as a
GPU cost proxy; it is not a provider billing record. No GPU experiment has been
run as part of this implementation.
