# CuTe DSL Search Model

## Relationship to cuBLAS

cuBLAS is a fixed external performance baseline for this project. NVIDIA supplies
many optimized implementations and internal dispatch logic that select a kernel for
the workload, dtype, and hardware. The implementation and selection space are mostly
opaque to this repository. cuBLAS is measured under the same workload contract, but
it is not an MCTS node and MCTS does not modify it.

CuTe DSL search instead constructs explicit, auditable kernels. Each MCTS node is one
complete effective CuTe configuration under a fixed workload, GPU, and toolchain. Its
state may eventually include:

- CTA and cluster decomposition;
- WGMMA and warp-group arrangement;
- pipeline stages and producer/consumer specialization;
- TMA copy layout;
- shared-memory layout and swizzle; and
- epilogue ownership and staging.

Only fields that the typed schema, deterministic renderer, and static validator
actually support belong to the active search space. Initially, only CTA tile and
cluster shape have deterministic mutation support.

## One expansion

Conceptually, one new realization is produced as follows:

```text
PUCT selects a semantic StrategyEdge
        |
        v
progressive widening permits another realization
        |
        v
proposal mechanism produces a typed candidate
   |                                  |
   +-- deterministic mutation         +-- stochastic LLM proposal
        |                                  |
        v                                  v
     B_mut                              B_gen
        |                                  |
        +---------------+------------------+
                        v
              static validation
                        v
          deterministic source rendering
                        v
          JIT -> correctness -> benchmark
                        v
                valid child node
```

PUCT continues to select semantic strategies. Progressive widening limits the number
of distinct concrete outcomes beneath a strategy. When an existing realization is
revisited, UCB selects among those existing children. Only valid measured children
become nodes and participate in backup.

### Choosing the proposal mechanism

The core search can configure one proposal mechanism directly or wrap a mutation and
LLM generator in the mixed-mechanism router. Direct CuTe mutation generation uses
`B_mut`, while direct LLM generation uses `B_gen`.

The initial mixed-mechanism router uses a simple mutation-first rule after PUCT has
selected the semantic strategy and progressive widening has admitted a new child:

1. use an untried supported typed mutation when one exists and `B_mut` remains;
2. otherwise use LLM generation when it is supported and `B_gen` remains; and
3. declare that parent-strategy proposal space exhausted when neither mechanism can
   produce another realization.

This is deliberately not another PUCT decision. PUCT chooses the semantic strategy;
the router chooses how to realize it. The budgets remain separate, and exhausted
mechanisms become ineligible without changing PUCT, progressive-widening, or UCB
formulas. A later explicit ablation may compare this rule with a bandit or learned
mechanism policy using observed validity, reward, and cost.

Each proposal trace records the eligible mechanism sequence and the selected
mechanism's budget kind. The CLI can construct mutation-only and mixed CuTe searches
and providers initialize a CuTe worker backend. Guarded remote H100 validation of
these paths remains pending.

## Deterministic mutations

A deterministic mutation applies a known typed transformation with explicit values:

```text
change_cta_tile: (128, 256) -> (128, 128)
```

Given the parent, strategy, and parameters, it always produces the same canonical
candidate. It does not call an LLM. The `B_mut` budget counts these mutation proposals
separately from model calls. Different mutation paths that reach
the same canonical configuration transpose to the same cached state.

The current standalone mutation mechanisms are:

- `change_cta_tile`; and
- `change_cluster_shape`; and
- `change_pipeline_stages`; and
- `change_epilogue_stages`.

The pipeline mutation changes only A/B mainloop staging and offers explicit depths
`2` and `3`. The pinned heuristic remains the root behavior, and epilogue staging
remains pinned. Explicit depth `4` is deliberately excluded from this initial
neighborhood because it is equivalent to the heuristic for the reference
`(128,256)` tile and would create duplicate effective states.

The epilogue mutation changes only output-pipeline depth and currently offers values
2 and 3 only for tile `(128,256)`, cluster `(2,1)`, the schedule validated on H100.
Pinned depth 4 is represented by `None`; explicit 4 is excluded as an identical-state
alias. Mutation generation filters statically invalid cross-schedule combinations.
The canonical backend CLI accepts a complete tile/cluster tuple for guarded hardware
validation, and a GPU-independent MCTS integration test verifies that an epilogue
realization creates a schema-v2 node under `B_mut` without consuming `B_gen`.
A bounded aggregate validation mode runs stages 2 and 3 through that canonical path
and checks identity, correctness, cache reuse, and distinct runtime fingerprints
before the field is exercised in a larger search.

For guarded experiments, the search CLI accepts all four `--cute-root-*` schedule
arguments together. They are legal only for `--backend=cute_dsl`, are validated and
rendered through the normal typed path, and the complete root representation is
recorded in run provenance. This permits an epilogue-focused run to begin at the
validated `(128,256)`, cluster `(2,1)` state without changing the default CuTe root.
`--cute-strategy` may be repeated to select an explicit subset of the registered
CuTe semantic actions for a guarded ablation. It changes only the configured action
set; selection and budget semantics within that set are unchanged.

The first epilogue-only mutation smoke uses two deterministic proposals so both
validated realizations can be represented beneath one strategy edge:

```bash
python -m kernel_mcts.search_cli \
  --provider nebius \
  --image docker.io/saarora/gpu-kernel-mcts:cutedsl-epilogue-v14 \
  --trace cutedsl-epilogue-bmut2-v14.sqlite \
  --backend cute_dsl \
  --generator cute-mutation \
  --generation-budget 0 \
  --mutation-budget 2 \
  --cute-root-tile-m 128 \
  --cute-root-tile-n 256 \
  --cute-root-cluster-m 2 \
  --cute-root-cluster-n 1 \
  --cute-strategy change_epilogue_stages \
  --k-max 2 \
  --max-depth 1 \
  --best-output cutedsl-epilogue-bmut2-v14-best.py \
  --nebius-project-id PROJECT_ID \
  --nebius-subnet-id SUBNET_ID \
  --nebius-ssh-public-key ~/.ssh/nebius.pub \
  --nebius-ssh-private-key ~/.ssh/nebius \
  --confirm-create-and-terminate
```

Run `d884b5c3-bc5c-4648-a3a6-d1c68d16b5a9` completed in two iterations with
`B_mut=2`, `B_gen=0`, three nodes, and two valid measured-leaf backups. Its detailed
trace analysis is recorded in
[CuTe epilogue mutation smoke v14](experiments/cutedsl-epilogue-bmut2-v14.md).

They are connected to core MCTS through a deterministic mutation generator and
covered by a GPU-independent end-to-end search test. Remote-worker orchestration and
the guarded H100 smoke-search command are not implemented yet.

## Stochastic LLM proposals

An LLM realization consumes `B_gen`, including any repair calls. It may return a
constrained typed representation or structural patch. Fresh calls can produce
different concrete realizations beneath the same semantic strategy.

An LLM does not automatically expand the representable design space. Its output must
still be expressible by the current typed schema, renderer, and validator. For
example, an LLM cannot introduce a new pipeline organization while pipeline structure
is fixed by the renderer.

The roles are therefore:

- deterministic mutations systematically explore known supported choices;
- LLM calls stochastically propose or combine choices within the supported language;
- repository/compiler development adds new schema fields, rendering behavior, and
  proven legality rules, thereby expanding the actual search space.

## Budgets and autotuning

The intended accounting is:

```text
B_gen  = LLM generation and repair calls
B_mut  = deterministic typed mutation proposals
B_tune = standalone mechanical tuning trials
```

Autotuning is not required at every MCTS node. Initially, MCTS should evaluate each
new complete configuration once and reuse its cached result. Standalone or post-search
autotuning remains separately budgeted. Any future leaf-local tuner should be an
explicit ablation rather than hidden work performed at every expansion.
