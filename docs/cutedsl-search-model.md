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
- `change_cluster_shape`.

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
