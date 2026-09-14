# Nebius profiled MCTS search, B_gen=10

## Purpose

This experiment tested whether lightweight NCU context and the new structural
tensor-core strategies could move the search beyond the approximately 6.3x root
speedup observed in the preceding B_gen=5 run. This is an experiment report, not a
change to the Milestone A specification.

## Configuration

- Date: 2026-09-14
- Run ID: `0ae828ed-212e-40c4-b2eb-92d90b7157bf`
- Trace: `nebius-llm-bgen10.sqlite` (local artifact, intentionally not committed)
- Model: `gpt-5.6-terra`, reasoning effort `medium`
- Generation budget: 10 calls, including repairs
- MCTS: `c_puct=1.5`, `c_pw=1.0`, `alpha_pw=0.5`, `c_ucb=1.0`,
  `K_max=4`, `max_depth=10`, seed 0
- Repairs: at most one per proposal
- Priors: uniform; `B_prior=0`
- Worker image: `docker.io/saarora/gpu-kernel-mcts:h100-worker-v9`
- Worker: Nebius NVIDIA H100 80GB HBM3 SXM, compute capability 9.0
- CUDA toolkit: 12.8.2; NCU: 2025.1.1

The workload was fixed 4096 x 4096 x 4096 BF16 GEMM with FP32 accumulation and BF16
output. A was row-major, B was column-major, and C was row-major. The fixed launch was
a 16 x 16 thread block for every logical 16 x 16 output tile.

A representative command is:

```bash
.venv/bin/python -m kernel_mcts.search_cli \
  --provider nebius \
  --image docker.io/saarora/gpu-kernel-mcts:h100-worker-v9 \
  --trace nebius-llm-bgen10.sqlite \
  --nebius-project-id NEBIUS_PROJECT_ID \
  --nebius-subnet-id NEBIUS_SUBNET_ID \
  --nebius-username NEBIUS_VM_USERNAME \
  --nebius-ssh-private-key ~/.ssh/nebius \
  --nebius-ssh-public-key ~/.ssh/nebius.pub \
  --generator openai \
  --model gpt-5.6-terra \
  --strategies configs/strategies.yaml \
  --generation-budget 10 \
  --best-output nebius-best-bgen10.cu \
  --timeout 1200 \
  --confirm-create-and-terminate
```

## Outcome

The search completed eight valid MCTS iterations from ten generation calls, created
eight valid non-root nodes, collected four lazy profiles, and retained the root plus
eight candidates in the search DAG. No transpositions occurred.

- Root median: 28,245.2 microseconds
- Best median: 3,136.8 microseconds
- Best reward: 2.197725
- Root-relative speedup: approximately 9.00x
- LLM usage: 24,000 input tokens and 29,329 output tokens
- Aggregate LLM request latency: 402.49 seconds

The separately measured cuBLAS baseline was approximately 185 microseconds. The best
generated kernel was therefore still about 17x slower. Equivalently, the generated
best reward was about 2.20 versus a cuBLAS-equivalent root-normalized reward near
5.03.

## Best path

```text
root
28,245 us, reward 0
  |
  +-- pipeline_tensor_core_data_movement, B_gen 7
      3,200 us, reward 2.1778
        |
        +-- reduce_register_pressure, B_gen 9
            3,137 us, reward 2.1977
```

The B_gen 10 vectorized-memory child of the best node ran in 3,379 microseconds and
did not replace the global best.

## Generation and repair history

| B_gen | Strategy | Attempt | Result | Median us | Reward |
|---:|---|---:|---|---:|---:|
| 1 | pipeline tensor-core data movement | 0 | correctness failure | - | - |
| 2 | pipeline tensor-core data movement | 1 | valid | 11,379 | 0.9092 |
| 3 | pipeline tensor-core data movement | 0 | valid | 9,230 | 1.1185 |
| 4 | coalesced global memory | 0 | valid | 7,600 | 1.3127 |
| 5 | coalesced global memory | 0 | valid | 8,152 | 1.2427 |
| 6 | pipeline tensor-core data movement | 0 | compile failure | - | - |
| 7 | pipeline tensor-core data movement | 1 | valid | 3,200 | 2.1778 |
| 8 | reduce register pressure | 0 | valid | 3,168 | 2.1879 |
| 9 | reduce register pressure | 0 | valid | 3,137 | 2.1977 |
| 10 | vectorized memory access | 0 | valid | 3,379 | 2.1233 |

The first failed candidate produced incorrect fixed-shape output and was repaired
successfully. The second failed candidate used an invalid WMMA BF16 type and attempted
an incompatible accumulator store. Its repair produced the largest improvement in the
run. Both failures consumed `B_gen` but created no nodes and received no reward backup.

## NCU evidence

Profiles were collected lazily when a valid node was selected for expansion. The
final best was selected again and therefore has a profile in the trace.

| Node | Median us | Registers/thread | Occupancy | SM throughput | Tensor pipe | DRAM | L2 | L1 | Instructions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Root | 28,245 | 30 | 99.42% | 64.93% | 0.00% | 1.37% | 11.35% | 94.87% | 13.6B |
| Early WMMA | 9,230 | 38 | 74.26% | 39.89% | 2.51% | 7.45% | 28.50% | 58.64% | 3.18B |
| B_gen 7 | 3,200 | 42 | 61.28% | 22.68% | 6.68% | 7.51% | 59.64% | 98.47% | 489M |
| Best | 3,137 | 40 | 73.27% | 22.42% | 6.77% | 7.59% | 60.58% | 98.56% | 417M |

High occupancy alone was not predictive: the scalar root had the highest occupancy
but no tensor-pipe use. The large improvement correlated with converting scalar work
to WMMA, assigning independent output tiles to warps, and reducing executed
instructions. The final register-pressure transformation reduced registers from 42
to 40, raised occupancy, and reduced instruction count, but improved runtime by only
about 2% over its parent.

Long-scoreboard stall values increased along the winning branch. Together with low
tensor-pipe utilization and high L1 activity, this suggests remaining dependency or
operand-delivery latency, but this is a hypothesis rather than a complete diagnosis
from the lightweight metrics alone.

## Winning kernel structure

The B_gen 7 candidate implemented the most important part of the structural strategy:

- Only one owner block in each group of eight logical tiles performed work.
- Each of the block's eight warps owned a distinct adjacent 16 x 16 output tile.
- The warps reused a shared A tile and accumulated their own complete K dimension.
- WMMA performed BF16 matrix operations with FP32 accumulation.
- The earlier cross-warp reduction of partial K accumulations was eliminated.

It did not implement actual asynchronous copies, double buffering, or overlap between
operand movement and MMA. It synchronized around every K tile, staged only A in shared
memory, and loaded B directly for each warp. The final register-pressure candidate
mainly flattened shared arrays and simplified indexing; it retained the same overall
algorithm.

## Search behavior

At the root, `pipeline_tensor_core_data_movement` received all eight visits, generated
three valid realizations from five calls, and ended with `Q_mean=1.6587` and
`Q_max=2.1977`. None of the other six root strategies was visited. The explicit
`tensor_core_output_tiling` strategy was therefore never selected, even though the
winning pipeline realization independently implemented that idea.

This behavior follows the specified PUCT formula, but it exposes a reward-scale issue
for this experiment. With seven uniform priors, `c_puct=1.5`, and eight root visits,
an unvisited action's exploration contribution is only about 0.61. The selected
action's positive mean reward was already much larger. Since log-speedup rewards are
not bounded to the range commonly used with PUCT, early positive results can dominate
unvisited strategies for a modest budget.

This is not recorded as an implementation bug: changing the formula or reward would
change specified search semantics. It is an experimental-design concern that should
be addressed explicitly before spending a B_gen=50 budget.

## Recommended next experiment

Do not simply repeat the same configuration at B_gen=50. First choose and document
one of these controlled experiments:

1. Use a focused strategy configuration to evaluate `tensor_core_output_tiling` and
   true operand pipelining directly.
2. Sweep `c_puct` without changing the PUCT formula, recording strategy coverage and
   best reward for each value.
3. Define a spec-level reward normalization or forced-initial-exploration experiment,
   then update the specification before changing search semantics.

For kernel performance, the next generated design should assign independent output
tiles to warps while adding real double buffering or asynchronous operand staging.
Longer-term H100 work can investigate WGMMA/TMA or a CUTLASS/CuTe backend rather than
expecting free-form WMMA rewrites alone to approach cuBLAS.

## Trace contents

The SQLite trace retains the complete assembled prompt, fixed API instructions, raw
model output, candidate source, token and latency usage, compiler/correctness evidence,
benchmarks, profiles, node and edge snapshots, and environment manifest. Generated
kernels and the SQLite database remain local artifacts and are intentionally excluded
from version control.
