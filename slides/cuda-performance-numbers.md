# CUDA performance numbers for slides

Use this file as the source for CUDA result slides. Keep measurements from different
runs visually separated. GPU timing varies across workers and runs, so comparisons
across those groups are contextual rather than causal.

## Fastest raw MCTS-generated CUDA kernel

| Metric | Value |
|---|---:|
| Kernel | `nebius-sol-best-bgen50-cpuct12-variantdiagnostic_v2-deltas-off.cu` |
| Model | `gpt-5.6-sol` |
| Generation budget | `B_gen = 50` |
| PUCT constant | `c_puct = 12` |
| Profile context | `diagnostic_v2`, incoming profile deltas off |
| Best reward | `3.6497085931` |
| Median latency | `734.4479985 µs` |
| Mean latency | `734.5248027 µs` |
| Minimum latency | `726.912048 µs` |
| Maximum latency | `743.776001 µs` |
| Root median | `28,249.408 µs` |
| Root-relative speedup | `38.46×` |

This is the fastest raw CUDA node across the recorded Nebius MCTS traces. The next
highest recorded CUDA rewards were `3.2347736286` with profile deltas on and
`3.1815947467` in the earlier Sol run.

Source: `nebius-sol-bgen50-cpuct12-variantdiagnostic_v2-deltas-off.sqlite`.

## Standalone autotuning runs

Both runs started from the same exported MCTS kernel but remeasured it on their own
workers. Compare each tuned result with the baseline measured inside the same tuning
run.

| Tuning run | Baseline median | Baseline reward | Best trial median | Best trial reward | Result |
|---|---:|---:|---:|---:|---|
| `B_tune = 4` | `738.8320005 µs` | `3.6510555562` | `905.279999 µs` | `3.4478818307` | No improvement |
| `B_tune = 18` | `739.0879825 µs` | `3.6507704380` | `726.384003 µs` | `3.6681086033` | Improved by 1.72% |

Winning `B_tune = 18` configuration:

```text
K_CHUNK=64
SMEM_PADDING=16
MIN_BLOCKS_PER_SM=2
```

Output kernel: `nebius-sol-tuned-wmma-grid-btune18.cu`.

Sources:

- `nebius-sol-tuning-wmma-grid-btune4.sqlite`
- `nebius-sol-tuning-wmma-grid-btune18.sqlite`

## cuBLAS context

| Reference | Median latency | Tuned CUDA / cuBLAS | Raw CUDA / cuBLAS | Comparability |
|---|---:|---:|---:|---|
| CUDA-era recorded reference | `185.376 µs` | `3.92×` slower | `3.96×` slower | Appropriate headline context, but not measured in the tuning run |
| Later same-VM CuTe reference | `176.096 µs` | `4.13×` slower | `4.17×` slower | Different experiment and environment |

Use the `185.376 µs` reference for the CUDA slide, rounded to `185.4 µs`. Mention
that it came from a separate measurement. Do not present the small difference
between the MCTS median (`734.448 µs`) and tuned median (`726.384 µs`) as the tuning
gain because the tuning run remeasured its input at `739.088 µs`. The supported
within-run tuning improvement is 1.72%.

## Recommended latency ladder

| Implementation | Median latency | Budget label |
|---|---:|---|
| Naive CUDA root | `28,249.408 µs` | Root |
| Fastest raw CUDA MCTS node | `734.448 µs` | `B_gen = 50` |
| Best standalone tuned CUDA variant | `726.384 µs` | `B_gen = 50`, then `B_tune = 18` |
| cuBLAS reference | `185.376 µs` | Vendor reference |

Suggested slide statement: **MCTS improved the CUDA root by 38.46×; an 18-trial
parameter sweep reached 726.384 µs, still 3.92× slower than the separately measured
cuBLAS reference.**

## CuTe DSL optimization path

The independently lowered typed CuTe DSL path begins with a serial kernel rather
than the historical naive CUDA root. The values below form a documented structural
optimization sequence, but not every measurement came from one uninterrupted run.

| Implementation | Median latency | Effect |
|---|---:|---:|
| Independent serial CuTe root | `675.408 µs` | CuTe path root |
| Software-prefetched CuTe | `274.256 µs` | `2.46×` faster than serial |
| Producer/consumer-specialized CuTe | `266.032 µs` | `1.03×` faster than prefetched |
| Best independent typed CuTe | `184.112 µs` | `3.67×` faster than serial |
| Corresponding cuBLAS reference | `176.096 µs` | Vendor reference |

The final typed CuTe kernel was approximately 4.55% slower than the separately
measured `176.096 µs` cuBLAS result. The final two transitions were exercised by a
two-iteration typed MCTS run using `B_mut = 2`, `B_gen = 0`, and `max_depth = 2`.

Source: `docs/experiments/cutedsl-wide-specialized-v69.md`.

## NVIDIA expert-template schedule sweep

This is a separate same-worker experiment over NVIDIA's parameterized CuTe DSL
template. These three values are directly comparable with one another.

| Implementation | Median latency | Relative to cuBLAS |
|---|---:|---:|
| Default expert CuTe template | `193.024 µs` | `1.090×` latency |
| Best typed schedule | `184.880 µs` | `1.044×` latency |
| Same-run cuBLAS | `177.056 µs` | `1.000×` latency |

The schedule sweep improved the default template by 4.4% and finished 4.4% slower
than same-run cuBLAS.

Source: `docs/experiments/cutedsl-schedule-tuning-v1.md`.

## Cross-run contextual latency ladder

This combined ladder is useful for a high-level slide, but it is not a controlled
same-run comparison. Label it as contextual because the CUDA and CuTe measurements
were collected on different workers or in different experiments.

| Implementation | Median latency | Provenance |
|---|---:|---|
| Naive CUDA root | `28,249.408 µs` | CUDA MCTS run |
| Fastest raw CUDA MCTS node | `734.448 µs` | CUDA MCTS run, `B_gen = 50` |
| Independent serial CuTe root | `675.408 µs` | Independent CuTe path |
| Software-prefetched CuTe | `274.256 µs` | Independent CuTe path |
| Producer/consumer-specialized CuTe | `266.032 µs` | Independent CuTe path |
| Best independent typed CuTe | `184.112 µs` | Typed CuTe MCTS path |
| CuTe-era cuBLAS reference | `176.096 µs` | Separate same-VM comparison |

Do not compare the CuTe-era `176.096 µs` cuBLAS value directly with the CUDA
results as though it were collected in their run. The CUDA-era contextual reference
is `185.376 µs`.
