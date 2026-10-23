# Research log

Updated 2026-10-23. This is a living account of why the project changed direction and what the recorded experiments found. It summarizes completed work; [the specification](spec.md) remains the source of truth for search behavior, [the CUDA-first direction](docs/cuda-first-next-direction.md) states the current research priority, and [TODO.md](TODO.md) tracks unfinished work. Links below point to reports with configuration, trace, and measurement detail. Latencies are saved-kernel medians in microseconds unless stated otherwise; lower is better. The crossover table explicitly uses the mean of four run medians.

## The question that shaped the project

We began with a fixed 4096³ BF16 GEMM on H100 and a naive CUDA root around 28 ms. The Milestone A question was whether **one global MCTS search over semantic optimization strategies**, with a fixed LLM-generation budget, could find valid faster kernels. Compilation, correctness, and timing gate node creation. Every generation and repair spends `B_gen`; an MCTS visit does not imply another GPU evaluation. The first CUDA experiments established a substantial root-to-best gain, but also a large gap to cuBLAS. That gap motivated both Hopper-specific CuTe work and richer context for free-form CUDA generation. [Spec](spec.md) · [CUDA-first direction](docs/cuda-first-next-direction.md)

## September: CUDA MCTS learned useful structure

| Experiment | Model and budget | Best µs | What changed in our thinking |
| --- | --- | ---: | --- |
| Early profiled search | Terra, `B_gen=10`, `c_puct=1.5` | 3,136.8 | Low exploration concentrated visits on one root strategy. [Report](docs/experiments/nebius-bgen10-profiled-search.md) |
| Broader MCTS | Terra, `B_gen=50`, `c_puct=12` | 2,669.9 | All seven strategies received visits. Tiling, synchronization reduction, and specialization formed the winning path, but one MMA per staged operand left tensor throughput low. [Report](docs/experiments/nebius-bgen50-cpuct12-profiled-search.md) |
| Add multi-accumulator strategy | Terra, `B_gen=50`, eight strategies | 1,375.6 | The new strategy nearly halved the previous best in one run. It also lowered call-level validity to 28/50, showing that valuable architecture changes can be hard to generate correctly. [Report](docs/experiments/nebius-bgen50-cpuct12-multiacc.md) |
| Same MCTS design, new model | Sol vs Terra, `B_gen=50` | 1,172.8 vs 1,375.6 | Sol found a faster kernel and 36/50 valid calls versus Terra's 28/50 in this paired model pilot. One run per model was insufficient for a general ranking. [Report](docs/experiments/nebius-sol-vs-terra-bgen50.md) |
| Fastest recorded raw CUDA MCTS | Sol, `B_gen=50`, diagnostic NCU, profile deltas off | **734.448** | A WMMA/async-copy branch reached 38.46× the naive root's speed. A separate `B_tune=18` sweep remeasured its input at 739.088 µs and improved it to 726.384 µs, a 1.72% within-sweep gain. The separately measured CUDA-era cuBLAS reference was 185.376 µs. [Numbers and trace identity](slides/cuda-performance-numbers.md) |

The early searches taught us that larger output tiles, multiple independent accumulators, and asynchronous staging can work together. They did **not** show that a strategy label uniquely determines the edit: generated candidates often changed tiling, memory movement, ownership, and synchronization together. The remaining roughly 4× gap between the fastest CUDA result and the separate cuBLAS reference suggested that prompt refinement alone might not supply all the Hopper mechanisms we needed. Do not read the rows above as a controlled algorithm ladder: models, prompts, strategies, and runs changed.

## September: test a structured Hopper path

We first adapted NVIDIA's pinned Hopper CuTe dense-GEMM example to the same workload and timing contract. The fixed CuTe kernel measured 192.544 µs versus 178.096 µs cuBLAS on one worker. Tuning CTA tile and cluster shape reached 184.880 µs versus 177.056 µs cuBLAS on another same-worker comparison. A typed MCTS mutation run later recovered the same schedule neighborhood (186.816 µs in that run), but did not beat direct grid tuning of this small expert-template space. [Same-worker baseline](docs/experiments/cutedsl-bf16-same-worker-baselines.md) · [Schedule tuning](docs/experiments/cutedsl-schedule-tuning-v1.md) · [Mutation search](docs/experiments/cutedsl-mutation-bmut35-v14.md)

We then built an **independently lowered** typed CuTe GEMM rather than treating template parameters as new kernel synthesis. Its documented structural path was serial 675.408 µs → software prefetch 274.256 µs → producer/consumer specialization 266.032 µs → wider two-consumer-group CTA **184.112 µs**. The last state was 4.55% slower than a separately measured 176.096 µs cuBLAS result. A two-step mutation-only MCTS run reproduced the final transitions with `B_mut=2`, `B_gen=0`. These values combine validated experiments and are not one uninterrupted run. [Wide specialized CTA](docs/experiments/cutedsl-wide-specialized-v69.md) · [CUDA-first synthesis](docs/cuda-first-next-direction.md)

Negative diagnostics were equally informative:

| Attempt | Evidence and decision |
| --- | --- |
| `single_warp_group` | Correct but regressed from 192.976 to 3,015.728 µs; excluded from MCTS. [TODO evidence](TODO.md) |
| Extra WGMMA in-flight groups | Different source parameter but identical normalized IR/fatbin at fixed pipeline setting; removed as a canonical search dimension. [TODO evidence](TODO.md) |
| Non-multicast TMA with the clustered template | Illegal memory access; a one-field override was not a valid transformation. [TODO evidence](TODO.md) |
| SW64 on the wide specialized kernel | Correct, but 224.192 versus 185.424 µs for SW128; not admitted to that neighborhood. [Interaction report](docs/experiments/cutedsl-wide-specialized-cluster-v70.md) |
| Clustered wide specialization | Device synchronization stalled; not a measured slow kernel and not a valid MCTS state. [Interaction report](docs/experiments/cutedsl-wide-specialized-cluster-v70.md) |
| Persistent scheduler | Ownership and full GEMM correctness were validated in stages, but the final 239.312 µs kernel was 30.0% slower than the 184.112 µs static specialized result. [Persistent report](docs/experiments/cutedsl-persistent-ownership-v72.md) |

The typed route brought canonical identity, legality checks, deterministic lowering, and transposition reuse, but required extensive BF16 GEMM and Hopper-specific schema and debugging work. A mixed typed-mutation/LLM/tuning smoke proved separate `B_mut`, `B_gen`, and `B_tune` accounting and a canonical LLM transposition; it did not establish an LLM advantage in the typed space. That is why the current direction preserves CuTe as a case study while prioritizing open-ended CUDA generation with better compiler and hardware evidence. [Mixed smoke](docs/experiments/cutedsl-mixed-tuning-smoke-v82.md) · [CUDA-first direction](docs/cuda-first-next-direction.md)

The supporting CuTe record is indexed here so intermediate validations and rejected controls remain discoverable:

- **Backend and early typed search:** [backend phase 2](docs/experiments/cutedsl-backend-phase2.md), [mutation smoke](docs/experiments/cutedsl-mutation-smoke-v1.md), [four-mutation run](docs/experiments/cutedsl-mutation-bmut4-v1.md), [CTA tile search v55](docs/experiments/cutedsl-cta-tiles-mcts-v55.md), [CTA tile search v57](docs/experiments/cutedsl-cta-tiles-mcts-v57.md), [mixed mutation/LLM smoke](docs/experiments/cutedsl-mixed-bmut4-bgen1-v1.md).
- **Template schedule controls:** [single warp group](docs/experiments/cutedsl-single-warp-group-v10.md), [WGMMA in-flight groups](docs/experiments/cutedsl-wgmma-inflight-v11.md), [TMA load policy](docs/experiments/cutedsl-tma-load-policy-v12.md), [epilogue stages](docs/experiments/cutedsl-epilogue-stages-v13.md), [canonical epilogue](docs/experiments/cutedsl-epilogue-canonical-v14.md), [epilogue MCTS](docs/experiments/cutedsl-epilogue-bmut2-v14.md), [pipeline mutation](docs/experiments/cutedsl-pipeline-bmut6-v7.md), [pipeline interactions](docs/experiments/cutedsl-pipeline-interactions-v8.md), [shared-memory swizzle](docs/experiments/cutedsl-smem-swizzle-v15.md), [canonical swizzle](docs/experiments/cutedsl-smem-swizzle-canonical-v18.md).
- **Independent structural path:** [TMA contract](docs/experiments/cutedsl-independent-tma-v23.md), [cooperative copy](docs/experiments/cutedsl-cooperative-v52.md), [CTA N](docs/experiments/cutedsl-cta-n-v54.md), [CTA MN](docs/experiments/cutedsl-cta-mn-v56.md), [cluster shape](docs/experiments/cutedsl-cluster-v58.md), [cluster MCTS](docs/experiments/cutedsl-cluster-mcts-v59.md), [pipeline depth](docs/experiments/cutedsl-pipeline4-v60.md), [prefetch](docs/experiments/cutedsl-prefetch-v63.md), [warp specialization](docs/experiments/cutedsl-warp-specialization-v67.md), [register repartition](docs/experiments/cutedsl-register-repartition-v69.md).

## October: make the CUDA context auditable

The next question was whether the model would benefit from knowing what NVCC, the GPU, and NCU actually produced. We added a versioned compiler-context record, retained hash-checked PTX blobs in SQLite, parsed `ptxas` resources, and stored bounded SASS category counts for valid nodes. A H100 smoke validated two durable PTX blobs and two SASS summaries. A same-worker flag comparison found identical normalized SASS and only a 1.17 µs span across four root timings with and without PTX-retention/reporting flags. [Compiler validation](docs/experiments/compiler-context-h100-v2.md)

We also added a diagnostic H100 resource graph, launch/resource constraints, and per-node dataflow views in the trace browser. The PTX extractor records bounded within-block dependencies and explicitly marks unknown aliasing, control-flow joins, runtime bytes, and critical paths. The H100 graph records capacities and architectural limits; sustainable bandwidth and latency on most links remain uncalibrated. NCU provides whole-kernel measurements separately. These distinctions prevent a static graph from being mistaken for measured movement. [H100 graph](docs/h100-resource-graph.md) · [Dataflow](docs/bf16-dataflow.md) · [Extraction plan](docs/ptx-dataflow-extraction-plan.md)

### Prompt-context study: many runs, no stable winner

A one-seed Terra `B_gen=10` pilot found 3,219.3 µs with source only, 4,080.1 with NCU, 1,856.8 with compiler summary, and 1,956.9 with both. It justified a larger study, not a ranking. [Pilot](docs/experiments/compiler-prompt-ablation.md)

The later `B_gen=50` study used three matched seeds per row below. Each cell is the run's best latency; the median is across seeds 0–2. “Compiler” here means bounded `ptxas` resources and static SASS categories, **not raw PTX or SASS**. All arms still collected NCU on the usual expansion schedule; source and compiler arms hid the parent profile from the prompt. [Full study and curves](docs/experiments/terra-sol-prompt-context-bgen50.md)

| Model | NCU set | Prompt | Seed 0 | Seed 1 | Seed 2 | Median |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Terra | Lightweight | Source | 1,792.9 | 1,936.2 | 1,062.0 | 1,792.9 |
| Terra | Lightweight | NCU | 3,073.6 | 1,803.7 | 3,170.4 | 3,073.6 |
| Terra | Lightweight | Compiler | 1,417.7 | 1,740.2 | 2,354.8 | 1,740.2 |
| Terra | Lightweight | NCU + compiler | 3,005.2 | 1,500.9 | 2,795.4 | 2,795.4 |
| Sol | Lightweight | Source | 821.3 | 1,042.3 | 815.8 | **821.3** |
| Sol | Lightweight | NCU | 842.7 | 842.0 | 1,000.8 | 842.7 |
| Sol | Lightweight | Compiler | **724.4** | 1,015.7 | 879.5 | 879.5 |
| Sol | Lightweight | NCU + compiler | 989.1 | 914.4 | 847.9 | 914.4 |
| Sol | Diagnostic | NCU | 893.7 | 971.3 | 876.4 | 893.7 |
| Sol | Diagnostic | NCU + compiler | 849.6 | 877.7 | 1,348.3 | 877.7 |

An additional unpaired Sol diagnostic NCU + compiler seed 3 ended at 959.8 µs; it is excluded from the medians. Sol was faster than Terra in all 12 matched lightweight cells, but prompt rankings crossed by seed. Source and compiler each won two of the six lightweight model/seed groups; NCU and combined each won one. Diagnostic NCU + compiler beat diagnostic NCU in two seeds and lost badly in the third. Across the 30 matched runs, 14 first found their final best after call 40, suggesting `B_gen=50` can stop while progress continues. These observations motivate deeper search and targeted prompt tests, **not** a claim that NCU confuses the model or that extra context reliably helps. [Analysis](docs/experiments/terra-sol-prompt-context-bgen50.md)

### PTX dataflow view pilot

We next tested the same compiler-summary prompt with four added PTX views, again using Sol, `B_gen=50`, one seed, and no parent NCU in generation prompts. Every recorded graph view was `partial`. [Pilot](docs/experiments/ptx-dataflow-view-seed0.md) · [Trace analysis](docs/experiments/ptx-dataflow-view-trace-analysis-seed0.md)

| Added PTX view | Best µs | Valid / 50 | Input tokens |
| --- | ---: | ---: | ---: |
| None: compiler control | 865.1 | 40 | 108,283 |
| Counts | 844.1 | 41 | 129,041 |
| Local paths | 831.4 | 42 | 169,640 |
| Bounded subgraph | **780.9** | 40 | 220,395 |
| Full adjacency | 1,087.1 | 40 | 880,052 |

The bounded subgraph won this seed, but its root view selected only three arithmetic edges and omitted 119 others; it did not yet expose a proven memory-to-tensor path. The full view used 8.13× the compiler control's input tokens and ended slower. Winning source edits combined known multi-accumulator, tiling, and async-copy changes. The trace cannot show whether a particular PTX edge caused those edits. The useful next distinction is **fixed-parent, fixed-strategy sampling under different views** versus whole-tree stochastic outcomes. [Trace analysis](docs/experiments/ptx-dataflow-view-trace-analysis-seed0.md)

## October: let Sol refine without a fixed strategy menu

We implemented a separate sequential baseline: each call proposes a complete CUDA replacement from the current verified source, benchmark, NCU profile, compiler summary, and PTX dataflow view. Valid proposals become current even if slower, while best verified source is tracked separately. There is no MCTS selection or strategy menu by default. The client replays bounded visible text turns, not hidden model state. [Design](docs/experiments/sequential-cuda-refinement.md)

| Sequential Sol run | `B_gen` | History | Best µs | Valid calls | Input tokens | Observation |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| Lightweight NCU + compiler + PTX paths | 50 | Two prior turns | **655.856** | 47 | 1,368,533 | Best at call 44; first full sequential pilot. [Report](docs/experiments/sequential-sol-bgen50-seed0.md) |
| Diagnostic NCU, otherwise matched | 50 | Two prior turns | 797.360 | 40, plus 2 infrastructure failures | 1,728,370 | Larger metric set did not improve this seed. [Report](docs/experiments/sequential-sol-diagnostic-bgen50-seed0.md) |
| Lightweight plus best-source field | 50 | Two prior turns | 1,306.080 | 49 | 1,485,707 | Repeating the saved best in 41 prompts did not help this seed. [Report](docs/experiments/sequential-sol-best-anchor-bgen50-seed0.md) |
| Diagnostic, full history, no H100 graph | 25 | All turns within budget | 903.280 | 24 | 3,297,437 | First full-history pilot control. [Report](docs/experiments/sequential-sol-full-history-h100-graph-bgen25-seed0.md) |
| Same, plus H100 architecture graph | 25 | All turns within budget | **779.360** | 22 | 3,115,168 | Graph arm won this one paired seed. [Report](docs/experiments/sequential-sol-full-history-h100-graph-bgen25-seed0.md) |
| Matched strategy-choice study: freeform | 50 | Two prior turns | **647.008** | 47 | 1,289,290 | Best at call 48; lowest verified CUDA C++ latency in this log. [Report](docs/experiments/sequential-strategy-choice-seed0.md) |
| Matched strategy-choice study: fixed eight-strategy catalog | 50 | Two prior turns | 690.416 | 43 | 1,570,914 | Reached 697.616 by call 6; final best at call 44 remeasured an earlier source. [Report](docs/experiments/sequential-strategy-choice-seed0.md) |

The 655.856 µs two-turn run found useful changes without explicit strategy selection: split-K tensor-core bootstrapping, accumulator reuse, shared staging, and CTA grid remapping. Most later edits fit existing strategy themes, but tensor-core bootstrapping and CTA scheduling expose catalog gaps. Its best was 9.46% below the 724.4 µs compiler-only MCTS seed-0 best, while using 12.8× as many input tokens and many more profiles. Prompt context and search policy both changed, so this comparison does not isolate the value of removing strategies. [Per-call edit log](docs/experiments/sequential-sol-bgen50-seed0.md)

The full-history graph arm reached 779.360 versus 903.280 µs without the graph, but this is one 25-call stochastic pair. The graph contains architectural capacities and links, **not** a live traffic or occupancy overlay; each prompt carries measured NCU values separately. Its 2,867,983 cached input tokens were 92.1% of 3,115,168 total. The no-graph arm did not record cache-hit counts, so token totals are not bills. Three graph-arm repair prompts omitted the graph from their latest message, although earlier history still contained it; the request-copy bug was fixed after the run. The [trace analysis](docs/experiments/sequential-sol-full-history-h100-graph-trace-analysis-seed0.md) found distinct 64×96 and 128×128 winner designs, with 650 versus 297 MB measured DRAM reads. It also found that a 68 KiB static shared-memory proposal failed this build's 48 KiB default linker limit despite fitting the architectural maximum. [Paired report](docs/experiments/sequential-sol-full-history-h100-graph-bgen25-seed0.md)

### October 22–23: strategy choice and a controlled kernel crossover

The first strategy-choice pair fixed Sol, seed 0, `B_gen=50`, two prior visible turns, the H100 worker image, lightweight NCU visibility, full bounded `ptxas`/SASS summary, PTX `paths`, and the 4096³ BF16 GEMM contract. Neither arm saw the H100 architecture graph or best-so-far source. The intended prompt difference was the ordered eight-strategy menu. The freeform arm reached **647.008 µs** and the catalog arm **690.416 µs**, a 43.408 µs gap in this pair. The catalog got below 700 µs by call 6, while the freeform best arrived at call 48. Freeform returned self-named strategy IDs on all 50 calls rather than the reserved `freeform` label; this did not gate verification, but the protocol deviation should be resolved before repeating the comparison. These are two sampled trajectories on separate workers, so neither the final gap nor early catalog progress establishes an expected search advantage. [Matched report and curves](docs/experiments/sequential-strategy-choice-seed0.md)

The [source/trace analysis](docs/experiments/sequential-strategy-choice-trace-analysis-seed0.md) found that both winners use 128×128 WMMA macro tiles, K32 double-buffered shared staging, and eight FP32 accumulators per warp. The freeform winner used contiguous productive CTA indices and `cp.async.ca`; the catalog winner used sparse owner CTAs and `cp.async.cg`. The catalog did try an isolated `.cg`→`.ca` edit and regressed from 698.176 to 737.904 µs on its kernel. Its final call-44 best reused the call-41 state, with only a 0.560 µs remeasurement difference. These observations suggested implementation-level choices to test, rather than proving that the semantic strategy set was missing a category.

We then made a **2×2 CTA-placement × cache-policy crossover** from the freeform best source, compiling and correctness-testing each variant four times on one pinned H100. All 16 evaluations passed. Each cell below is the mean of four 30-sample benchmark medians, in µs; the [report](docs/experiments/sequential-cta-cache-crossover/README.md) retains every round, PTX, profile, and worker manifest.

| Productive CTA placement | `cp.async.ca` | `cp.async.cg` |
| --- | ---: | ---: |
| Compact prefix | **644.936** | 660.344 |
| Sparse owners | 673.720 | 688.156 |

On this freeform source family, compact placement saved 28.784 µs with `.ca` and 27.812 µs with `.cg`; `.ca` saved 15.408 µs with compact placement and 14.436 µs with sparse placement. The estimated interaction was about 0.972 µs, below the variation among four rounds. Both intermediate variants improved over sparse-`.cg`, so this crossover does **not** show a performance valley that requires the two choices to be made jointly. A joint proposal might still save `B_gen`; that is a search-efficiency hypothesis, not a benchmark result. The controlled compact-`.ca` versus sparse-`.cg` difference of 43.220 µs resembles the earlier 43.408 µs freeform/catalog gap, but other code differences prevent assigning the historical gap to only these two factors. The catalog's opposite `.ca` response also shows that cache policy remains kernel-specific. [Crossover details](docs/experiments/sequential-cta-cache-crossover/README.md)

## Current interpretation and next discriminating tests

1. **Search representation is consequential.** Typed CuTe reached near cuBLAS on this GEMM, but its validated transformation space was expensive to build and remains specialized. CUDA generation is open-ended and improved far beyond the scalar root; the best logged CUDA C++ search result is now 647.008 µs, still much slower than the separately measured vendor reference. [Direction](docs/cuda-first-next-direction.md)
2. **Model capability was the clearest repeated signal in the prompt study.** Sol beat Terra in all 12 matched lightweight cells. Prompt-content signals were weaker and crossed across three seeds. [Study](docs/experiments/terra-sol-prompt-context-bgen50.md)
3. **Context must be accurate and bounded.** Compiler resources and static PTX dependencies are useful evidence to test, but the present graph cannot assign dynamic bytes or measured time to individual edges. Full adjacency can cost substantially more tokens without a better outcome in a single run. [Trace analysis](docs/experiments/ptx-dataflow-view-trace-analysis-seed0.md)
4. **A single best-of-run latency is a noisy estimator of prompt value.** Compare multiple seeds, best-so-far curves, validity, profiles, actual edits, and token cost. For causal insight, replay identical parent/strategy situations with repeated model samples. [Study](docs/experiments/terra-sol-prompt-context-bgen50.md) · [PTX analysis](docs/experiments/ptx-dataflow-view-trace-analysis-seed0.md)
5. **The next controlled sequential tests** are repeated matched freeform/catalog seeds, after fixing the response-label protocol; then the planned graph and history-depth cells and one-factor ablations for compiler detail, dataflow view, and NCU visibility. The 655.856 µs run's call-34/call-44 remapping replay remains a separate historical check; the completed crossover tested the newer 647.008 µs source family. The graph's measured-traffic overlay and PTX/SASS/NCU path reconciliation remain research tasks. [TODO](TODO.md)

## Reading and updating this log

Add a dated entry or update the relevant phase when a new experiment finishes. Record the exact comparison, budget, model, root/workload, worker image, profile visibility, result, validity, token/cache usage when available, and a link to its detailed report. Distinguish a verified trace result from an interpretation or an untested idea. Avoid pooling latencies across different hardware/toolchain environments as if they were one controlled run. Local SQLite traces contain exact prompts and measurements but are generally ignored by Git; the linked Markdown reports are the durable summary.
