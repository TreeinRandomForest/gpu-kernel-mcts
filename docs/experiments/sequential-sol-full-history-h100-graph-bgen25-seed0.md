# Sequential CUDA refinement: full history and H100 graph, Sol seed 0

This paired 25-call pilot tests whether adding the bounded H100 architectural resource graph to a full-history sequential prompt helps. Both arms used `gpt-5.6-sol` at medium reasoning, the same 4096³ BF16 GEMM root, CUDA C++ backend, `diagnostic_v2` NCU profile with full visibility, full compiler summary (PTX and SASS summary), PTX dataflow `paths` with limits, one repair per proposal, and the pinned H100 worker image `sha256:f90da644bcabc3a07756ecea933f4e1bc2d7faa5785e1c5d39c851eecedb0429`. Both used seed 0 and retained up to 25 prior text turns, so every call carried the complete history accumulated within this run. No fixed strategy menu or best-so-far source field was enabled.

| Measure | No H100 graph | H100 graph |
| --- | ---: | ---: |
| Best saved-node median | 903.280 µs | **779.360 µs** |
| Call that created best node | 17 | 25 |
| Valid / invalid generation calls | 24 / 1 | 22 / 3 |
| Distinct nodes and profiles, including root | 25 / 25 | 22 / 22 |
| Input / output tokens | 3,297,437 / 87,964 | 3,115,168 / 60,396 |
| Recorded cached / cache-write input tokens | unavailable | 2,867,983 / 247,110 |

The graph-arm best was 123.920 µs (13.7%) below the no-graph best. This is one stochastic paired seed and does not establish a general benefit. The runs followed different candidate trajectories, and the graph arm spent three calls on invalid proposals versus one in the control. The H100 graph is architectural context; it has no measured per-link traffic, sustainable bandwidth, latency, or live occupancy overlay. Measured NCU occupancy and traffic were available separately in each call's `current_profile`.

The graph appears in all 22 normal graph-arm proposal messages. The three repair messages omitted the graph from their latest message because the shared repair path rebuilt `GenerationRequest` without its newer optional fields. The full-history input still contained the graph in earlier turns. That repair-path omission was fixed **after** the experiment by preserving all request fields when adding repair evidence; the saved trace records the original prompts. The change also preserves the optional best-so-far field for future repairs. This caveat matters when interpreting the pilot as a graph ablation.

OpenAI returned cache-hit counts for the graph arm: 2,867,983 of 3,115,168 input tokens (92.1%) were cached. The no-graph arm preceded cache-usage instrumentation, so its actual cache share is unknown. Total input-token counts therefore should not be interpreted as uncached-token usage or cost. The graph arm's maximum single-request input was 247,113 tokens; the no-graph arm's was 272,887.

Run IDs: no graph `e5692824-dec8-42e1-80de-56ac25052f34`; H100 graph `04bfeb1a-29e5-441d-9e11-0f13f83c1a6b`. Local artifacts: [no-graph trace](../../sequential-sol-full-history-diagnostic-bgen25-nograph-seed0.sqlite), [no-graph best source](../../sequential-sol-full-history-diagnostic-bgen25-nograph-seed0-best.cu), [graph trace](../../sequential-sol-full-history-diagnostic-bgen25-h100graph-seed0.sqlite), and [graph best source](../../sequential-sol-full-history-diagnostic-bgen25-h100graph-seed0-best.cu). Both traces passed SQLite integrity checks. All 50 prompt arrays contained the expected full-history length, and all 47 stored profiles identified `diagnostic_v2`. The runners exited successfully, and Nebius listed no instance or disk afterward.
