# GLM-5.3 Triton H100 smoke — 2026-10-28

A three-generation MCTS smoke completed on a Nebius H100 SXM with the
Messages-compatible EnMaaS GLM-5.3 endpoint. All three candidates compiled,
passed correctness, and received CUDA-event timing. The VM and disk were
deleted after the run; Nebius listed no remaining instance.

## Configuration and result

- Source: `d0db3502135fb97ad4cd7f18493fc06f86953a20` on
  `feature/triton-h100`, with uncommitted Messages thinking-mode wiring
  (`dirty_tree=1`). Focused local tests: 38 passed.
- Worker image: `docker.io/saarora/gpu-kernel-mcts@sha256:d0cd8e7b784ac8828691aa0add0e41d1802e22ef11745780de018afaa12c0469`.
- Run ID: `f99195e4-0940-42a3-8635-337b999b6baa`; trace:
  `triton-h100-glm53-bgen3-thinking-disabled.sqlite` (Git-ignored).
- Model: `rits/zai-org/glm-5-3` through Anthropic-compatible Messages;
  `--messages-thinking disabled`, seed 0, `B_gen=3`, one repair allowed,
  `lightweight_v1` NCU, and `configs/triton_h100_strategies.json`.
- Same-worker calibration: Triton root 819.760 µs, correctness-passing
  cuBLAS 177.184 µs, correctness-passing CUTLASS 308.800 µs.

| Node | Strategy | Median µs | GLM latency s | Status |
| --- | --- | ---: | ---: | --- |
| Measured search root | — | 820.672 | — | Valid |
| Generation 1 | `improve_b_layout_access` | 243.872 | 8.33 | Valid |
| Generation 2 | `improve_b_layout_access` | 239.616 | 9.50 | Valid |
| Generation 3 | `change_warp_count` | **199.712** | 9.61 | Valid |

The best node is 4.11× faster than this run's measured Triton root. It is
12.7% slower than the same-worker cuBLAS reference. This is one sampled tree,
not evidence that GLM or a particular strategy wins an ablation.

The run recorded four nodes, three generation calls, three lazy NCU profiles,
four PTX artifacts, and one `worker_calibration` event. Generation 1 and 2
both expanded the root; generation 3 expanded the second child. The recorded
prompts include the dynamic `launch(A, B, C, M, N, K)` ABI, measured parent
latency (820.672 µs for generations 1 and 2; 239.616 µs for generation 3),
the 177.184 µs cuBLAS reference, parent source, lightweight NCU profile,
workload, hardware target, and selected strategy. This arm did not include a
Triton compiler summary, dataflow view, or hardware resource graph in prompts.

## Gateway finding

The first attempt used default Messages behavior and a 180-second client
timeout. Root evaluation and NCU passed, but proposal 1 timed out with a
transport error; that trace ended at `B_gen=1` with no generation record. A
GPU-free retry of its exact 1,959-input-token prompt exhausted 4,096 output
tokens in 65 seconds and then 16,384 output tokens in 249 seconds, both with
`stop_reason=max_tokens` and no text block. The gateway accepted
`thinking: {"type": "disabled"}`; the same prompt then returned text using
551 output tokens. That setting was added as the opt-in CLI flag above and
recorded in the successful trace. The failed attempt's VM was also deleted.
