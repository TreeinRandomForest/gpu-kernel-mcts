# CUDA compiler-summary prompt ablation

The generator already receives CUDA source, workload, strategy, target hardware,
and the parent NCU profile. This experiment varies only two prompt inputs:

| Arm | Parent NCU profile | Parent compiler summary | CLI flags |
|---|---|---|---|
| Source control | hidden | hidden | `--omit-parent-profile` |
| NCU | shown | hidden | defaults |
| Compiler | hidden | shown | `--omit-parent-profile --include-compiler-summary` |
| Combined | shown | shown | `--include-compiler-summary` |

All arms still collect NCU on expansion and final-best profiling. This keeps the
evaluation and profile schedule fixed; `--omit-parent-profile` hides NCU evidence
from generation only. Use the uniform prior and leave incoming profile deltas off.
Keep the worker image, root, model, strategy catalog, search configuration, seed,
and budget identical across arms. Pilot at `B_gen=10` for one seed, then compare
`B_gen=50` over three matched seeds if the pilot is sound. Repairs consume `B_gen`.

The opt-in `parent_compiler_summary` contains only allowlisted integer `ptxas`
resources, static SASS category-site counts, a schema version, and a limitation
statement. Raw PTX, artifact paths, dataflow arrows, and inferred traffic are not
included. Missing compiler evidence is explicit as `status: unavailable`.

Each generation's full `prompt_text`, `prompt_hash`, and exact summary in
`metadata_json.generation.parent_compiler_summary` are stored in SQLite. The
run configuration stores both option flags. Before comparing performance, check
that these fields match the assigned arm for every generation, that all runs used
the same worker image/model/strategy catalog, and that `final_b_gen` reached the
budget. Report valid generations, best latency/speedup versus `B_gen`, input and
output tokens, profile calls, and wall time. Static SASS counts are compiler
evidence, not dynamic instruction counts or measured memory traffic.

## B_gen=10 pilot, seed 0

Run on 2026-10-12 with `gpt-5.6-terra`, medium reasoning, uniform priors,
`c_puct=12`, one repair, `lightweight_v1` profiles, and
`docker.io/saarora/gpu-kernel-mcts:compiler-context-v2-8ce8d7e`. Each run used
`configs/strategies.yaml` at SHA-256
`cc7820e2558eb13b9f9dfa344ee2c2f09b101b56a4648e473c3e0273593dafdc`.
The worker manifests all report H100 SXM, CUDA toolkit 12.8.2, and NCU 2025.1.1.
The image was recorded by tag, not immutable digest; pin it before the main study.
The registry reported that tag as
`sha256:f90da644bcabc3a07756ecea933f4e1bc2d7faa5785e1c5d39c851eecedb0429`
after the pilot. Use the digest reference for all main-study arms.

| Arm | Valid / 10 | Best median (µs) | Root median (µs) | Input / output tokens | Profiles |
|---|---:|---:|---:|---:|---:|
| Source | 5 | 3,219.3 | 28,458.9 | 20,554 / 24,142 | 2 |
| NCU | 6 | 4,080.1 | 28,460.2 | 28,101 / 25,448 | 2 |
| Compiler | 6 | 1,856.8 | 28,246.0 | 22,152 / 24,043 | 2 |
| Combined | 5 | 1,956.9 | 28,245.6 | 28,496 / 28,791 | 3 |

Local traces are `prompt-ablation-{source,ncu,compiler,combined}-bgen10-seed0.sqlite`
and are intentionally ignored by Git. All four reached `final_b_gen=10`, passed
SQLite integrity checks, and had 10 prompt hashes matching their stored text.
Every prompt exposed exactly the assigned NCU/summary fields; compiler-arm
summaries were `available` and matched the copy in generation metadata. Nebius
listed no remaining instances or disks after the pilot.

The compiler arms found faster candidates in this one-seed pilot, but the sample
is too small to attribute the difference to compiler context. The completed
50-call Terra/Sol and Sol diagnostic comparisons are reported in
[the main study](terra-sol-prompt-context-bgen50.md).
