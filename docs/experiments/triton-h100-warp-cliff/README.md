# Triton H100 four-warp `BN=256` cliff — 2026-10-30

This follow-up uses three exact sources from the [winner crossover](../triton-h100-winner-crossover/README.md): grouped scheduling with `(BN, num_warps)` of `(128, 4)`, `(256, 4)`, and `(256, 8)`. `BM=128`, `BK=64`, the pointer loop, and `num_stages=3` are fixed. One Nebius H100 SXM worker compiled, correctness-checked, and timed each source, captured Triton's compiler resource record and PTX, and ran the `lightweight_v1` NCU profile. All three passed correctness. The VM and disk were deleted after the run.

| `BN` | Warps | Median, µs | Registers/thread | Triton `n_spills` | Shared memory, KiB | Active warps, % | Tensor pipe active, % | SM throughput, % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 4 | 224.416 | 232 | 0 | 96 | 12.13 | 89.85 | 87.52 |
| 256 | 4 | **1,777.296** | **255** | **256** | 144 | **6.28** | **10.68** | **11.39** |
| 256 | 8 | 201.280 | 213 | 0 | 144 | 12.48 | 91.59 | 87.14 |

The slow `(256, 4)` compile reaches the 255-register/thread limit and is the only one for which Triton reports spills. It executes 76.6 million instructions in the NCU pass, versus 40.0 million for `(256, 8)`, with 3.40 versus 1.26 long-scoreboard stalled warps per issue-active cycle. The large timing gap, spill record, extra instructions, and collapsed tensor-pipe activity together strongly implicate register pressure and spill-induced work. The two `BN=256` variants use the same 144 KiB of shared memory, so their shared-memory allocation alone does not explain the gap.

The measured active-warp percentages are also consistent with a single resident CTA per SM: 4/64 warps is 6.25%, and 8/64 is 12.5%. The 144 KiB shared-memory allocation can limit residency to one CTA on H100. Thus the occupancy percentages by themselves do not establish that register pressure reduced CTA residency. The lightweight profile does not include local-memory transaction counts or source-attributed SASS. Each saved PTX has zero textual `ld.local` and `st.local` sites; the nonzero `n_spills` is a *compiled cubin* resource result, so PTX text is not sufficient to observe the spill traffic directly. The exact contribution of spills versus other code-generation changes remains unquantified.

The same-worker startup calibration measured cuBLAS at 177.664 µs, the Triton root at 821.248 µs, and CUTLASS at 308.368 µs. The three candidate timings reproduce the direction and scale of the earlier four-round crossover. This is one shape, one worker, one toolchain, and one evaluation per variant; the crossover provides the repeated timing evidence.

[run.py](run.py) checks each source hash against the crossover record and writes [results.json](results.json) with the worker manifest, correctness, timing samples, compiler metadata, and NCU metrics. The three sibling `.ptx` files are the exact captured artifacts. Source revision was `d06e2347aa7fee93b7a4c9cc6b9290892daac5ca` with a dirty tree; the pinned worker image was `docker.io/saarora/gpu-kernel-mcts@sha256:d0cd8e7b784ac8828691aa0add0e41d1802e22ef11745780de018afaa12c0469`.
