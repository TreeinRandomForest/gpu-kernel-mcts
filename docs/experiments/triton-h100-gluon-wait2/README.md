# Four-slot wait-two Gluon GEMM on H100

## Question

Can a fourth A/B shared-memory slot and `wgmma.wait_group 2` improve the
validated three-slot, wait-one Gluon GEMM? The candidate keeps the same
`128 × 256 × 64` CTA tile, two-CTA cluster, 8 warps, ordinary per-CTA B
TMA loads, epilogue, and fixed `4096³` BF16 timing workload. It preloads
four K tiles. After issuing the current WGMMA, it allows the two newest
groups to remain pending and refills the slot from two steps earlier only
after that group's completion.

## Correctness and compiler result

Both arms compiled on H100 with no spills and passed exact comparison to
`torch.mm` for seed 0. Outputs were checked again after every timing round.
The candidate and control each passed five additional launches for seeds 1
and 2 at `4096³`, plus `2048³` and `4096 × 2048 × 2048` at seed 0.

SASS confirms one loop wait with operand `0x2` in the four-slot candidate
and `0x1` in the three-slot control. The final epilogue waits with `0x0` in
both. NCU counted the same 1,048,576 dynamic HGMMA warp instructions per arm.

| Resource or NCU metric | Four-slot wait-two | Three-slot wait-one |
| --- | ---: | ---: |
| Shared memory per CTA | 197.664 KB | 148.504 KB |
| Registers per thread / spills | 154 / 0 | 154 / 0 |
| Tensor pipe active | 84.46% | 84.70% |
| DRAM bytes read | 174.107 MB | 174.076 MB |
| Executed warp instructions | 20.912 M | 24.524 M |

NCU used separate replay passes; its duration is not a latency sample. The
additional shared memory and lower instruction count did not raise tensor-pipe
activity or change measured DRAM traffic meaningfully.

## Paired CUDA Graph timing

The first same-worker run used three rounds, 30 samples per arm per round:

| Arm | Round 0 | Round 1 | Round 2 |
| --- | ---: | ---: | ---: |
| Four-slot wait-two | 187.296 µs | 199.472 µs | 197.520 µs |
| Three-slot wait-one | 187.792 µs | 187.984 µs | 199.008 µs |
| Optimized Triton B | 186.080 µs | 195.488 µs | 192.784 µs |
| cuBLAS | 176.640 µs | 186.720 µs | 185.472 µs |

The candidate-minus-control paired differences were −0.496, +11.488, and
−1.488 µs. In rounds 1 and 2, the candidate ran last while the reference
arms also slowed, so these timings did not identify a stable gain.

A second H100 run used the **same cubins** and eight rounds with each arm
twice in every order position. The eight candidate-minus-control differences
were +0.592, +3.936, +0.096, +0.784, −7.440, −2.928, −0.400, and
+0.272 µs. Their median was **+0.184 µs**; the candidate was faster in
three rounds and slower in five. All arms drifted upward by more than 10 µs
over the run. The balanced evidence supports **no demonstrated material
latency improvement** from the extra slot and wait-two schedule for this
fixed workload. A sub-microsecond effect remains below this measurement's
resolution.

## Artifacts and decision

- `kernel.py`: four-slot candidate. The three-slot control is
  `../triton-h100-gluon-wait1/kernel.py`.
- `run.py` and `paired.py`: first timing, correctness stress, compiler
  capture, and matched NCU passes.
- `run_balanced.py` and `paired_balanced.py`: balanced-order timing follow-up.
- `results.json`, `balanced_results.json`, and the respective `output/`
  folders preserve identities and raw measurements. Run `analyze.py` to
  regenerate `analysis.json`.

Both Nebius H100 instances and their boot disks were removed; resource lists
were empty after each run. Keep the simpler three-slot wait-one kernel as the
validated fixed-shape choice. The next optimization question is the remaining
gap to cuBLAS, for which CTA scheduling and data movement merit a matched
profile before another source rewrite.
