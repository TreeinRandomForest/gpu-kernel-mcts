# A-reuse WGMMA wait policy

The [initial A-reuse experiment](../triton-h100-gluon-a-reuse/README.md)
used `wgmma.wait_group 0` inside the K loop. The original 256-wide Gluon
kernel used `wait_group 1`. This paired H100 experiment tests whether that
serialization explains much of the split kernel's latency. The [kernel](kernel.py)
has a compile-time wait-group parameter; [run provenance](results.json),
[paired timings](output/paired.json), emitted PTX, and NCU exports are in
this directory. The split arms use identical tiling, A reuse, TMA loads,
shared allocation, and launch geometry; only the per-step wait differs.

All arms compiled and passed bitwise equality against `torch.mm` for the
fixed 4096³ BF16 `A @ B.T` workload. Ten rotating-order CUDA Graph rounds
on one H100 SXM yielded:

| Arm | Median of round medians | Range |
| --- | ---: | ---: |
| cuBLAS | 181.448 µs | 175.168–191.200 µs |
| Original 256-wide Gluon | 188.048 µs | 181.488–196.896 µs |
| Split A reuse, wait 0 | 249.824 µs | 231.408–269.856 µs |
| Split A reuse, wait 1 | 204.128 µs | 197.248–214.688 µs |

The paired wait-0 minus wait-1 difference was positive in all ten rounds,
with a **46.968 µs median**. Wait 1 remained **16.136 µs slower** than the
original at the paired median. NCU showed essentially identical DRAM reads
(174.087 MB), TMA loads (98,304), and executed instructions (33.736M) for
the two split arms. Tensor-pipe activity during elapsed cycles rose from
63.10% to 83.96%. The emitted PTX confirms `wait_group 0` versus
`wait_group 1` inside the K loop.

The [wait-2 follow-up](../triton-h100-gluon-a-reuse-wait2/README.md) found
no material additional gain. These are single-worker, single-seed,
single-shape measurements. Correctness passed for this shape, but this is
not a general proof of buffer safety for altered pipeline depth or tiling.
