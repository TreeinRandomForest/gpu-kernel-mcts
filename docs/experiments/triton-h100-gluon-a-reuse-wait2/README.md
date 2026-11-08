# Wait 1 versus wait 2 for split A reuse

This follow-up compares one versus two outstanding WGMMA groups for the
two-128-wide-tile A-reuse kernel. The [source](kernel.py),
[paired timings](output/paired.json), [NCU exports](output), and
[run provenance](results.json) are preserved. All arms passed bitwise
equality against `torch.mm` for the fixed 4096³ BF16 case.

| Arm | Median of ten round medians |
| --- | ---: |
| cuBLAS | 178.400 µs |
| Original 256-wide Gluon | 185.696 µs |
| Split A reuse, wait 1 | 202.896 µs |
| Split A reuse, wait 2 | 202.528 µs |

The paired wait-1 minus wait-2 difference was **0.128 µs median**, ranging
from −2.448 to +7.600 µs. This does not establish a useful wait-2 speedup.
Wait 2 remained **16.864 µs slower** than the original at the paired median.
The split arms had the same NCU DRAM reads (~174.079 MB), TMA load count
(98,304), executed instruction count (33.736M), and similar elapsed
tensor-pipe activity (83.93% versus 83.11%). PTX confirms the intended
`wait_group 1` and `wait_group 2` instructions.

The extra overlap allowed by wait 2 does not remove the split design's
extra GMMA and instruction work. Future performance work should preserve
the original 256-wide MMA shape or reduce the split kernel's instruction
count before further wait tuning. This result is limited to one worker,
shape, and input seed.
