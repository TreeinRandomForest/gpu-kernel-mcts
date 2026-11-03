# CTA-local B descriptor attempt

This experiment asks whether changing the clustered ordinary-B descriptor
layout to CTA-local can remove the extra warp-group waits seen in the
[NCU study](../triton-h100-cluster-gluon-ncu/README.md). The candidate keeps
the same source computation and ordinary B TMA request, but removes the
two-CTA replication map from the B shared-memory layout.

Triton 3.9.0 rejected the CTA-local candidate during Gluon parsing in all
three rounds: the B shared-memory layout has one CTA per CGA while the kernel
context requires two. It emitted no PTX or cubin, so there is no correctness
or performance result for this candidate. The ordinary-replicated and
multicast-replicated controls passed exact correctness and had median-of-round
medians of 505.920 and 392.400 µs, respectively. The median paired multicast
advantage was 113.520 µs, consistent with earlier runs.

The rejected layout is not a valid way to isolate the compiler schedule.
The [PTX control experiment](../triton-h100-cluster-gluon-ptx-control/) follows
the exact compiler output instead.

[`analysis.json`](analysis.json) records the successful controls, the failed
candidate, and artifact hashes. [`results.json`](results.json) and the ignored
`output/paired.json` contain the run and per-round records. The source and
harness are [`kernel.py`](kernel.py), [`paired.py`](paired.py),
[`run.py`](run.py), and [`analyze.py`](analyze.py). The Nebius worker was
cleaned up after the run.
