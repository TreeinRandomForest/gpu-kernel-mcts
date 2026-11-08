# Reconciling Gluon search timing with CUDA Graph replay

The [earlier search-adapter crossover](../triton-h100-gluon-search-adapter/README.md)
ranked the 66-cluster persistent Gluon kernel behind the one-CTA kernel
(267.760 versus 250.800 µs), while the [factorial CUDA Graph run](../triton-h100-gluon-factorial/README.md)
measured the persistent kernel near 184 µs. This experiment executes the
**same source fixtures and compiled cubin per candidate** under four timing
modes on one H100 SXM. This measurement preceded the subsequent timing
contract change; the [Graph contract validation](../triton-h100-graph-contract-validation/README.md)
documents the implementation.

The fixed workload is 4096³ BF16 `A @ B` with the search ABI's column-major
B view. Both Gluon kernels and cuBLAS passed bitwise equality against
`torch.mm`. The prepared Gluon calls verified that they use the same cubin
as their respective `launch()` calls. Eight rounds randomized the order of
the three kernels and four modes. Each per-call mode used 10 warmups and
30 event samples per round; the batched mode timed 30 normal calls within
one event pair and divided by 30. The [measurement code](measure.py),
[round observations](paired.json), and [worker/source provenance](results.json)
are preserved.

## What each mode includes

| Mode | Work between start and end CUDA events |
| --- | --- |
| `search_event` | The current search harness's `launch()`: Python validation, B transpose view, layout/descriptor construction, Triton dispatch |
| `prepared_event` | Direct call to the same compiled cubin with descriptors prepared before the start event |
| `batched_event` | 30 ordinary `launch()` calls queued between one event pair; per-call time is the event interval divided by 30 |
| `graph_event` | Replay of a captured `launch()` GPU operation; Python descriptor construction happened during capture, not replay |

The start event can execute on the GPU while Python is still building
descriptors. The end event executes after the kernel. Thus a per-call CUDA
event interval can include GPU idle time between those two submissions; it
does not automatically isolate device execution.

## Same-worker results

Medians of eight round medians, in microseconds:

| Kernel | Search event | Prepared event | Batched normal | Graph replay |
| --- | ---: | ---: | ---: | ---: |
| One CTA Gluon | 259.696 | 201.944 | 187.908 | 191.800 |
| 66-cluster persistent Gluon | 276.576 | 201.936 | 188.457 | 191.568 |
| cuBLAS | 197.944 | 195.848 | 179.665 | 184.040 |

The persistent-minus-one-CTA **paired median** was +16.216 µs in
`search_event` (persistent slower in all eight rounds), but −0.848 µs in
`prepared_event` and −0.384 µs in `graph_event`. The two kernels are not
meaningfully separated by the latter modes on this worker.

Within each kernel, `search_event − prepared_event` was **56.520 µs** for
one CTA and **76.320 µs** for persistent at the paired median. The median
Python `launch()` wall times measured separately without per-call sync were
58.309 and 76.400 µs, respectively. Their difference closely tracks the
search-mode ranking difference. `prepared_event − graph_event` was about
10–11 µs for both Gluon kernels and cuBLAS, while batched normal launches
were close to Graph replay. These results strongly implicate host submission
and per-call descriptor construction in the old timing gap; they do not
partition every microsecond exactly because CPU and GPU work can overlap,
and absolute times drifted over rounds.

## Implication for search

At the time of this experiment, the [Triton harness](../../../src/kernel_mcts/triton_harness.py)
placed CUDA events around the Python `launch()` call, including descriptor
construction. This experiment showed that timing placement changed candidate
ranking. `spec.md` Section 50 now defines CUDA Graph replay as the
`triton_h100` search reward, with per-call launch events retained as a
diagnostic. Historical per-call reward curves must be kept separate from
Graph-timed curves. Graph replay measures the captured GPU work and is not
automatically representative of a one-off application launch.
