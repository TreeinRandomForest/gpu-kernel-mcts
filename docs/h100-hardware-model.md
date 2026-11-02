# H100 hardware model: current state

**Status:** experimental diagnostic model, updated 2026-11-02. The canonical
implementation is [`resource_graph.py`](../src/kernel_mcts/resource_graph.py);
PTX annotation rules are in
[`resource_overlay.py`](../src/kernel_mcts/resource_overlay.py). This document
tracks evidence and open questions as we inspect more kernels. `spec.md`
remains the source of truth for search behavior. The model does not currently
choose MCTS actions, predict latency, or prune candidates.

## Scope and identity

The graph is generated only for one non-MIG H100 SXM 80 GB GPU with compute
capability 9.0, using an observed worker manifest. Its graph format is
`schema_version=2`; the current set of resources and routes is
`topology_revision=2`. An older stored graph without `topology_revision`
denotes revision 1. Source, PTX, binary, and profiler evidence belong to a
particular kernel/node; the architectural graph is shared. The worker manifest
identifies the GPU and supplies observed HBM capacity or SM count when
available. Published nominal values fill missing fields and retain that basis.

Graph nodes and routes are **diagnostic abstractions**. For example,
`l2_shared_cp_async` is a logical L1-bypassing path, not a claim that L2 and
shared memory have a dedicated physical wire. A candidate route says an
instruction *may* use those resources. It does not identify a runtime cache
hit, executed instruction count, transferred bytes, or bottleneck.

## Resources and architectural limits

| Resource | Scope | Current recorded property | Evidence basis |
| --- | --- | --- | --- |
| HBM | GPU | Worker-observed bytes when present; otherwise 80 GB nominal | Manifest or published nominal |
| L2 | GPU | 50 MB nominal | Published nominal |
| SM | GPU | Worker-observed count when present; otherwise 132 nominal | Manifest or published nominal |
| L1 | Per SM | Capacity unknown because L1/shared carveout varies | Explicit unknown |
| Shared memory | Per SM | 228 KiB configurable maximum | Published maximum |
| Registers | Per SM | 65,536 32-bit registers | Published nominal |
| Scalar compute | Per SM | 128 FP32 CUDA cores | Published nominal; not measured throughput |
| Tensor compute | Per SM | 4 tensor cores | Published nominal; not measured throughput |
| Warp exchange | Per warp | Shuffle-unit count and throughput unknown | Logical resource from `shfl`/`vote` |

The graph also records limits: 255 registers/thread, 65,536 registers/block,
1,024 threads/block, 2,048 resident threads/SM, 32 resident blocks/SM, and
227 KiB shared memory/block. L1 and shared memory share a configurable
256 KiB per-SM pool, so their separate maxima cannot be added. These are
architectural limits, not achievable occupancy or a performance prediction.
The code retains the cited source and basis for each value.

## Logical routes

Every route below has `sustainable_bandwidth=null` and `latency=null`.
`hbm_l2` additionally carries a **published peak** of 3.35 TB/s; that is not
a measured sustainable rate for this route.

| Route ID | Direction | Intended operation family |
| --- | --- | --- |
| `hbm_l2` | Both | Memory-fabric participation |
| `l2_l1` | Both | Cache fills and movement |
| `l1_registers` | Both | L1-participating global loads/stores |
| `l2_registers` | L2 → registers | `.cg` global load bypassing L1 |
| `l1_shared` | Both | L1-participating `cp.async` |
| `l2_shared` | Both | TMA candidate route |
| `l2_shared_cp_async` | L2 → shared | `.cg` async copy bypassing L1 |
| `shared_registers` | Both | Shared loads/stores, including `ldmatrix` |
| `registers_scalar` | Both | Scalar/vector arithmetic operands/results |
| `shared_tensor` | Shared → tensor | Shared-fed tensor operations such as WGMMA |
| `registers_tensor` | Registers → tensor | Register-fed `mma.sync` operands |
| `tensor_registers` | Tensor → registers | Tensor accumulators/results |
| `registers_warp_exchange` | Both | Cross-lane register exchange |

Revision 2 added the L1-bypass, register-fed tensor, and warp-exchange routes
after a real attention kernel exposed those gaps. The
[PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/) describes
`.cg` as L2 caching without L1, `ldmatrix` as shared-to-register loading,
and `shfl.sync` as register exchange within a warp. The graph still avoids
assigning a physical-link rate to these logical routes.

## Measurements that are *not* link calibrations

The standalone [H100 characterization](experiments/h100-characterization/README.md)
keeps raw timing, NCU, and provenance files. Selected dependent-read medians
from one H100 worker are:

| End-to-end dependency chain | Working set | Median ns/hop |
| --- | ---: | ---: |
| `.ca` L1-eligible global load | 4 KiB | 20.36 |
| `.cg` L1-bypassing global load | 1 MiB | 142.71 |
| `.cg` HBM-participating random load | 128 MiB | 346.68 |
| Shared-memory load | 4 KiB | 14.34 |
| Register arithmetic control | — | 2.24 |

These times include instruction issue, dependencies, address calculation,
loop overhead, and sometimes address translation. The HBM-participating case
is not an isolated HBM latency. The [latency analysis](experiments/h100-characterization/output/latency_analysis.json)
therefore leaves `physical_link_calibrations` empty.

Access pattern and concurrency are also consequential. At 528 Triton
programs, an 8 MiB contiguous read/reduce took 2.639 µs versus 17.594 µs
with a 32-word stride, despite zero measured DRAM reads in both separate
NCU passes. For 256 MiB, the corresponding medians were 93.802 µs and
1,448.490 µs, with 15.6× more measured DRAM traffic in the strided case.
Changing the contiguous 256 MiB grid from 132 to 2,112 programs changed
the median from 229.769 to 86.895 µs. These are conditioned workload-path
observations in the [sensitivity analysis](experiments/h100-characterization/output/sensitivity_analysis.json),
not link bandwidths or a pure occupancy sweep.

## Kernel overlay validation

The [overlay experiment](experiments/h100-resource-overlay/README.md) links
PTX sites to candidate graph resources while preserving exact source/PTX
hashes, original proven def-use edges, and parser uncertainty. Whole-kernel
NCU evidence remains separate. Its [local viewer](experiments/h100-resource-overlay/viewer.html)
can follow proven edges in an individual overlay.

| Case | What it established | Remaining caveat |
| --- | --- | --- |
| Fast/slow Triton GEMM variants | A slow 4-warp case had 255 registers/thread, 256 compiled spills, and heavy measured local-memory sectors; an 8-warp variant had 213 registers/thread and no compiled spills | PTX had no textual local loads in the slow case; SASS/NCU evidence cannot be assigned to exact PTX sites |
| Real 256-token causal BF16 attention | Correctness passed; 750 PTX sites and 1,039 proven edges include async copies, `ldmatrix`, `mma.sync`, exponentials, and warp shuffles | Shared-memory aliases, loop-carried values, predication, and runtime execution counts remain unresolved |

The [attention analysis](experiments/h100-resource-overlay/output/attention_analysis.json)
has no remaining *topology* path gap for that kernel. Its separate NCU L2
hit-rate value was 101.2%, outside the valid range, and is marked invalid.
This validation tests representation coverage; it does not show that graph
context improves search or that the model predicts attention performance.

## Open parts of the model

- No route has an isolated sustainable bandwidth or latency calibration.
  Future probes need payload size, access order, alignment, working set,
  cache policy, concurrency, clocks, raw samples, and counter evidence.
- Local-memory traffic, including spills, lacks a reliable PTX-site route.
  A compiled kernel may show local loads in SASS/NCU while its PTX has no
  textual `ld.local` site.
- Shared-memory aliasing, loop-carried values, predication, and cross-block
  reuse limit the static dataflow graph. A missing proven path is
  inconclusive.
- Tensor/scalar issue throughput, special-function operations, synchronization,
  and contention have not been characterized. TMA, cluster, and multicast
  behavior are not validated by the current attention case.
- The graph has architectural limits but no complete per-kernel live
  occupancy/capacity overlay. Per-valid-node persistence and prompt use remain
  [TODOs](../TODO.md), subject to a `spec.md` update before search behavior
  changes.

## How to update this document and model

1. **Keep an exact kernel identity.** Record source/PTX hashes, binary
   fingerprint when available, compiler/driver versions, GPU manifest,
   correctness, and profiling scope. A new kernel's static overlay belongs
   to that kernel; the hardware graph is shared.
2. **Identify the gap before adding a route.** Inspect the PTX opcode and,
   where relevant, SASS and NCU. Distinguish a missing topology route from a
   missing def-use edge, unobserved cache outcome, or unsupported instruction.
   Preserve uncertainty rather than creating a speculative connection.
3. **Classify numeric evidence.** Keep published limits, end-to-end path
   measurements, whole-kernel counters, and isolated link calibrations in
   separate fields. Do not copy a path time or aggregate NCU counter into a
   link or instruction site.
4. **Version and verify changes.** Update `resource_graph.py` and annotation
   tests together. Increment `topology_revision` when resources/routes change;
   increment `schema_version` only when the graph format changes. Regenerate
   affected overlay summaries, record the evidence and limitations here, and
   keep old persisted traces identifiable by their recorded revision.
5. **Gate search use separately.** A graph change does not by itself enter
   MCTS prompts or priors. Update `spec.md` before changing search/trace
   behavior, then compare graph-informed and uninformed runs under the same
   budget. Measured latency remains the reward.

### Revision history

| Revision | Change | Evidence |
| --- | --- | --- |
| 1 | Initial H100 resources, memory paths, compute paths, and architectural limits | Published H100/Hopper sources and worker manifest |
| 2 | Logical `.cg` bypass paths, register-fed `mma.sync`, and warp exchange; costs remain unknown | Compiled causal-attention PTX, correctness run, and PTX ISA semantics |
