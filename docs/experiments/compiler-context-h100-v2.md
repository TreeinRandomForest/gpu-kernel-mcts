# H100 compiler-context v2 validation

## Setup

- Date: 2026-10-12
- Code commit in worker image: `8ce8d7e` (`compiler-context-v2-8ce8d7e`)
- Controller: that commit plus the `SearchCLIProgress.store_compiler_artifact`
  forwarding fix in the working tree
- Worker: Nebius H100 80 GB SXM, CUDA 12.8.2, Nsight Compute 2025.1.1
- Search: deterministic smoke generator, one generation, no mutation budget
- Run ID: `4dacb701-9df4-47d4-8967-86f2bb527836`
- Local trace: `compiler-context-h100-v2.sqlite` (ignored by Git)

## SQLite artifact and SASS result

The run completed one MCTS iteration with `B_gen=1`, one profile call, and two
valid nodes. The root and candidate each linked to one PTX blob in
`compiler_artifacts`; both blobs passed a fresh SHA-256 check after copying the
trace from `/tmp`. SQLite `PRAGMA integrity_check` returned `ok`.

| Node | Median (us) | PTX bytes | SASS instructions | SASS status |
|---|---:|---:|---:|---|
| Root | 28,244.735 | 7,220 | 160 | parsed |
| Smoke candidate | 148,914.101 | 3,848 | 264 | parsed |

The PTX content is in `compiler_artifacts.content`, referenced through
`node_compiler_artifacts`. Node metadata stores hashes, sizes, storage status,
`ptxas` resources, and bounded SASS category counts. The worker-local paths in
metadata are provenance only; the BLOBs remain available after VM teardown.

The first H100 attempt exposed a missing artifact-storage forwarding method in
the CLI progress wrapper before root evaluation. The wrapper was fixed and
covered by a focused test; the successful run above used that fix. Neither
attempt left a VM or disk behind.

## Same-worker compiler flag comparison

A separate H100 session compiled the same root source and harness twice. The
first build used production flags `--keep --keep-dir ... -Xptxas=-v`; the second
omitted those three retention/reporting options. Both used `-arch=sm_90`,
`-std=c++17`, `-O3`, `-rdc=true`, and `-lcublas`. Both passed correctness.

The executable-file SHA-256 values differed, but the normalized SASS
fingerprints were identical (`d2a96dfee1a331dee8bcd11f3139a6d23716ca4cb1ceef5d61352c0cf2a18333`).
The four benchmark medians were interleaved in retained/plain/plain/retained
order:

| Build | First median (us) | Second median (us) |
|---|---:|---:|
| With retention/reporting flags | 28,456.976 | 28,457.521 |
| Without those flags | 28,458.144 | 28,457.328 |

The observed medians span 1.17 us (about 0.004% of the kernel time). This is
evidence that the flags did not change generated device instructions or
meaningfully change timing for this root on this worker. It is not a claim that
the executable files are byte-identical or that every candidate is unaffected.
The comparison VM and disk were deleted; a final Nebius list returned no
instances or disks.
