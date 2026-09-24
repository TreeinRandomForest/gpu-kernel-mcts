# Canonical CuTe epilogue validation v14

The promoted schema-v2 epilogue control was evaluated on one NVIDIA H100 80GB HBM3
SXM (SM 9.0) through the canonical `CuTeDSLBackend` path. The environment used CUDA
12.9.1, driver 580.173.02, CUTLASS/CuTe DSL 4.5.1, and PyTorch 2.8.0+cu129.

Both states used tile `(128,256)`, cluster `(2,1)`, 10 warmups, and 30 CUDA-event
measurements under the repository BF16 GEMM contract.

| Epilogue stages | Median (us) | Mean (us) | Stddev (us) | Max/mean error |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 189.008 | 189.893 | 3.218 | 0.0 / 0.0 |
| 3 | 189.168 | 190.130 | 2.669 | 0.0 / 0.0 |

Every aggregate validation check passed. Both representations retained schema v2
and the requested schedule, both evaluations were valid and correct, and the second
compile lookup reused the initial in-memory JIT artifact. Configuration hashes,
normalized compiler IR hashes, and fatbin hashes were distinct. The IR and fatbin
hashes exactly matched the earlier standalone diagnostic:

| Stages | Configuration | Normalized IR | Fatbin |
| ---: | --- | --- | --- |
| 2 | `95dd3f63b75182be...` | `e625141f828ecad2...` | `3c696361140a2c37...` |
| 3 | `978971357ac0951f...` | `ac08ca92545fde89...` | `64e57298c00c8bb4...` |

The 0.160 us median difference is too small to establish a performance winner. This
run validates identity, execution, correctness, and cache semantics; it does not
claim that stage 2 is generally faster than stage 3.
