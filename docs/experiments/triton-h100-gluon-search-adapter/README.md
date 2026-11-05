# Wait-one Gluon source under the Triton H100 search contract

This is a local integration step. It does not claim a new H100 measurement.

The adapter at
`src/kernel_mcts/benchmarks/kernels/bf16_gemm_gluon_wait1.py` retains the
validated device `gemm` function from
`docs/experiments/triton-h100-gluon-wait1/kernel.py`. Its host launcher now
has the search ABI `launch(A,B,C,M,N,K)`: A and C are row-major BF16 views;
B is a column-major BF16 `[K,N]` view with strides `(1,K)`. `B.T` supplies
a zero-copy, contiguous physical `[N,K]` view to the existing TMA descriptor.
The launcher checks shape, dtype, strides, full tiles, and the three-stage K
minimum, then returns the compiled Gluon kernel. The packaged ordinary
Triton root remains the default search root.

The standalone Gluon results used a pinned Triton 3.9 worker image. Earlier
Triton search ablations used a different pinned image; this adapter must be
validated under a Gluon-capable image before a mixed-source run.

`TritonH100Backend` records the compiled kernel name in the artifact and
compiler context. For a lazy profile, NCU selects that exact function name
inside the harness's `KERNEL_MCTS_TARGET` NVTX range. The backend rejects a
CSV report with a different or ambiguous kernel name. This replaces the
fixed `bf16_gemm_triton` selector and lets a valid Gluon source keep its
`gemm` name. The generation prompt now permits `@triton.jit` or `@gluon.jit`
while requiring the same launcher and returned-kernel contract.

Local tests check device-function AST identity, launcher signature and B
transpose, NVTX range lifetime, profile selection and reported identity,
and the updated generation prompt. Relevant tests passed with 67 tests total.

Next: on one pinned H100 worker, run this source through the ordinary search
harness for compile, correctness, CUDA-event timing, and lazy NCU profiling.
Compare the packaged Triton root and same-worker cuBLAS under that same
timing contract. The prior standalone CUDA Graph latencies are not directly
interchangeable with search-harness CUDA-event latencies. Do not use this
candidate as a performance reference until the worker check passes.
