import triton
import triton.language as tl


@triton.jit
def bf16_gemm_triton(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    a_ptr = tl.make_block_ptr(
        base=A,
        shape=(M, K),
        strides=(K, 1),
        offsets=(pid_m * BM, 0),
        block_shape=(BM, BK),
        order=(1, 0),
    )

    # Column-major [K, N] is physically row-major [N, K]. Loading that
    # representation gives contiguous K accesses for every output column.
    bt_ptr = tl.make_block_ptr(
        base=B,
        shape=(N, K),
        strides=(K, 1),
        offsets=(pid_n * BN, 0),
        block_shape=(BN, BK),
        order=(1, 0),
    )

    acc = tl.zeros((BM, BN), dtype=tl.float32)

    for _ in range(tl.cdiv(K, BK)):
        a = tl.load(
            a_ptr,
            boundary_check=(0, 1),
            padding_option="zero",
        )
        bt = tl.load(
            bt_ptr,
            boundary_check=(0, 1),
            padding_option="zero",
        )
        acc = tl.dot(a, tl.trans(bt), acc)
        a_ptr = tl.advance(a_ptr, (0, BK))
        bt_ptr = tl.advance(bt_ptr, (0, BK))

    c_ptr = tl.make_block_ptr(
        base=C,
        shape=(M, N),
        strides=(N, 1),
        offsets=(pid_m * BM, pid_n * BN),
        block_shape=(BM, BN),
        order=(1, 0),
    )
    tl.store(
        c_ptr,
        acc.to(tl.bfloat16),
        boundary_check=(0, 1),
    )


def launch(A, B, C, M, N, K):
    bm, bn, bk = 128, 128, 64
    grid = (triton.cdiv(M, bm), triton.cdiv(N, bn))
    return bf16_gemm_triton[grid](
        A,
        B,
        C,
        M,
        N,
        K,
        bm,
        bn,
        bk,
        num_warps=8,
        num_stages=3,
    )