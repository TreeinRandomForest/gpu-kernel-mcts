"""One bounded static-feedback repair of the model-generated serial WGMMA source."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

from kernel_mcts.benchmarks import BF16_GEMM_WORKLOAD
from kernel_mcts.config import load_data, parse_strategies
from kernel_mcts.domain import EvaluationResult, InvalidReason, KernelProgram, ProposalStatus
from kernel_mcts.generation import GenerationRequest
from kernel_mcts.llm_generation import LLMKernelGenerator
from kernel_mcts.messages_client import MessagesClient, MessagesConfig
from kernel_mcts.strategy_applicability import triton_parent_stage


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUTPUT = HERE / "v9-wgmma-serial-static-repair"
FINDINGS = (
    "Static API review of your previous source: "
    "`gl.NVMMASharedLayout.get_default_for([BM,BK], gl.bfloat16)` uses a shape "
    "and dtype, not dtype/rep_shape/elem_count keywords; create those layouts "
    "in host launch and pass them through TensorDescriptor. "
    "Inside @gluon.jit allocate `gl.allocate_shared_memory(dtype, "
    "a_desc.block_shape, a_desc.layout)` (likewise B/C), not gl.allocate_shared. "
    "The TMA call order is `tma.async_load(a_desc, [off_m, off_k], bar, a_smem)` "
    "and the analogous B call; put the mbarrier before destination shared memory. "
    "Use `TensorDescriptor.from_tensor(A, [BM,BK], a_layout)` and likewise "
    "`B.T, [BN,BK], b_layout` and `C, [BM,BN], c_layout`; there is no extra "
    "`[1,1]` argument. In host launch check dtype against `torch.bfloat16`, not "
    "`triton.bfloat16`. At output use `c_smem.store(acc.to(gl.bfloat16))`, "
    "not `gl.store(c_smem,...)` or a convert_layout to a shared layout. Then "
    "`fence_async_shared()`, `tma.async_store(c_desc,[off_m,off_n],c_smem)`, "
    "and `tma.store_wait(pendings=0)`; remove `tma.store_commit()`. "
    "Assign `acc = warpgroup_mma_wait(num_outstanding=0,deps=(acc,))` before "
    "the next buffer reuse. Preserve the serial one-buffer loop and complete ABI."
)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    base = os.environ.get("ENMAAS_BASE_URL", "").rstrip("/")
    if not base.startswith("https://") or not os.environ.get("ENMAAS_KEY"):
        raise RuntimeError("ENMAAS_BASE_URL and ENMAAS_KEY must be set")
    parent = KernelProgram((HERE / "v8-packaged-root-mcts-bgen4-complete-gluon.py").read_text(),
                           backend="triton_h100")
    previous = KernelProgram((HERE / "v9-wgmma-serial-probe/attempt-0-candidate.py").read_text(),
                             backend="triton_h100")
    strategy = next(item for item in parse_strategies(load_data(
        ROOT / "configs/triton_h100_gluon_bridge_local_v9.json"))
        if item.id == "single_cta_gluon_tma_wgmma_serial")
    strategy = replace(strategy, prompts={"triton_h100": strategy.prompt_for("triton_h100")
                                           + " " + FINDINGS})
    request = GenerationRequest(
        parent=parent, strategy=strategy, workload=BF16_GEMM_WORKLOAD,
        hardware={"gpu_model": "H100"}, profile=None,
        current_benchmark={"median_us": 1718.960, "measurement_count": 30},
        reference_baselines={"cublas_median_us": 181.040}, attempt=1,
        previous_program=previous,
        previous_result=EvaluationResult(
            ProposalStatus.INVALID, program=previous,
            invalid_reason=InvalidReason.OTHER,
            metadata={"static_review": FINDINGS},
        ),
    )
    client = MessagesClient(MessagesConfig(
        model="rits/zai-org/glm-5-3", url=base + "/messages",
        max_output_tokens=8192, timeout_seconds=180, thinking_mode="disabled"))
    OUTPUT.mkdir()
    result = LLMKernelGenerator(client).generate(request)
    (OUTPUT / "prompt.json").write_text(result.prompt_text or "")
    (OUTPUT / "raw.txt").write_text(result.raw_output)
    source = result.program.source if result.program else None
    if source is not None:
        (OUTPUT / "candidate.py").write_text(source)
    report = {
        "schema_version": 1, "b_gen_for_this_probe": 1,
        "cumulative_wgmma_probe_b_gen": 2, "gpu_compile_attempted": False,
        "model": (result.metadata or {}).get("model"),
        "generation_id": result.generation_id, "prompt_sha256": result.prompt_hash,
        "feedback_sha256": hashlib.sha256(FINDINGS.encode()).hexdigest(),
        "source_sha256": hashlib.sha256(source.encode()).hexdigest() if source else None,
        "candidate_stage": triton_parent_stage(source) if source else "missing",
        "input_tokens": result.input_tokens, "output_tokens": result.output_tokens,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
