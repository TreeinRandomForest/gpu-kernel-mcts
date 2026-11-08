"""Render bounded one-CTA Gluon tile variants from a validated source."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from kernel_mcts.domain import KernelProgram
from kernel_mcts.generation import GenerationRequest, GenerationResult, ProposalBudgetKind
from kernel_mcts.strategy_applicability import triton_parent_stage


HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "single_cta_gluon_fixture.py"
TEMPLATE_SHA256 = "0a19a0237606f07d36148bd735b0a8a2ff70a798c76739eee39d432c51dd2393"


@dataclass(frozen=True)
class SingleCtaShape:
    cta_n: int = 256
    block_k: int = 64

    def __post_init__(self) -> None:
        if self.cta_n not in (128, 256) or self.block_k not in (32, 64):
            raise ValueError("unsupported one-CTA tile shape")


def render(shape: SingleCtaShape, template: str | None = None) -> str:
    """Change only the declared tile constants and matching WGMMA N shape."""
    source = TEMPLATE.read_text() if template is None else template
    if hashlib.sha256(source.encode()).hexdigest() != TEMPLATE_SHA256:
        raise ValueError("one-CTA template changed; review the renderer")
    substitutions = (
        ("CTA_N = 256", f"CTA_N = {shape.cta_n}", 1),
        ("BLOCK_K = 64", f"BLOCK_K = {shape.block_k}", 1),
        ("instr_shape=[16, 256, 16]", f"instr_shape=[16, {shape.cta_n}, 16]", 2),
    )
    for old, new, expected in substitutions:
        if source.count(old) != expected:
            raise ValueError(f"unexpected template sites for {old}")
        source = source.replace(old, new)
    if triton_parent_stage(source) != "single_cta_gluon":
        raise ValueError("rendered source lost one-CTA Gluon stage")
    return source


class SingleCtaShapeMutationGenerator:
    """Offer each bounded variant once from the validated source root."""

    strategy_id = "single_cta_tile_shape"
    shapes = (SingleCtaShape(cta_n=128), SingleCtaShape(block_k=32))

    def __init__(self, template: str | None = None) -> None:
        self.template = TEMPLATE.read_text() if template is None else template
        render(SingleCtaShape(), self.template)
        self._issued: set[SingleCtaShape] = set()

    @staticmethod
    def proposal_budget_kind(_request: GenerationRequest) -> ProposalBudgetKind:
        return ProposalBudgetKind.MUTATION

    def can_generate(self, request: GenerationRequest) -> bool:
        return (
            request.parent.backend == "triton_h100"
            and request.parent.source == self.template
            and request.strategy.id == self.strategy_id
            and request.workload.benchmark_id == "bf16_gemm_4096_h100"
            and request.hardware.get("gpu_model") == "H100"
            and any(shape not in self._issued for shape in self.shapes)
        )

    def generate(self, request: GenerationRequest) -> GenerationResult:
        if not self.can_generate(request):
            raise RuntimeError("no untried one-CTA shape mutation for parent and action")
        shape = next(item for item in self.shapes if item not in self._issued)
        source = render(shape, self.template)
        self._issued.add(shape)
        parameters = {"cta_n": shape.cta_n, "block_k": shape.block_k}
        identity = {"parent_sha256": TEMPLATE_SHA256, "strategy_id": self.strategy_id,
                    "parameters": parameters}
        canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        return GenerationResult(
            generation_id=f"mutation:{uuid4()}",
            raw_output=json.dumps(parameters, sort_keys=True),
            program=KernelProgram(source, backend="triton_h100"),
            prompt_hash=hashlib.sha256(canonical.encode()).hexdigest(),
            metadata={"generator": "typed_mutation", "llm_call": False,
                      "template_sha256": TEMPLATE_SHA256, "parameters": parameters},
        )
