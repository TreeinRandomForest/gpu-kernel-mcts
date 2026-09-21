from __future__ import annotations

import hashlib

import pytest

from kernel_mcts.cute_capabilities import inspect_cute_structural_capabilities


_EXAMPLE = '''
class HopperWgmmaGemmKernel:
    def __init__(self, acc_dtype, tile_shape_mn=(128, 256), pipeline_stages=3, use_2cta_instrs=False):
        assert pipeline_stages >= 2, "pipeline requires at least two stages"
        if tile_shape_mn[0] % 64:
            raise ValueError("tile must match WGMMA")
        self.pipeline_stages = pipeline_stages
        self.use_2cta_instrs = use_2cta_instrs
        self.smem_capacity = 232448
        self.occupancy = 1

    @staticmethod
    def _compute_stages(self, smem_capacity, occupancy) -> tuple[int, int]:
        available = smem_capacity // 1024
        if available < 2:
            raise ValueError("insufficient shared memory")
        return self.pipeline_stages, occupancy

def run(a, b, *, use_tma_store=True, epilogue_tile=(64, 64)) -> None:
    producer_warp = 0
    return tma_store(epilogue_tile, producer_warp)

def is_valid_cluster(cluster_shape_mn):
    return cluster_shape_mn[0] > 0

parser.add_argument("--pipeline-stages", type=int, choices=(2, 3, 4), default=3)
parser.add_argument("--use-2cta-instrs", action="store_true")
'''.lstrip()


def test_structural_capability_report_is_static_evidence(tmp_path) -> None:
    example = tmp_path / "dense_gemm.py"
    example.write_text(_EXAMPLE, encoding="utf-8")

    report = inspect_cute_structural_capabilities(example)

    assert report["source"]["sha256"] == hashlib.sha256(
        _EXAMPLE.encode("utf-8")
    ).hexdigest()
    constructor = report["symbols"]["HopperWgmmaGemmKernel.__init__"]
    assert [parameter["name"] for parameter in constructor["parameters"]] == [
        "self",
        "acc_dtype",
        "tile_shape_mn",
        "pipeline_stages",
        "use_2cta_instrs",
    ]
    assert constructor["parameters"][3]["default"] == "3"
    run = report["symbols"]["run"]
    assert run["parameters"][2]["kind"] == "keyword_only"
    assert run["returns"] == "None"
    assert report["cli_options"][0]["flags"] == ["--pipeline-stages"]
    assert report["cli_options"][0]["keywords"]["choices"] == "(2, 3, 4)"
    assert report["validation_evidence"]["assertions"][0]["test"] == (
        "pipeline_stages >= 2"
    )
    assert "ValueError" in report["validation_evidence"]["raises"][0]["expression"]
    assert report["validation_evidence"]["validation_functions"][0]["name"] == (
        "is_valid_cluster"
    )
    controls = report["candidate_structural_controls"]
    assert "pipeline_stages" in controls["pipeline"]["identifiers"]
    stage_definition = controls["pipeline"]["definitions"][0]
    assert stage_definition["name"] == "_compute_stages"
    assert stage_definition["decorators"] == ["staticmethod"]
    assert [item["name"] for item in stage_definition["parameters"]] == [
        "self",
        "smem_capacity",
        "occupancy",
    ]
    assert stage_definition["return_expressions"] == [
        "(self.pipeline_stages, occupancy)"
    ]
    assert stage_definition["assignments"] == [
        {
            "line": 13,
            "targets": ["available"],
            "value": "smem_capacity // 1024",
        }
    ]
    assert stage_definition["conditions"] == [
        {"line": 14, "test": "available < 2"}
    ]
    assert controls["pipeline"]["state_assignments"][:2] == [
        {"line": 8, "targets": ["self.smem_capacity"], "value": "232448"},
        {"line": 9, "targets": ["self.occupancy"], "value": "1"},
    ]
    assert "use_2cta_instrs" in controls["wgmma"]["identifiers"]
    assert "use_tma_store" in controls["tma"]["identifiers"]
    assert "epilogue_tile" in controls["epilogue"]["identifiers"]
    assert "producer_warp" in controls["warp_specialization"]["identifiers"]
    assert all(control["status"] == "evidence_only" for control in controls.values())
    assert all(
        not any("GPU device kernel" in identifier for identifier in control["identifiers"])
        for control in controls.values()
    )
    assert report["selection_decision"] == "not_made"


def test_structural_capability_report_requires_pinned_symbols(tmp_path) -> None:
    example = tmp_path / "other.py"
    example.write_text("def run(): pass\n", encoding="utf-8")

    with pytest.raises(ValueError, match="HopperWgmmaGemmKernel"):
        inspect_cute_structural_capabilities(example)
