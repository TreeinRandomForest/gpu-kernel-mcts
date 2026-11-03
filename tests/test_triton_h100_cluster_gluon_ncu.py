"""CPU checks for targeted Gluon profiling selection."""

import importlib.util
from pathlib import Path


HERE = (Path(__file__).resolve().parents[1] /
        "docs/experiments/triton-h100-cluster-gluon-ncu")


def load(name):
    spec = importlib.util.spec_from_file_location(f"gluon_ncu_{name}", HERE / name)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PROFILE = load("profile_one.py")
RUN = load("run.py")
ANALYZE = load("analyze.py")


def test_profile_arms_are_the_three_double_buffered_controls():
    assert PROFILE.ARMS == {
        "single_pipeline": (False, False, True),
        "cluster_pipeline_ordinary": (True, False, True),
        "cluster_pipeline_multicast": (True, True, True),
    }
    assert set(RUN.ARMS) == set(PROFILE.ARMS)


def test_ncu_command_targets_one_nvtx_scoped_launch():
    command = RUN.profile_command("cluster_pipeline_multicast")
    assert command[command.index("--nvtx-include") + 1] == "GLUON_PROFILE_CLUSTER_PIPELINE_MULTICAST/"
    assert command[command.index("--launch-count") + 1] == "1"
    assert command[command.index("--cache-control") + 1] == "all"


def test_sass_parser_separates_multicast_and_wait_sites(tmp_path):
    sass = tmp_path / "sass.csv"
    sass.write_text(
        'Kernel Name,gemm,\n'
        'Address,Source,Warp Stall Sampling (All Samples),Instructions Executed\n'
        '0x1,"@UP1  UTMALDG.2D.MULTICAST [UR8], [UR20], UR17",2,64\n'
        '0x2,"WARPGROUP.DEPBAR.LE gsb0, 0x0",8,64\n'
        '0x3,"WARPGROUP.DEPBAR.LE gsb0, 0x0",1,1\n'
    )
    parsed = ANALYZE.parse_sass(sass)
    assert parsed["dynamic_executions"]["UTMALDG.2D.MULTICAST"] == 64
    assert parsed["warpgroup_wait_sites"] == 2
    assert parsed["warpgroup_wait_executions"] == 65
    assert parsed["warpgroup_wait_stall_samples"] == 9
