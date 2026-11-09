from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path

_REPO_SRC = Path(__file__).resolve().parents[2]
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

import pytest

from baselines.slurm import launch

BASE = dict(job_name="j", partition="gpu_v100s", image="/s/mlip-mpnn.sif", run_dir="/s/runs/r", commands=["echo hi"], scratch_root="/s", repo_dir="/s/Eq-FSA")


def test_template_renders_directives_binds_and_commands():
    text = launch.render_sbatch(**BASE)
    for needle in ("#SBATCH --job-name=j", "#SBATCH --partition=gpu_v100s", "#SBATCH --gres=gpu:1", "#SBATCH --time=5-00:00:00",
                   'IMAGE="/s/mlip-mpnn.sif"', '--bind "/s:/s"', '--bind "/s/Eq-FSA:/repo:ro"', "/opt/image_digest.txt:ro", "echo hi",
                   'export SINGULARITYENV_PYTHONPATH="/repo/src"', 'export APPTAINERENV_PYTHONPATH="/repo/src"',
                   'command -v apptainer || command -v singularity', 'exec --nv "${BINDS[@]}"', "#SBATCH --output=logs/%x_%j.out"):
        assert needle in text, needle
    assert "$TPL_" not in text and "/s/mlip-mpnn.sif.sha256" in text
    assert "/usr/bin/singularity" not in text  # hard-coded path replaced by a PATH lookup


def test_cpu_job_has_no_gres_and_no_nv_flag():
    text = "\n".join(ln for ln in launch.render_sbatch(**{**BASE, "gpus": 0}).splitlines() if not ln.lstrip().startswith("#") or ln.startswith("#SBATCH"))
    assert "--gres" not in text and "--nv" not in text


def test_default_walltime_by_partition():
    for partition, expected in (("gpu_v100s", "5-00:00:00"), ("gpu_a100", "5-00:00:00"), ("gpu_a40-test", "02:00:00"), ("stingy", "05:00:00")):
        assert f"--time={expected}" in launch.render_sbatch(**{**BASE, "partition": partition})
    assert "--time=00:15:00" in launch.render_sbatch(**BASE, smoke=True)
    assert "--time=1-00:00:00" in launch.render_sbatch(**BASE, time="1-00:00:00")


def test_unknown_partition_without_explicit_time_raises():
    with pytest.raises(ValueError, match="--time is required"):
        launch.render_sbatch(**{**BASE, "partition": "mystery"})
    assert "--time=01:00:00" in launch.render_sbatch(**{**BASE, "partition": "mystery"}, time="01:00:00")


def test_unknown_model_and_force_field_and_image_rejected():
    with pytest.raises(ValueError, match="unknown model"):
        launch.train_commands("equiformer", "/e", "/r", 0, 1)  # Equiformer v1 was dropped
    with pytest.raises(ValueError, match="forcefield must be"):
        launch.classical_commands("amber99", "/e", "/r", "/c")
    with pytest.raises(ValueError, match="no smoke test"):
        launch.smoke_commands("mlip-foundation")  # tier 3 was dropped


def test_train_commands_pass_seed_epochs_extra_args_and_quote_paths():
    train, evaluate = launch.train_commands("mace", "/e x", "/r", 4, 9, "--lr 0.001")[1:]
    assert "--data-dir '/e x'" in train and "--seed 4" in train and "--epochs 9" in train and train.endswith("--lr 0.001")
    assert evaluate.startswith("[[ -f /r/metrics.json ]] || ") and "--seed 4" in evaluate


def test_export_commands_skip_a_complete_export():
    lines = launch.export_commands("qm9", "s", "/d", "/o", None, None, None, 0)
    assert lines[0].startswith("if [[ -f /o/split.json ]]") and "already complete" in lines[0] and lines[-1] == "fi"


def test_main_writes_a_script_and_never_submits(tmp_path, capsys):
    out = tmp_path / "x.sbatch"
    code = launch.main(["--model", "equiformerv2", "--export-dir", "/s/exports/e", "--run-dir", "/s/runs/r", "--scratch-root", "/s",
                        "--repo-dir", "/s/Eq-FSA", "--partition", "gpu_a100", "--out", str(out)])
    assert code == 0 and out.exists()
    assert "baselines.trainers.fairchem_wrapper" in out.read_text() and f"sbatch {out}" in capsys.readouterr().out


def test_launch_never_calls_sbatch_subprocess():
    call_pattern = re.compile(r"subprocess\.(run|call|Popen|check_call|check_output)\s*\(\s*\[\s*[\"']sbatch[\"']")
    code_only = "\n".join(ln for ln in inspect.getsource(launch).splitlines() if not ln.strip().startswith("#"))
    assert call_pattern.search(code_only) is None and "import subprocess" not in code_only
