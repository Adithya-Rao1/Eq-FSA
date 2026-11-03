import inspect
import os
import sys
from pathlib import Path

_REPO_SRC = Path(__file__).resolve().parents[2]
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

import pytest

from baselines.slurm import launch, requeue_chain


def test_train_template_renders_singularity_and_adapter_cli(tmp_path):
    script = launch.render_sbatch(
        job_type="train",
        tier="tier1_mpnn",
        model="mace",
        dataset="rmd17",
        split="rmd17_fixed_1000",
        partition="gpu_a100",
        image="/home/fakeuser/scratch/mlip-mpnn.sif",
        run_dir=str(tmp_path / "run0"),
    )
    assert "srun /usr/bin/singularity exec" in script
    assert "--nv" in script
    assert '--bind "$WORKDIR:/work"' in script
    assert "python -m baselines.adapters.tier1_mpnn --model mace --dataset rmd17 --split rmd17_fixed_1000 --run-dir " in script
    # placeholders must be fully resolved --> no unresolved "$TPL_..." marker should exist
    assert "$TPL_" not in script
    # bash's own variables must survive safe_substitute untouched (purpose of safe_substitute + a TPL_ prefix instead of just substitute()).
    assert '"$IMAGE"' in script
    assert "$SLURM_CPUS_PER_TASK" in script


def test_smoke_test_template_renders_import_check(tmp_path):
    script = launch.render_sbatch(
        job_type="smoke_test",
        tier="tier0_legacy",
        model="schnet",
        dataset="rmd17",
        split="rmd17_fixed_1000",
        partition="gpu_a40-test",
        image="/home/fakeuser/scratch/mlip-legacy.sif",
        run_dir=str(tmp_path / "run0"),
    )
    assert "import baselines.adapters.tier0_legacy" in script
    assert "cuda_available" in script
    assert "#SBATCH --time=00:15:00" in script


def test_tier3_foundation_requires_mode(tmp_path):
    with pytest.raises(ValueError):
        launch.render_sbatch(
            job_type="train",
            tier="tier3_foundation",
            model="mace_mp_0",
            dataset="rmd17",
            split="rmd17_fixed_1000",
            partition="gpu_a100",
            image="/home/fakeuser/scratch/mlip-foundation.sif",
            run_dir=str(tmp_path / "run0"),
        )


def test_tier3_foundation_mode_appears_in_adapter_args(tmp_path):
    script = launch.render_sbatch(
        job_type="train",
        tier="tier3_foundation",
        model="mace_off23",
        dataset="rmd17",
        split="rmd17_fixed_1000",
        partition="gpu_v100s",
        image="/home/fakeuser/scratch/mlip-foundation.sif",
        run_dir=str(tmp_path / "run0"),
        mode="fine_tune",
    )
    assert "--model mace_off23" in script
    assert "--mode fine_tune" in script


def test_unknown_model_for_tier_rejected(tmp_path):
    with pytest.raises(ValueError):
        launch.render_sbatch(
            job_type="train",
            tier="tier0_legacy",
            model="nequip",  # valid for tier1_mpnn, not tier0_legacy
            dataset="rmd17",
            split="rmd17_fixed_1000",
            partition="gpu_a100",
            image="/home/fakeuser/scratch/mlip-legacy.sif",
            run_dir=str(tmp_path / "run0"),
        )


def test_default_walltime_by_partition(tmp_path):
    script = launch.render_sbatch(
        job_type="train",
        tier="tier0_legacy",
        model="schnet",
        dataset="rmd17",
        split="rmd17_fixed_1000",
        partition="debugq",
        image="/home/fakeuser/scratch/mlip-legacy.sif",
        run_dir=str(tmp_path / "run0"),
    )
    assert "#SBATCH --time=00:30:00" in script


def test_unknown_partition_without_explicit_time_raises(tmp_path):
    with pytest.raises(ValueError):
        launch.render_sbatch(
            job_type="train",
            tier="tier0_legacy",
            model="schnet",
            dataset="rmd17",
            split="rmd17_fixed_1000",
            partition="some_future_partition",
            image="/home/fakeuser/scratch/mlip-legacy.sif",
            run_dir=str(tmp_path / "run0"),
        )


def test_write_script_and_cli_main_never_submit(tmp_path, capsys):
    out_path = tmp_path / "rendered.sbatch"
    exit_code = launch.main(
        [
            "--job-type", "train",
            "--tier", "tier2_transformer",
            "--model", "equiformer",
            "--dataset", "rmd17",
            "--split", "rmd17_fixed_1000",
            "--partition", "gpu_a100",
            "--image", "/home/fakeuser/scratch/mlip-transformer.sif",
            "--run-dir", str(tmp_path / "run0"),
            "--out", str(out_path),
        ]
    )
    assert exit_code == 0
    assert out_path.exists()
    printed = capsys.readouterr().out
    assert f"sbatch {out_path}" in printed
    assert "wrote" in printed


def test_requeue_chain_prints_scontrol_requeue_when_run_dir_incomplete(tmp_path, capsys):
    run_dir = tmp_path / "run0"
    os.makedirs(run_dir)
    exit_code = requeue_chain.main(
        [
            "--run-dir", str(run_dir),
            "--tier", "tier1_mpnn",
            "--model", "mace",
            "--dataset", "rmd17",
            "--split", "rmd17_fixed_1000",
            "--partition", "gpu_a100",
            "--image", "/home/fakeuser/scratch/mlip-mpnn.sif",
            "--chain-mode", "requeue",
        ]
    )
    assert exit_code == 0
    printed = capsys.readouterr().out
    assert "scontrol requeue" in printed


def test_requeue_chain_reports_complete_run_as_done(tmp_path, capsys):
    run_dir = tmp_path / "run0"
    checkpoint_dir = run_dir / "checkpoints"
    os.makedirs(checkpoint_dir)
    (checkpoint_dir / "checkpoint_final.pt").write_bytes(b"")
    exit_code = requeue_chain.main(
        [
            "--run-dir", str(run_dir),
            "--tier", "tier1_mpnn",
            "--model", "mace",
            "--dataset", "rmd17",
            "--split", "rmd17_fixed_1000",
            "--partition", "gpu_a100",
            "--image", "/home/fakeuser/scratch/mlip-mpnn.sif",
        ]
    )
    assert exit_code == 0
    printed = capsys.readouterr().out
    assert "already has a final checkpoint" in printed


def test_requeue_chain_time_left_parsing():
    assert requeue_chain._parse_slurm_time_left("1-02:03:04") == 1 * 86400 + 2 * 3600 + 3 * 60 + 4
    assert requeue_chain._parse_slurm_time_left("00:05:00") == 300
    assert requeue_chain._parse_slurm_time_left("") is None
    assert requeue_chain._parse_slurm_time_left("garbage") is None


def test_launch_never_calls_sbatch_subprocess():
    import re

    call_pattern = re.compile(r"subprocess\.(run|call|Popen|check_call|check_output)\s*\(\s*\[\s*[\"']sbatch[\"']")
    for module in (launch, requeue_chain):
        code_lines = [line for line in inspect.getsource(module).splitlines() if not line.strip().startswith("#")]
        code_only = "\n".join(code_lines)
        assert call_pattern.search(code_only) is None, f"found a real sbatch-submitting subprocess call in {module.__name__}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
