# Tests for src/baselines/slurm/campaign.py (phased-timeline job expansion) and the train template's data mount / auto-resume logic.
from __future__ import annotations

import inspect
import json
import os
import re
import subprocess
import sys
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import pytest

from baselines.config.schema import RunPaths
from baselines.datasets.splits import RMD17_MOLECULES
from baselines.slurm import campaign, launch, requeue_chain

SEEDS = 3
RMD17_COUNT = 10
MD22_COUNT = 7
SCRATCH = "/home/fakeuser/scratch"

# Calibration (qm9 + rmd17 + md22), per seed: schnet qm9 1 + rmd17 10, painn 1 + 10, ani2x rmd17 10, and 3 tier-1 models x (10 rmd17 + 7 md22).
CALIBRATION_JOBS = SEEDS * ((1 + RMD17_COUNT) * 2 + RMD17_COUNT + 3 * (RMD17_COUNT + MD22_COUNT))
# SPICE, per seed: ani2x, aimnet2, nequip, allegro, mace are launchable; equiformer and equiformerv2 are blocked (tier 2 has no training loop).
SPICE_READY_JOBS = SEEDS * 5
SPICE_BLOCKED_JOBS = SEEDS * 2


def _run_cli(argv: list[str]) -> int:
    return campaign.main(argv)


def test_calibration_phase_job_count_matches_the_hand_derived_matrix():
    jobs = campaign.expand_phase("phase1_calibration")
    assert CALIBRATION_JOBS == 249
    assert len(jobs) == CALIBRATION_JOBS
    assert {j.status for j in jobs} == {"partial"}


def test_spice_phase_marks_tier2_blocked_and_everything_else_partial():
    jobs = campaign.expand_phase("phase1_spice")
    assert sum(j.status == "partial" for j in jobs) == SPICE_READY_JOBS
    assert sum(j.status == "blocked" for j in jobs) == SPICE_BLOCKED_JOBS
    assert {j.tier for j in jobs if j.status == "blocked"} == {"tier2_transformer"}
    assert all(j.dataset == "spice" and j.split == "molecule_95_5" for j in jobs)


def test_nothing_is_marked_ready_except_the_smoke_tests():
    # No adapter evaluates val/test, so even the working training jobs only count as partial.
    for phase in campaign.PHASES:
        for job in campaign.expand_phase(phase):
            assert (job.status == "ready") == (job.job_type == "smoke_test")


def test_smoke_phase_has_one_job_per_tier_on_the_test_partition():
    jobs = campaign.expand_phase("phase0_smoke")
    assert sorted(j.tier for j in jobs) == sorted(launch.TIER_SPECS)
    assert {j.partition for j in jobs} == {"gpu_a40-test"}
    assert all(j.mode == ("zero_shot" if launch.TIER_SPECS[j.tier].needs_mode else None) for j in jobs)


def test_omol25_phases_are_fully_blocked_and_name_both_reasons():
    for phase in ("phase2_zero_shot", "phase2_omol25_finetune"):
        jobs = campaign.expand_phase(phase)
        assert jobs and {j.status for j in jobs} == {"blocked"}
        assert all("omol25" in j.note and "neither zero_shot nor fine_tune" in j.note for j in jobs)
    assert {j.model for j in campaign.expand_phase("phase2_zero_shot")} == set(campaign.ZERO_SHOT_MODELS)


def test_solvation_and_md_phases_expand_to_no_jobs_and_report_the_gap():
    for phase in ("phase3_solvation", "phase4_md"):
        assert campaign.expand_phase(phase) == []
        assert campaign.phase_gap(phase)


def test_unknown_phase_is_rejected():
    with pytest.raises(ValueError, match="unknown phase"):
        campaign.expand_phase("phase9_nope")


def test_every_training_job_names_a_dataset_the_dataset_loading_dispatch_accepts():
    from baselines.datasets import md22

    assert campaign.MD22_MOLECULES == md22.MD22_MOLECULES
    for phase in ("phase1_calibration", "phase1_spice"):
        for job in campaign.expand_phase(phase):
            name, _, molecule = job.dataset.partition(":")
            assert name in {"qm9", "spice", "rmd17", "md22"}
            if name == "rmd17":
                assert molecule in RMD17_MOLECULES
            elif name == "md22":
                assert molecule in md22.MD22_MOLECULES
            else:
                assert molecule == ""
            if name == "rmd17":
                assert re.fullmatch(r"rmd17_split_\d+", job.split)


def test_every_run_name_is_unique_within_a_phase_so_run_dirs_never_collide():
    for phase in campaign.PHASES:
        names = [j.run_name for j in campaign.expand_phase(phase)]
        assert len(names) == len(set(names))


def test_every_tier_model_pair_is_valid_for_the_launcher():
    for phase in campaign.PHASES:
        for job in campaign.expand_phase(phase):
            assert job.model in launch.TIER_SPECS[job.tier].models


def test_models_and_datasets_filters_narrow_the_expansion():
    only = campaign.expand_phase("phase1_calibration", models=("mace",), datasets=("md22",))
    assert len(only) == SEEDS * MD22_COUNT
    assert {(j.model, j.dataset.split(":")[0]) for j in only} == {("mace", "md22")}


def test_pilot_uses_seed_zero_one_dataset_instance_per_family_on_the_test_partition():
    jobs = campaign.expand_phase("phase1_calibration", pilot=True)
    # schnet x (qm9, rmd17), painn x (qm9, rmd17), ani2x x rmd17, 3 tier-1 models x (rmd17, md22)
    assert len(jobs) == 2 + 2 + 1 + 6
    assert {j.seed for j in jobs} == {0}
    assert {j.partition for j in jobs} == {"gpu_a40-test"}


def test_rendered_train_script_has_data_mount_data_root_epochs_seed_and_resume(tmp_path):
    job = [j for j in campaign.expand_phase("phase1_spice") if j.model == "mace" and j.seed == 1][0]
    script = campaign.render_job(job, SCRATCH, epochs=7, extra_adapter_args="--lr 0.01", time=None)
    assert f"{SCRATCH}/mlip-mpnn.sif" in script
    assert f'--bind "{SCRATCH}/data:/data:ro"' in script
    assert "--seed 1 --epochs 7 --data-root /data/spice/SPICE-2.0.1.hdf5 --lr 0.01 $RESUME_ARG" in script
    assert f"--run-dir {SCRATCH}/runs/phase1_spice/tier1_mpnn_mace_spice_molecule_95_5_s1" in script
    assert "#SBATCH --partition=gpu_v100s" in script and "#SBATCH --cpus-per-task=8" in script and "#SBATCH --mem=90G" in script
    assert "$TPL_" not in script


def test_rendered_smoke_script_has_no_data_mount(tmp_path):
    job = campaign.expand_phase("phase0_smoke")[0]
    script = campaign.render_job(job, SCRATCH, epochs=1, extra_adapter_args="", time=None)
    assert ":/data:ro" not in script
    assert "#SBATCH --partition=gpu_a40-test" in script and "#SBATCH --mem=40G" in script


def test_tier2_gets_the_a100_partition_and_no_epochs_flag():
    blocked = [j for j in campaign.expand_phase("phase1_spice") if j.tier == "tier2_transformer"][0]
    script = campaign.render_job(blocked, SCRATCH, epochs=5, extra_adapter_args="", time=None)
    assert blocked.partition == "gpu_a100" and "--epochs" not in script


def test_submit_script_chains_extra_copies_with_afterany_and_uses_relative_paths():
    text = campaign.submit_script(["phase1_spice/a.sbatch", "phase1_spice/b.sbatch"], chain=3)
    lines = text.splitlines()
    assert lines[0] == "#!/bin/bash" and "set -euo pipefail" in lines
    assert any(line.startswith('cd "$(dirname "${BASH_SOURCE[0]}")"') for line in lines)
    sbatch_lines = [line for line in lines if "sbatch --parsable" in line]
    assert len(sbatch_lines) == 6
    assert sum("--dependency=afterany:$PREV" in line for line in sbatch_lines) == 4
    assert all("/home" not in line and '"phase1_spice/' in line for line in sbatch_lines)
    assert campaign.submit_script(["x.sbatch"], chain=1).count("sbatch --parsable") == 1


def test_campaign_module_never_executes_anything():
    source = inspect.getsource(campaign)
    for forbidden in ("import subprocess", "os.system", "os.popen", "Popen("):
        assert forbidden not in source


def test_cli_writes_scripts_submit_script_and_manifest_and_skips_blocked_by_default(tmp_path, capsys):
    out = tmp_path / "camp"
    code = _run_cli(["--phase", "phase1_spice", "--scratch-root", SCRATCH, "--out-dir", str(out), "--epochs", "3", "--chain", "2"])
    assert code == 0
    assert len(list((out / "phase1_spice").glob("*.sbatch"))) == SPICE_READY_JOBS
    manifest = json.loads((out / "manifest_phase1_spice.json").read_text())
    assert len(manifest) == SPICE_READY_JOBS + SPICE_BLOCKED_JOBS
    submit = (out / "submit_phase1_spice.sh").read_text()
    assert submit.count("sbatch --parsable") == SPICE_READY_JOBS * 2
    printed = capsys.readouterr().out
    assert "blocked jobs were skipped" in printed and "no training loop" in printed

    code = _run_cli(["--phase", "phase1_spice", "--scratch-root", SCRATCH, "--out-dir", str(out), "--include-blocked"])
    assert len(list((out / "phase1_spice").glob("*.sbatch"))) == SPICE_READY_JOBS + SPICE_BLOCKED_JOBS


def test_cli_returns_nonzero_and_writes_nothing_for_phases_with_no_code(tmp_path, capsys):
    out = tmp_path / "camp"
    for phase in ("phase3_solvation", "phase4_md"):
        assert _run_cli(["--phase", phase, "--scratch-root", SCRATCH, "--out-dir", str(out)]) == 1
    assert not out.exists()
    assert "BLOCKED" in capsys.readouterr().out


def _resume_block(script: str) -> str:
    start = script.index('if [[ -f "$WORKDIR/checkpoints/checkpoint_final.pt" ]]')
    return script[start:script.index("srun /usr/bin/singularity")]


def _run_resume_block(script: str, workdir: Path) -> subprocess.CompletedProcess:
    program = f'set -euo pipefail\nWORKDIR="{workdir}"\n{_resume_block(script)}\necho "RESUME=[$RESUME_ARG]"\n'
    return subprocess.run(["bash", "-c", program], capture_output=True, text=True, check=False)


def _train_script() -> str:
    return launch.render_sbatch(
        job_type="train", tier="tier1_mpnn", model="mace", dataset="qm9", split="s", partition="gpu_v100s", image="/x.sif", run_dir="/r",
    )


def test_template_resume_block_exits_cleanly_when_the_run_is_already_final(tmp_path):
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "checkpoint_final.pt").write_bytes(b"x")
    result = _run_resume_block(_train_script(), tmp_path)
    assert result.returncode == 0
    assert "already complete" in result.stdout and "RESUME=" not in result.stdout


def test_template_resume_block_passes_no_resume_flag_without_checkpoints(tmp_path):
    result = _run_resume_block(_train_script(), tmp_path)
    assert result.returncode == 0 and "RESUME=[]" in result.stdout


def test_template_resume_block_resumes_from_the_newest_checkpoint(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    older, newer = ckpt_dir / "checkpoint_mace_epoch0.pt", ckpt_dir / "checkpoint_mace_epoch1.pt"
    older.write_bytes(b"a")
    newer.write_bytes(b"b")
    os.utime(older, (1_000_000, 1_000_000))
    os.utime(newer, (2_000_000, 2_000_000))
    result = _run_resume_block(_train_script(), tmp_path)
    assert result.returncode == 0 and f"RESUME=[--resume {newer}]" in result.stdout


def test_template_final_checkpoint_path_matches_runpaths_and_requeue_chain(tmp_path):
    expected = tmp_path / "checkpoints" / "checkpoint_final.pt"
    assert Path(RunPaths(str(tmp_path)).checkpoint("final")) == expected
    assert "$WORKDIR/checkpoints/checkpoint_final.pt" in _train_script()
    assert not requeue_chain.is_training_complete(str(tmp_path))
    expected.parent.mkdir()
    expected.write_bytes(b"x")
    assert requeue_chain.is_training_complete(str(tmp_path))


def test_every_adapter_training_loop_writes_the_final_checkpoint():
    from baselines.adapters import tier0_legacy, tier1_mpnn

    for module in (tier0_legacy, tier1_mpnn):
        source = inspect.getsource(module)
        loops = source.count("for epoch in range(start_epoch, args.epochs):")
        assert loops >= 1
        assert source.count('_save_checkpoint(paths.checkpoint("final"), args.epochs - 1,') == loops


def test_repo_dir_binds_the_checkout_and_puts_it_on_pythonpath_in_every_job_type():
    base = dict(tier="tier0_legacy", model="schnet", dataset="qm9", split="s", partition="gpu_v100s", image="/x.sif", run_dir="/r")
    for job_type in ("train", "eval", "smoke_test"):
        script = launch.render_sbatch(job_type=job_type, repo_dir="/home/u/Eq-FSA", **base)
        assert '--bind "/home/u/Eq-FSA:/repo:ro"' in script
        assert 'export SINGULARITYENV_PYTHONPATH="/repo/src"' in script
        assert "$TPL_" not in script
        assert "SINGULARITYENV_PYTHONPATH" not in launch.render_sbatch(job_type=job_type, **base)


def test_every_campaign_job_mounts_the_repo_including_smoke_tests():
    for phase in ("phase0_smoke", "phase1_spice"):
        for job in campaign.expand_phase(phase)[:3]:
            script = campaign.render_job(job, SCRATCH, epochs=1, extra_adapter_args="", time=None)
            assert f'--bind "{SCRATCH}/Eq-FSA:/repo:ro"' in script
    custom = campaign.render_job(campaign.expand_phase("phase0_smoke")[0], SCRATCH, 1, "", None, repo_dir="/elsewhere/repo")
    assert '--bind "/elsewhere/repo:/repo:ro"' in custom


def test_the_pythonpath_the_template_sets_makes_the_adapters_importable_from_a_bare_checkout():
    # What the container sees: only /repo/src on PYTHONPATH, run from an unrelated working directory.
    code = "import baselines.adapters.tier0_legacy, baselines.adapters.tier1_mpnn, baselines.slurm.campaign; print('ok')"
    env = {**os.environ, "PYTHONPATH": str(_SRC_ROOT)}
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd="/", env=env, check=False)
    assert result.returncode == 0 and "ok" in result.stdout, result.stderr


def test_requeue_chain_dependent_script_keeps_the_repo_and_data_mounts(tmp_path):
    out = tmp_path / "next.sbatch"
    code = requeue_chain.main([
        "--run-dir", str(tmp_path / "run"), "--tier", "tier1_mpnn", "--model", "mace", "--dataset", "qm9", "--split", "s",
        "--partition", "gpu_v100s", "--image", "/x.sif", "--chain-mode", "chain", "--out", str(out),
        "--repo-dir", "/home/u/Eq-FSA", "--data-dir", "/home/u/data",
    ])
    text = out.read_text()
    assert code == 0 and '--bind "/home/u/Eq-FSA:/repo:ro"' in text and '--bind "/home/u/data:/data:ro"' in text


def test_data_dir_option_adds_a_readonly_bind_only_when_given():
    base = dict(job_type="train", tier="tier0_legacy", model="schnet", dataset="qm9", split="s", partition="gpu_v100s", image="/x.sif", run_dir="/r")
    assert '--bind "/d:/data:ro"' in launch.render_sbatch(**base, data_dir="/d")
    assert ":/data:ro" not in launch.render_sbatch(**base)
