# Tests for src/baselines/slurm/campaign.py (phase expansion) and the job scripts it renders.
from __future__ import annotations

import inspect
import json
import re
import subprocess
import sys
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import pytest

from baselines.datasets.splits import RMD17_MOLECULES
from baselines.slurm import campaign, launch

SEEDS = 3
RMD17_COUNT = 10
MD22_COUNT = 7
N_MODELS = 6  # schnet, painn, nequip, allegro, mace, equiformerv2
SCRATCH = "/home/fakeuser/scratch"
REPO = f"{SCRATCH}/Eq-FSA"


def _render(job, epochs=1, extra="", time=None, repo=REPO):
    return campaign.render_job(job, SCRATCH, repo, epochs, extra, time)


def test_job_counts_match_the_hand_derived_matrix():
    n = lambda phase, **kw: len(campaign.expand_phase(phase, **kw))  # noqa: E731
    assert n("phase0_smoke") == 4
    assert n("phase1_export") == 1 + RMD17_COUNT + MD22_COUNT + 1 + 1  # qm9, rmd17, md22, spice, ani1x
    assert n("phase2_train_calibration") == N_MODELS * (1 + RMD17_COUNT + MD22_COUNT) * SEEDS == 324
    assert n("phase2_train_spice") == N_MODELS * SEEDS
    assert n("phase2_train_ani1x") == N_MODELS * SEEDS
    assert n("phase3_classical") == 2 * (RMD17_COUNT + MD22_COUNT + 1)  # gaff2 + openff, no seeds


def test_classical_never_covers_datasets_without_a_bond_graph():
    families = {j.dataset.split(":")[0] for j in campaign.expand_phase("phase3_classical")}
    assert families == {"rmd17", "md22", "spice"}
    assert {j.model for j in campaign.expand_phase("phase3_classical")} == {"gaff2", "openff"}


def test_dropped_models_are_not_scheduled_anywhere():
    for gone in ("equiformer", "ani2x", "aimnet2", "mace_mp_0", "mace_off23", "orb_v2", "orb_v3", "uma"):
        assert gone not in launch.MODEL_SPECS
        assert gone not in {m for m in campaign.TRAIN_MATRIX}
    assert set(launch.MODEL_SPECS) == {"schnet", "painn", "nequip", "allegro", "mace", "equiformerv2"}
    assert not {p for p in campaign.PHASES if "zero_shot" in p or "omol25" in p or "solvation" in p or "md" == p[-2:]}


def test_every_job_names_a_dataset_the_loading_dispatch_accepts():
    from baselines.datasets import md22

    assert campaign.MD22_MOLECULES == md22.MD22_MOLECULES
    for phase in campaign.PHASES:
        for job in campaign.expand_phase(phase):
            if job.job_type == "smoke_test":
                continue
            name, _, molecule = job.dataset.partition(":")
            assert name in {"qm9", "spice", "rmd17", "md22", "ani1x"}
            if name == "rmd17":
                assert molecule in RMD17_MOLECULES and re.fullmatch(r"rmd17_split_\d+(_val\d+)?", job.split)
            elif name == "md22":
                assert molecule in md22.MD22_MOLECULES
            else:
                assert molecule == ""


def test_run_names_are_unique_within_a_phase_so_run_dirs_never_collide():
    for phase in campaign.PHASES:
        names = [j.run_name for j in campaign.expand_phase(phase)]
        assert len(names) == len(set(names)), phase


def test_every_model_is_served_by_a_wrapper_in_the_image_that_carries_its_library():
    for model, spec in launch.MODEL_SPECS.items():
        assert spec.module.startswith("baselines.trainers.")
        assert spec.takes_model_flag == (spec.module.endswith(("schnetpack_wrapper", "nequip_wrapper")))
    assert {launch.MODEL_SPECS[m].image for m in ("nequip", "allegro", "mace")} == {"mlip-mpnn"}
    assert launch.MODEL_SPECS["equiformerv2"].image == "mlip-transformer"
    assert {launch.MODEL_SPECS[m].image for m in ("schnet", "painn")} == {"mlip-legacy"}


def test_filters_narrow_the_expansion():
    only = campaign.expand_phase("phase2_train_calibration", models=("mace",), datasets=("md22",))
    assert len(only) == SEEDS * MD22_COUNT and {(j.model, j.dataset.split(":")[0]) for j in only} == {("mace", "md22")}
    assert [j.model for j in campaign.expand_phase("phase3_classical", models=("gaff2",), datasets=("spice",))] == ["gaff2"]


def test_pilot_uses_seed_zero_one_dataset_instance_per_family_on_the_test_partition():
    jobs = campaign.expand_phase("phase2_train_calibration", pilot=True)
    assert len(jobs) == N_MODELS * 3  # qm9, rmd17, md22 once each
    assert {j.seed for j in jobs} == {0} and {j.partition for j in jobs} == {"gpu_a40-test"}


def test_equiformerv2_gets_the_a100_partition_and_the_rest_the_v100():
    jobs = campaign.expand_phase("phase2_train_spice")
    assert {j.partition for j in jobs if j.model == "equiformerv2"} == {"gpu_a100"}
    assert {j.partition for j in jobs if j.model != "equiformerv2"} == {"gpu_v100s"}


def test_ani1x_export_is_capped_with_one_seed_and_the_cap_is_in_the_name():
    job = campaign.expand_phase("phase1_export", datasets=("ani1x",))[0]
    script = _render(job)
    assert "--max-train 100000" in script and "--max-val 10000" in script and "--max-test 10000" in script
    assert "--subset-seed 0" in script and "n100000" in job.export_name
    other = _render(campaign.expand_phase("phase1_export", datasets=("qm9",))[0])
    assert "--max-train" not in other


def test_train_jobs_and_their_export_agree_on_the_export_directory():
    export = {j.export_name for j in campaign.expand_phase("phase1_export")}
    for phase in ("phase2_train_calibration", "phase2_train_spice", "phase2_train_ani1x", "phase3_classical"):
        for job in campaign.expand_phase(phase):
            assert job.export_name in export, (phase, job.export_name)
            assert f"{SCRATCH}/exports/{job.export_name}" in _render(job)


def test_every_rendered_script_is_valid_bash_with_no_unfilled_placeholders(tmp_path):
    for phase in campaign.PHASES:
        for i, job in enumerate(campaign.expand_phase(phase)[:25]):
            script = _render(job, epochs=2, extra="--lr 0.001")
            assert "$TPL_" not in script, (phase, job.run_name)
            path = tmp_path / f"{phase}_{i}.sh"
            path.write_text(script)
            assert subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True).returncode == 0, (phase, job.run_name)


def test_binds_repo_and_scratch_in_every_job_type():
    for phase in campaign.PHASES:
        script = _render(campaign.expand_phase(phase)[0])
        assert f'--bind "{SCRATCH}:{SCRATCH}"' in script and f'--bind "{REPO}:/repo:ro"' in script
        assert 'export SINGULARITYENV_PYTHONPATH="/repo/src"' in script
    assert '--bind "/elsewhere/repo:/repo:ro"' in _render(campaign.expand_phase("phase0_smoke")[0], repo="/elsewhere/repo")


def _code(script: str) -> str:
    return "\n".join(ln for ln in script.splitlines() if not ln.lstrip().startswith("#") or ln.startswith("#SBATCH"))


def test_gpu_directives_only_on_gpu_jobs():
    gpu = _code(_render(campaign.expand_phase("phase2_train_spice")[0]))
    assert "#SBATCH --gres=gpu:1" in gpu and "exec --nv" in gpu
    for phase in ("phase1_export", "phase3_classical"):
        cpu = _code(_render(campaign.expand_phase(phase)[0]))
        assert "--gres" not in cpu and "--nv" not in cpu, phase


def test_smoke_script_checks_the_right_imports_per_image():
    scripts = {j.image: _render(j) for j in campaign.expand_phase("phase0_smoke")}
    assert set(scripts) == {"mlip-legacy", "mlip-mpnn", "mlip-transformer", "mlip-classical"}
    assert "baselines.trainers.schnetpack_wrapper" in scripts["mlip-legacy"]
    assert "baselines.trainers.fairchem_wrapper" in scripts["mlip-transformer"]
    assert "openmmforcefields" in scripts["mlip-classical"] and "torch" not in scripts["mlip-classical"]  # that image has no torch
    assert "--time=00:15:00" in scripts["mlip-mpnn"]


def _commands(script: str) -> str:
    return script[script.index("[[ -f"):] if "[[ -f" in script else script


def _run_job_body(script: str, tmp_path: Path, fake_sing: str = 'sing() { echo "SING $*"; }') -> subprocess.CompletedProcess:
    # Executes the rendered command list with the container call replaced by an echo, so the control flow is tested for real.
    body = script[script.index("sing() {"):]
    body = body[body.index("\n}\n") + 3:]  # everything after the sing() definition
    return subprocess.run(["bash", "-c", f"set -euo pipefail\n{fake_sing}\n{body}"], capture_output=True, text=True, cwd=tmp_path)


def _train_job(model="mace"):
    return next(j for j in campaign.expand_phase("phase2_train_spice", models=(model,)) if j.seed == 0)


def _retarget(script: str, tmp_path: Path) -> str:
    return script.replace(SCRATCH, str(tmp_path))


def test_train_script_stops_with_a_clear_error_when_the_export_is_missing(tmp_path):
    result = _run_job_body(_retarget(_render(_train_job()), tmp_path), tmp_path)
    assert result.returncode == 2 and "run the phase1_export jobs first" in result.stderr and "SING" not in result.stdout


def test_train_script_trains_then_evaluates_and_skips_evaluation_once_metrics_exist(tmp_path):
    job = _train_job("nequip")
    export = tmp_path / "exports" / job.export_name
    export.mkdir(parents=True)
    (export / "split.json").write_text("{}")
    script = _retarget(_render(job, epochs=7), tmp_path)
    first = _run_job_body(script, tmp_path)
    lines = [ln for ln in first.stdout.splitlines() if ln.startswith("SING")]
    assert first.returncode == 0 and len(lines) == 2
    assert "baselines.trainers.nequip_wrapper --model nequip" in lines[0] and "--epochs 7" in lines[0] and "--seed 0" in lines[0]
    assert "baselines.evaluation.run_eval --model nequip" in lines[1] and "metrics.json" in lines[1]
    run_dir = tmp_path / "runs" / job.phase / job.run_name
    run_dir.mkdir(parents=True)  # the template's own `mkdir -p "$WORKDIR"` does this before the command list runs
    (run_dir / "metrics.json").write_text("{}")
    again = _run_job_body(script, tmp_path)  # a chained copy of a finished run: training is a wrapper no-op, evaluation is skipped
    assert again.returncode == 0 and sum(ln.startswith("SING") for ln in again.stdout.splitlines()) == 1  # training call only


def test_a_failed_or_unfinished_training_never_reaches_the_evaluation(tmp_path):
    job = _train_job()
    export = tmp_path / "exports" / job.export_name
    export.mkdir(parents=True)
    (export / "split.json").write_text("{}")
    result = _run_job_body(_retarget(_render(job), tmp_path), tmp_path, fake_sing='sing() { echo "SING $*"; return 1; }')
    assert result.returncode != 0 and result.stdout.count("SING") == 1 and "run_eval" not in result.stdout


def test_wrappers_that_serve_one_model_get_no_model_flag(tmp_path):
    assert "--model" not in " ".join(launch.train_commands("mace", "/e", "/r", 0, 1)[1].split("--data-dir")[0:1])
    assert "--model schnet" in launch.train_commands("schnet", "/e", "/r", 0, 1)[1]
    assert "--model" not in launch.train_commands("equiformerv2", "/e", "/r", 0, 1)[1].split("--data-dir")[0]


def test_classical_script_uses_the_charge_cache_and_the_export(tmp_path):
    job = campaign.expand_phase("phase3_classical", models=("openff",), datasets=("spice",))[0]
    script = _render(job)
    assert "baselines.classical.run" in script and "--forcefield openff" in script and f"--charge-cache {SCRATCH}/classical_charges" in script
    assert f"--out {SCRATCH}/runs/phase3_classical/{job.run_name}/metrics.json" in script


def test_submit_script_chains_extra_copies_with_afterany_and_uses_relative_paths():
    text = campaign.submit_script(["phase/a.sbatch", "phase/b.sbatch"], chain=3)
    assert text.count("sbatch --parsable") == 6 and text.count("--dependency=afterany:$PREV") == 4
    assert 'sbatch --parsable "phase/a.sbatch"' in text and "/home" not in text


def test_campaign_module_never_executes_anything():
    source = inspect.getsource(campaign)
    for forbidden in ("import subprocess", "os.system", "os.popen", "Popen("):
        assert forbidden not in source


def test_cli_writes_scripts_submit_script_and_manifest(tmp_path, capsys):
    out = tmp_path / "camp"
    code = campaign.main(["--phase", "phase2_train_spice", "--scratch-root", SCRATCH, "--out-dir", str(out), "--epochs", "3", "--chain", "2"])
    assert code == 0
    assert len(list((out / "phase2_train_spice").glob("*.sbatch"))) == N_MODELS * SEEDS
    assert len(json.loads((out / "manifest_phase2_train_spice.json").read_text())) == N_MODELS * SEEDS
    assert (out / "submit_phase2_train_spice.sh").read_text().count("sbatch --parsable") == N_MODELS * SEEDS * 2
    assert "review, then run on Burgundy" in capsys.readouterr().out


def test_cli_with_filters_that_match_nothing_returns_nonzero_and_writes_nothing(tmp_path):
    out = tmp_path / "camp"
    assert campaign.main(["--phase", "phase3_classical", "--scratch-root", SCRATCH, "--out-dir", str(out), "--datasets", "qm9"]) == 1
    assert not out.exists()


def test_the_rmd17_key_warning_is_attached_to_rmd17_jobs_only(capsys, tmp_path):
    assert all(j.note for j in campaign.expand_phase("phase1_export", datasets=("rmd17",)))
    assert not any(j.note for j in campaign.expand_phase("phase1_export", datasets=("md22", "qm9")))
    campaign.main(["--phase", "phase1_export", "--scratch-root", SCRATCH, "--out-dir", str(tmp_path), "--datasets", "rmd17"])
    assert "malonaldehyde" in capsys.readouterr().out


def test_unknown_phase_is_rejected():
    with pytest.raises(ValueError, match="unknown phase"):
        campaign.expand_phase("phase9_nonsense")


def test_the_pythonpath_the_template_sets_makes_the_wrappers_importable_from_a_bare_checkout():
    code = "import baselines.trainers.mace_wrapper, baselines.slurm.campaign, baselines.slurm.launch; print('ok')"
    import os

    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd="/", env={**os.environ, "PYTHONPATH": str(_SRC_ROOT)})
    assert result.returncode == 0 and "ok" in result.stdout, result.stderr


def test_gpu_partition_override_replaces_every_gpu_partition_but_not_the_cpu_one():
    smoke = campaign.expand_phase("phase0_smoke", gpu_partition="mygpu")
    assert {j.partition for j in smoke} == {"mygpu"}
    train = campaign.expand_phase("phase2_train_calibration", models=("schnet", "equiformerv2"), datasets=("rmd17",), gpu_partition="mygpu")
    assert {j.partition for j in train} == {"mygpu"}  # equiformerv2 normally goes to train_large, schnet to train
    assert {j.partition for j in campaign.expand_phase("phase1_export", datasets=("rmd17",), gpu_partition="mygpu")} == {campaign.PARTITION["cpu"]}
    assert {j.partition for j in campaign.expand_phase("phase0_smoke")} == {campaign.PARTITION["smoke"]}  # default unchanged


def test_submit_script_creates_the_logs_directory_the_jobs_write_to():
    text = campaign.submit_script(["phase/a.sbatch"], 1)
    assert "mkdir -p logs" in text and text.index("mkdir -p logs") < text.index("sbatch --parsable")
