# Tests for src/baselines/slurm/run_local.py
from __future__ import annotations

import json
import stat
import subprocess
import sys
from pathlib import Path

_REPO_SRC = Path(__file__).resolve().parents[2]
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

import pytest

from baselines.slurm import campaign, run_local

FAKE = """#!/bin/bash
echo "$@" >> "$FAKE_LOG"
args=("$@")
for i in "${!args[@]}"; do
  case "${args[$i]}" in
    --out-dir) mkdir -p "${args[$((i+1))]}"; touch "${args[$((i+1))]}/split.json";;
    --out) mkdir -p "$(dirname "${args[$((i+1))]}")"; echo '{}' > "${args[$((i+1))]}";;
  esac
done
exit ${FAKE_EXIT:-0}
"""


@pytest.fixture
def env(tmp_path, monkeypatch):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    for image in ("mlip-mpnn", "mlip-legacy", "mlip-classical"):
        (scratch / f"{image}.sif").write_text("")
    fake = tmp_path / "fakebin"
    fake.write_text(FAKE)
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "fake.log"
    monkeypatch.setenv("FAKE_LOG", str(log))
    monkeypatch.delenv("FAKE_EXIT", raising=False)
    return scratch, str(fake), log


def argv(scratch, fake, *extra):
    return ["--scratch-root", str(scratch), "--container-bin", fake, "--cpus", "2", *extra]


PILOT = ["--phase", "phase1_export", "phase2_train_calibration", "--models", "schnet", "--datasets", "rmd17", "--only", "rmd17:aspirin", "--pilot"]


def calls(log: Path) -> list[str]:
    return log.read_text().splitlines() if log.exists() else []


def test_scripts_are_valid_bash_and_use_nv_only_for_gpu_jobs(env):
    scratch, fake, _ = env
    for job in campaign.expand_phase("phase1_export", datasets=("rmd17",))[:1] + campaign.expand_phase("phase2_train_calibration", models=("schnet",), datasets=("rmd17",), pilot=True):
        text = run_local.local_script(job, str(scratch), str(scratch / "Eq-FSA"), 1, "", "apptainer", 4)
        path = scratch / f"{job.run_name}.sh"
        path.write_text(text)
        assert subprocess.run(["bash", "-n", str(path)]).returncode == 0
        assert "srun" not in text and "SLURM" not in text
        assert ("--nv" in text) == (job.job_type == "train")
        assert f"{scratch}:{scratch}" in text and ":/repo:ro" in text and "APPTAINERENV_PYTHONPATH=/repo/src" in text


def test_pilot_runs_export_then_train_then_eval_and_records_results(env, tmp_path):
    scratch, fake, log = env
    assert run_local.main(argv(scratch, fake, *PILOT)) == 0
    rows = [json.loads(line) for line in (scratch / "campaign/local/results.jsonl").read_text().splitlines()]
    assert [r["status"] for r in rows] == ["done", "done"] and [r["job_type"] for r in rows] == ["export", "train"]
    assert (scratch / "exports/rmd17-aspirin_rmd17_split_0_val50/split.json").exists()
    assert list((scratch / "runs/phase2_train_calibration").glob("*/metrics.json"))
    text = " ".join(calls(log))
    assert "baselines.export.export_splits" in text and "baselines.trainers.schnetpack_wrapper" in text and "baselines.evaluation.run_eval" in text
    assert text.index("export_splits") < text.index("schnetpack_wrapper") < text.index("run_eval")
    assert "rmd17_split_0_val50" in text and "--nv" in text


def test_rerun_skips_finished_jobs_without_calling_the_container(env):
    scratch, fake, log = env
    assert run_local.main(argv(scratch, fake, *PILOT)) == 0
    before = len(calls(log))
    assert run_local.main(argv(scratch, fake, *PILOT)) == 0
    assert len(calls(log)) == before
    rows = [json.loads(line) for line in (scratch / "campaign/local/results.jsonl").read_text().splitlines()]
    assert [r["status"] for r in rows[-2:]] == ["skipped", "skipped"]


def test_a_failing_job_is_reported_and_the_runner_keeps_going_by_default(env, monkeypatch, capsys):
    scratch, fake, log = env
    monkeypatch.setenv("FAKE_EXIT", "3")
    two_exports = ["--phase", "phase1_export", "--datasets", "rmd17", "--only", "rmd17:aspirin", "rmd17:ethanol"]
    assert run_local.main(argv(scratch, fake, *two_exports)) == 1
    assert len(calls(log)) == 2  # both attempted
    assert "FAILED" in capsys.readouterr().err


def test_fail_fast_stops_at_the_first_failure(env, monkeypatch):
    scratch, fake, log = env
    monkeypatch.setenv("FAKE_EXIT", "3")
    two_exports = ["--phase", "phase1_export", "--datasets", "rmd17", "--only", "rmd17:aspirin", "rmd17:ethanol", "--fail-fast"]
    assert run_local.main(argv(scratch, fake, *two_exports)) == 1
    assert len(calls(log)) == 1


def test_dry_run_writes_scripts_but_runs_nothing(env):
    scratch, fake, log = env
    assert run_local.main(argv(scratch, fake, *PILOT, "--dry-run")) == 0
    assert not log.exists() and not (scratch / "campaign/local/results.jsonl").exists()
    assert list((scratch / "campaign/local").rglob("*.sh"))


def test_missing_image_is_reported_before_anything_runs(env):
    scratch, fake, log = env
    (scratch / "mlip-legacy.sif").unlink()
    assert run_local.main(argv(scratch, fake, *PILOT)) == 2
    assert not log.exists()


def test_only_filter_keeps_exactly_the_named_datasets():
    args = run_local.build_parser().parse_args(["--phase", "phase1_export", "--scratch-root", "/s", "--datasets", "rmd17", "--only", "rmd17:aspirin"])
    jobs = run_local.select_jobs(["phase1_export"], args)
    assert [j.dataset for j in jobs] == ["rmd17:aspirin"]


def test_job_commands_are_what_the_sbatch_script_contains():
    job = campaign.expand_phase("phase2_train_calibration", models=("mace",), datasets=("qm9",), pilot=True)[0]
    sbatch = campaign.render_job(job, "/s", "/s/Eq-FSA", 3, "", None)
    for line in campaign.job_commands(job, "/s", 3, ""):
        assert line in sbatch


def test_runner_and_campaign_import_without_torch():
    code = "import sys; sys.modules['torch'] = None; from baselines.slurm import run_local, campaign"
    result = subprocess.run([sys.executable, "-c", code], cwd=_REPO_SRC, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
