from __future__ import annotations

import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from baselines.slurm import campaign, check_outputs
from baselines.slurm.campaign import job_layout

ARGS = ["--phase", "phase1_export", "phase2_train_calibration", "--models", "schnet", "--datasets", "rmd17", "--only", "rmd17:aspirin", "--seeds", "0"]


def _write(path, text="x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _jobs(scratch):
    return [j for ph in ("phase1_export", "phase2_train_calibration")
            for j in campaign.expand_phase(ph, (0,), models=("schnet",), datasets=("rmd17",)) if j.dataset == "rmd17:aspirin"]


def _populate(scratch, metrics=None):
    export_job, train_job = _jobs(scratch)
    export_dir, _, _ = job_layout(export_job, scratch)
    for name in check_outputs.EXPORT_FILES:
        _write(Path(export_dir) / name)
    _, run_dir, _ = job_layout(train_job, scratch)
    _write(Path(run_dir) / "metrics.json", json.dumps(metrics or {"val": {"energy_mae_meV": 1.0}, "test": {"energy_mae_meV": 2.0, "force_mae_meV_per_A": 3.0}}))


def test_complete_outputs_pass(tmp_path, capsys):
    _populate(str(tmp_path))
    assert check_outputs.main(ARGS + ["--scratch-root", str(tmp_path)]) == 0
    assert "2/2 jobs complete" in capsys.readouterr().out


def test_missing_export_file_and_missing_metrics_are_reported(tmp_path, capsys):
    _populate(str(tmp_path))
    export_job, train_job = _jobs(str(tmp_path))
    (Path(job_layout(export_job, str(tmp_path))[0]) / "test.xyz").unlink()
    (Path(job_layout(train_job, str(tmp_path))[1]) / "metrics.json").unlink()
    assert check_outputs.main(ARGS + ["--scratch-root", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "missing test.xyz" in out and "missing metrics.json" in out


def test_empty_validation_file_is_flagged(tmp_path, capsys):
    _populate(str(tmp_path))
    export_job, _ = _jobs(str(tmp_path))
    (Path(job_layout(export_job, str(tmp_path))[0]) / "val.xyz").write_text("")
    assert check_outputs.main(ARGS + ["--scratch-root", str(tmp_path)]) == 1
    assert "val.xyz is empty" in capsys.readouterr().out


def test_non_finite_metrics_are_flagged(tmp_path):
    _populate(str(tmp_path), metrics={"val": {"energy_mae_meV": 1.0}, "test": {"energy_mae_meV": float("nan")}})
    assert check_outputs.main(ARGS + ["--scratch-root", str(tmp_path)]) == 1


def test_traceback_in_a_log_fails_the_check(tmp_path, capsys):
    _populate(str(tmp_path))
    _write(tmp_path / "campaign" / "logs" / "job_1.err", "Traceback (most recent call last):\n  File x\nValueError\n")
    assert check_outputs.main(ARGS + ["--scratch-root", str(tmp_path)]) == 1
    assert "LOG" in capsys.readouterr().out


def test_clean_logs_pass(tmp_path):
    _populate(str(tmp_path))
    _write(tmp_path / "campaign" / "logs" / "job_1.err", "FutureWarning: torch.load weights_only\n")  # warnings are not errors
    assert check_outputs.main(ARGS + ["--scratch-root", str(tmp_path)]) == 0
