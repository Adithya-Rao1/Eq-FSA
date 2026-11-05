from __future__ import annotations

import os
import subprocess

import numpy as np
import pytest

pytest.importorskip("ase")

from baselines.trainers import mace_wrapper as mw  # noqa: E402
from baselines.export.extxyz import write_extxyz  # noqa: E402


def _data(tmp_path, n_train=8, forces=True):
    g = np.random.default_rng(0)
    d = tmp_path / "data"
    for name, n in (("train", n_train), ("val", 3), ("test", 3)):
        rows = []
        for _ in range(n):
            pos = g.normal(scale=0.8, size=(3, 3))
            rows.append({"atomic_numbers": [8, 1, 1], "positions": pos, "energy": float(-10 + g.normal()),
                         "forces": g.normal(size=(3, 3)) if forces else None})
        write_extxyz(str(d / f"{name}.xyz"), rows)
    return str(d)


def test_command_has_the_exchange_keys_and_always_allows_restart(tmp_path):
    cmd = mw.build_command("D", "R", 7, device="cpu", epochs=3)
    s = " ".join(cmd)
    assert "--energy_key REF_energy" in s and "--forces_key REF_forces" in s
    assert "--restart_latest" in cmd and "--seed 7" in s and "--max_num_epochs 3" in s
    assert "--train_file D/train.xyz" in s and "--test_file D/test.xyz" in s
    assert "--E0s average" in s  # per-element referencing for ~1e4 eV totals


@pytest.mark.parametrize("max_l,expected", [(0, "16x0e"), (1, "16x0e+16x1o"), (2, "16x0e+16x1o+16x2e")])
def test_hidden_irreps_follow_channels_and_max_l(max_l, expected):
    cmd = mw.build_command("D", "R", 0, num_channels=16, max_l=max_l)
    assert cmd[cmd.index("--hidden_irreps") + 1] == expected


def test_run_is_a_noop_once_the_final_model_exists(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "mace.model").write_bytes(b"x")
    assert mw.run("D", str(run), 0) == 0  # would try to launch a real training if it did not return early


def test_dry_run_does_not_launch(tmp_path, capsys):
    assert mw.run("D", str(tmp_path / "r"), 0, dry_run=True) == 0
    assert "mace.cli.run_train" in capsys.readouterr().out


@pytest.mark.parametrize("forces", [True, False])
def test_native_trainer_trains_and_calculator_evaluates(tmp_path, forces):
    # force=false is for QM9 case
    pytest.importorskip("mace")
    from baselines.evaluation.evaluate import evaluate_file

    data, run = _data(tmp_path, forces=forces), str(tmp_path / "run")
    rc = mw.run(data, run, 0, device="cpu", epochs=2, batch_size=4, r_max=3.0, num_channels=8, max_l=1, num_interactions=2,
                dtype="float64", ema=False)
    assert rc == 0 and os.path.exists(mw.final_model_path(run))
    m = evaluate_file(mw.load_calculator(mw.final_model_path(run)), os.path.join(data, "test.xyz"), os.path.join(run, "metrics.json"))
    assert m["n_structures"] == 3 and np.isfinite(m["energy_mae_meV"])
    assert ("force_mae_meV_per_A" in m) == forces


def test_single_interaction_is_rejected_up_front():
    with pytest.raises(ValueError, match="num_interactions"):
        mw.build_command("D", "R", 0, num_interactions=1)
