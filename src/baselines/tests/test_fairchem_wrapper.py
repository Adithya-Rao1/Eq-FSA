from __future__ import annotations

import os
import re

import numpy as np
import pytest

pytest.importorskip("ase")
pytest.importorskip("yaml")

import baselines  # noqa: E402
from baselines.tests.test_nequip_wrapper import _export  # noqa: E402  (fake dataset -> exported splits)
from baselines.trainers import fairchem_wrapper as fw  # noqa: E402

SRC = os.path.dirname(os.path.dirname(os.path.abspath(baselines.__file__)))
TINY = dict(num_layers=1, sphere_channels=8, attn_hidden_channels=8, num_heads=2, attn_alpha_channels=4, attn_value_channels=2,
            ffn_hidden_channels=8, lmax_list=[2], mmax_list=[1], edge_channels=8, num_distance_basis=8, grid_resolution=14,
            max_neighbors=20)


def _summary(tmp_path, monkeypatch, cutoff=3.0, **kw):
    data = _export(tmp_path, monkeypatch, **kw)
    return data, fw.build_databases(data, str(tmp_path / "db"), cutoff)


def test_pad_cell_keeps_geometry_and_separates_periodic_images():
    from ase import Atoms

    g = np.random.default_rng(0)
    a = Atoms("OHH", positions=g.normal(size=(3, 3)))
    pad = 4.0
    b = fw.pad_cell(a, pad)
    assert b.pbc.all() and not a.pbc.any()  # input untouched
    np.testing.assert_allclose(b.get_all_distances(), a.get_all_distances(), atol=1e-12)  # rigid translation only
    extent = np.ptp(a.get_positions(), axis=0)
    np.testing.assert_allclose(np.diag(b.cell), extent + 2 * pad, atol=1e-12)
    # Closest approach between an atom and any periodic image of any atom is >= 2*pad, so with pad = cutoff nothing interacts with an image.
    d = b.get_all_distances(mic=True)
    images = [np.linalg.norm(b.get_positions()[:, None] - b.get_positions()[None] + np.array(s) @ b.cell.array, axis=-1)
              for s in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]]
    assert min(im.min() for im in images) >= 2 * pad - 1e-9
    assert d.shape == (3, 3)


def test_database_matches_exported_files(tmp_path, monkeypatch):
    import ase.db

    from baselines.export.extxyz import ENERGY_KEY, FORCES_KEY, read_extxyz

    data, s = _summary(tmp_path, monkeypatch)
    assert s["has_forces"] and s["counts"] == {"train": 9, "val": 3, "test": 3}
    assert s["avg_num_nodes"] == 3.0 and 0 < s["avg_degree"] <= 2.0  # 3 atoms: at most 2 neighbours each
    ref = read_extxyz(os.path.join(data, "test.xyz"))[1]
    row = list(ase.db.connect(s["paths"]["test"]).select())[1]
    atoms = row.toatoms(add_additional_information=False)
    assert atoms.pbc.all() and atoms.get_potential_energy() == ref.info[ENERGY_KEY]  # exact float64
    np.testing.assert_array_equal(atoms.get_forces(), ref.arrays[FORCES_KEY])
    assert atoms.get_atomic_numbers().tolist() == ref.get_atomic_numbers().tolist()


def test_energy_only_config_has_no_force_head_and_valid_normalizer(tmp_path, monkeypatch):
    _, s = _summary(tmp_path, monkeypatch, forces=False, varying=True)
    c = fw.build_config(s, str(tmp_path / "run"), epochs=1, batch_size=3)
    assert c["trainer"] == "energy" and list(c["model"]["heads"]) == ["energy"] and list(c["outputs"]) == ["energy"]
    assert c["dataset"]["train"]["a2g_args"]["r_forces"] is False and c["model"]["backbone"]["regress_forces"] is False
    fit = c["dataset"]["train"]["transforms"]["normalizer"]["fit"]
    assert fit["targets"] == ["energy"] and fit["override_values"] == {}  # must be a dict, not None (library bug otherwise)
    assert "element_references" in c["dataset"]["train"]["transforms"]  # composition varies


def test_forces_config_and_fixed_composition(tmp_path, monkeypatch):
    _, s = _summary(tmp_path, monkeypatch)
    c = fw.build_config(s, str(tmp_path / "run"), epochs=1, batch_size=3)
    assert c["trainer"] == "forces" and set(c["model"]["heads"]) == {"energy", "forces"}
    assert c["outputs"]["energy"]["property"] == "energy" and c["outputs"]["forces"]["property"] == "forces"
    t = c["dataset"]["train"]["transforms"]
    assert "element_references" not in t and t["normalizer"]["fit"]["override_values"] == {"forces": {"mean": 0.0}}
    assert c["model"]["backbone"]["avg_num_nodes"] == 3.0 and c["task"]["prediction_dtype"] == "float32"


def test_scheduler_milestones_respect_the_library_100_step_minimum(tmp_path, monkeypatch):
    _, s = _summary(tmp_path, monkeypatch)
    short = fw.build_config(s, str(tmp_path / "r"), epochs=2, batch_size=3)["optim"]
    assert short["lr_milestones"] == [] and short["warmup_steps"] >= 1
    s["counts"]["train"] = 10_000
    long = fw.build_config(s, str(tmp_path / "r"), epochs=5, batch_size=10)["optim"]  # 5000 steps
    assert long["lr_milestones"] == sorted(long["lr_milestones"]) and min(long["lr_milestones"]) >= 100
    assert long["lr_milestones"] == [3000, 4000, 4750] and long["warmup_steps"] == 500


def test_command_resumes_only_when_a_checkpoint_exists(tmp_path):
    run = str(tmp_path / "run")
    assert "--checkpoint" not in fw.build_command("c.yml", run, 3, "cuda", resume=False)
    cmd = fw.build_command("c.yml", run, 3, "cpu", resume=True)
    assert cmd[cmd.index("--checkpoint") + 1] == os.path.join(fw.checkpoint_dir(run), "checkpoint.pt")
    assert "--cpu" in cmd and cmd[cmd.index("--seed") + 1] == "3" and cmd[cmd.index("--timestamp-id") + 1] == fw.TIMESTAMP_ID


def test_finished_run_is_a_noop(tmp_path):
    (tmp_path / fw.TRAIN_DONE).write_bytes(b"")
    assert fw.run("D", str(tmp_path), 0) == 0  # returns before touching the (nonexistent) data dir


def test_dry_run_writes_config_and_databases_only(tmp_path, monkeypatch, capsys):
    data = _export(tmp_path, monkeypatch)
    assert fw.run(data, str(tmp_path / "run"), 0, dry_run=True, cutoff=3.0, device="cpu") == 0
    assert os.path.exists(tmp_path / "run" / "config.yml") and os.path.exists(tmp_path / "run" / "fc_data" / "train.db")
    assert not os.path.exists(tmp_path / "run" / fw.TRAIN_DONE) and "fairchem.core._cli" in capsys.readouterr().out


def _last(pattern: str, text: str) -> float:
    return float(re.findall(pattern, text)[-1])


@pytest.mark.parametrize("forces,varying", [(True, False), (False, True)], ids=["forces_fixed", "energy_only_varying"])
def test_native_trainer_trains_and_calculator_reproduces_its_validation_metrics(tmp_path, monkeypatch, capfd, forces, varying):
    pytest.importorskip("fairchem.core")
    monkeypatch.setenv("PYTHONPATH", SRC)
    from baselines.evaluation.evaluate import evaluate_calculator
    from baselines.export.extxyz import read_extxyz

    data, run = _export(tmp_path, monkeypatch, forces=forces, varying=varying), str(tmp_path / "run")
    rc = fw.run(data, run, 0, device="cpu", cutoff=3.0, epochs=2, batch_size=3, model_overrides=TINY)
    log = capfd.readouterr()
    text = log.out + log.err
    assert rc == 0 and os.path.exists(fw.best_checkpoint_path(run)) and os.path.exists(os.path.join(run, fw.TRAIN_DONE))
    m = evaluate_calculator(fw.load_calculator(run), read_extxyz(os.path.join(data, "val.xyz")))
    assert m["energy_mae_meV"] / 1000 == pytest.approx(_last(r"energy_mae: ([0-9.]+)", text), abs=5e-5)
    if forces:
        assert m["force_mae_meV_per_A"] / 1000 == pytest.approx(_last(r"forces_mae: ([0-9.]+)", text), abs=5e-5)
    else:
        assert "force_mae_meV_per_A" not in m


def test_resume_loads_the_checkpoint(tmp_path, monkeypatch, capfd):
    pytest.importorskip("fairchem.core")
    monkeypatch.setenv("PYTHONPATH", SRC)
    data, run = _export(tmp_path, monkeypatch), str(tmp_path / "run")
    kw = dict(device="cpu", cutoff=3.0, batch_size=3, model_overrides=TINY)
    assert fw.run(data, run, 0, epochs=1, **kw) == 0
    capfd.readouterr()
    os.remove(os.path.join(run, fw.TRAIN_DONE))  # simulate a requeued job that did not finish
    assert fw.run(data, run, 0, epochs=2, **kw) == 0
    out = capfd.readouterr()
    assert "Loading checkpoint from" in out.out + out.err


def test_cli_exposes_ema_decay_and_loss_coefficients_with_unchanged_defaults():
    from baselines.trainers import fairchem_wrapper as fw

    a = fw.build_parser().parse_args(["--data-dir", "d", "--run-dir", "r"])
    assert (a.ema_decay, a.energy_coeff, a.forces_coeff) == (0.999, 2.0, 100.0)  # the values build_config used prior to these flags
    b = fw.build_parser().parse_args(["--data-dir", "d", "--run-dir", "r", "--ema-decay", "0.99", "--energy-coeff", "20", "--forces-coeff", "10"])
    assert (b.ema_decay, b.energy_coeff, b.forces_coeff) == (0.99, 20.0, 10.0)


def test_conservative_config_has_energy_head_only_but_still_trains_on_forces(tmp_path, monkeypatch):
    _, s = _summary(tmp_path, monkeypatch)
    c = fw.build_config(s, str(tmp_path / "run"), epochs=1, batch_size=3, conservative=True)
    assert c["trainer"] == "forces_conservative" and list(c["model"]["heads"]) == ["energy"]
    assert set(c["outputs"]) == {"energy", "forces"} and [list(l)[0] for l in c["loss_functions"]] == ["energy", "forces"]
    assert c["dataset"]["train"]["transforms"]["normalizer"]["fit"]["targets"] == ["energy", "forces"]  # the trainer scale needs both
    assert fw.build_config(s, str(tmp_path / "run"), epochs=1, batch_size=3)["trainer"] == "forces"  # default unchanged


def test_conservative_requires_force_labels(tmp_path, monkeypatch):
    _, s = _summary(tmp_path, monkeypatch, forces=False, varying=True)
    with pytest.raises(ValueError, match="force labels"):
        fw.build_config(s, str(tmp_path / "run"), epochs=1, batch_size=3, conservative=True)


def test_conservative_flag_is_recorded_and_the_training_command_registers_the_trainer(tmp_path, monkeypatch):
    data = _export(tmp_path, monkeypatch)
    run = str(tmp_path / "run")
    assert fw.run(data, run, 0, dry_run=True, cutoff=3.0, epochs=1, batch_size=3, conservative=True) == 0
    import json
    assert json.load(open(os.path.join(run, "wrapper.json")))["conservative"] is True
    assert "baselines.trainers.fairchem_conservative" in " ".join(fw.build_command("c.yml", run, 0, "cpu", False))
    assert fw.build_parser().parse_args(["--data-dir", "d", "--run-dir", "r", "--conservative"]).conservative is True


def test_conservative_model_forces_are_the_energy_gradient(tmp_path, monkeypatch, capfd):
    pytest.importorskip("fairchem.core")
    monkeypatch.setenv("PYTHONPATH", SRC)
    from baselines.export.extxyz import read_extxyz

    data, run = _export(tmp_path, monkeypatch), str(tmp_path / "run")
    assert fw.run(data, run, 0, device="cpu", cutoff=3.0, epochs=2, batch_size=3, model_overrides=TINY, conservative=True) == 0
    capfd.readouterr()
    import torch

    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        calc = fw.load_calculator(run)
        model_dtypes = sorted({str(p.dtype) for p in calc.inner.trainer.model.parameters()})
        atoms = read_extxyz(os.path.join(data, "val.xyz"))[0]
        atoms.calc = calc

        # making energy deterministic by reseeding each rotation frame prior to each call
        def seeded(fn):
            torch.manual_seed(0)
            return fn()

        def fresh_energy():
            calc.reset()  # ASE caches results for unchanged atoms; force a new forward pass
            return atoms.get_potential_energy()

        unseeded_spread = float(np.ptp([fresh_energy() for _ in range(4)]))
        calc.reset()  # the unseeded calls above cached their forces
        f = seeded(atoms.get_forces)

        def central_difference(h):
            fd = np.zeros_like(f)
            for i in range(len(atoms)):
                for k in range(3):
                    e = []
                    for sgn in (+1, -1):
                        b = atoms.copy()
                        p = b.get_positions()
                        p[i, k] += sgn * h
                        b.set_positions(p)
                        b.calc = calc
                        e.append(seeded(b.get_potential_energy))
                    fd[i, k] = -(e[0] - e[1]) / (2 * h)
            return fd

        # reported for diagnosis --> truncation error shrinks like h^2, rounding noise grows like 1/h
        errors = {h: float(np.abs(f - central_difference(h)).max()) for h in (1e-2, 1e-3, 1e-4)}
        fd = central_difference(1e-4)
    finally:
        torch.set_default_dtype(prev)
    np.testing.assert_allclose(f, fd, atol=1e-7, rtol=0, err_msg=f"max abs diff by step {errors}; energy spread over repeated unseeded calls {unseeded_spread:.3e} eV; model parameter dtypes {model_dtypes}")
