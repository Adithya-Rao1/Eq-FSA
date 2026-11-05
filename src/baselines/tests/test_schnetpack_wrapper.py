from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("ase")
pytest.importorskip("yaml")

import baselines  # noqa: E402
from baselines.tests.test_nequip_wrapper import _export  # noqa: E402  (fake dataset -> exported splits)
from baselines.trainers import schnetpack_wrapper as sw  # noqa: E402
from baselines.trainers.spk_launch import ensure_schnetpack_importable  # noqa: E402

SRC = os.path.dirname(os.path.dirname(os.path.abspath(baselines.__file__)))
TINY = {"model.representation.n_atom_basis": 8, "model.representation.n_interactions": 1, "model.representation.radial_basis.n_rbf": 6}


def _need_schnetpack():
    ensure_schnetpack_importable()  # importorskip itself imports schnetpack, so the shim has to come first
    pytest.importorskip("schnetpack")


def _summary(tmp_path, monkeypatch, **kw):
    _need_schnetpack()
    data = _export(tmp_path, monkeypatch, **kw)
    return data, sw.build_database(data, str(tmp_path / "db"))


def test_fit_atomrefs_recovers_exact_per_element_energies():
    from ase import Atoms

    from baselines.export.extxyz import ENERGY_KEY

    truth = {1: -0.5, 6: -37.8, 8: -75.0}
    frames = []
    for numbers in ([8, 1, 1], [6, 1, 1], [8, 6, 1], [6, 6, 8, 1], [8, 8, 1, 1]):
        a = Atoms(numbers=numbers, positions=np.zeros((len(numbers), 3)))
        a.info[ENERGY_KEY] = sum(truth[z] for z in numbers)
        frames.append(a)
    refs = sw.fit_atomrefs(frames)
    for z, v in truth.items():
        assert refs[z] == pytest.approx(v, abs=1e-9)
    assert refs[7] == 0.0 and len(refs) == 100


def test_database_matches_exported_files_and_split(tmp_path, monkeypatch):
    _need_schnetpack()
    from schnetpack.data import ASEAtomsData

    from baselines.export.extxyz import ENERGY_KEY, FORCES_KEY, read_extxyz

    data, s = _summary(tmp_path, monkeypatch)
    assert (s["n_train"], s["n_val"], s["n_test"], s["has_forces"]) == (9, 3, 3, True)
    sp = np.load(s["split"])
    assert sp["train_idx"].tolist() == list(range(9)) and sp["val_idx"].tolist() == [9, 10, 11] and sp["test_idx"].tolist() == [12, 13, 14]
    db = ASEAtomsData(s["db"])
    assert len(db) == 15
    ref = read_extxyz(os.path.join(data, "test.xyz"))[1]  # index 13
    item = db[13]
    assert float(item["energy"]) == ref.info[ENERGY_KEY]  # exact float64
    np.testing.assert_array_equal(item["forces"].numpy(), ref.arrays[FORCES_KEY])
    assert item["_atomic_numbers"].tolist() == ref.get_atomic_numbers().tolist()


def test_atomrefs_only_stored_for_varying_composition(tmp_path, monkeypatch):
    _need_schnetpack()
    from schnetpack.data import ASEAtomsData

    _, fixed = _summary(tmp_path / "a", monkeypatch, varying=False)
    _, varying = _summary(tmp_path / "b", monkeypatch, varying=True)
    assert fixed["fixed_composition"] and not varying["fixed_composition"]
    assert "energy" not in ASEAtomsData(fixed["db"]).atomrefs
    assert bool(ASEAtomsData(varying["db"]).atomrefs["energy"].any())


def test_energy_only_experiment_has_no_force_head(tmp_path, monkeypatch):
    _, s = _summary(tmp_path, monkeypatch, forces=False, varying=True)
    assert s["has_forces"] is False
    e = sw.build_experiment(s, str(tmp_path / "run"), 4)
    assert [o["name"] for o in e["task"]["outputs"]] == ["energy"] and e["task"]["outputs"][0]["loss_weight"] == 1.0
    assert all("Forces" not in m["_target_"] for m in e["model"]["output_modules"])
    assert e["data"]["transforms"][1]["remove_atomrefs"] is True and e["model"]["postprocessors"][1]["add_atomrefs"] is True
    assert e["seed"] == 4 and e["run"]["id"] == "spk" and os.path.isabs(e["run"]["path"])


def test_forces_experiment_and_fixed_composition(tmp_path, monkeypatch):
    _, s = _summary(tmp_path, monkeypatch)
    e = sw.build_experiment(s, str(tmp_path / "run"), 0, overrides={"trainer.max_epochs": 7, "model.representation.n_atom_basis": 8})
    assert [o["name"] for o in e["task"]["outputs"]] == ["energy", "forces"]
    assert "remove_atomrefs" not in e["data"]["transforms"][1]
    assert e["trainer"]["max_epochs"] == 7 and e["model"]["representation"]["n_atom_basis"] == 8
    assert e["data"]["property_units"] == {"energy": "eV", "forces": "eV/Ang"}


def test_unknown_model_and_finished_run(tmp_path):
    with pytest.raises(ValueError, match="model must be"):
        sw.build_command("cfg", "mace")
    os.makedirs(tmp_path / "spk")
    (tmp_path / "spk" / "best_model").write_bytes(b"x")
    assert sw.run("schnet", "D", str(tmp_path), 0) == 0  # returns before touching the (nonexistent) data dir


def test_shim_restores_T_co_when_torch_renamed_it(monkeypatch):
    import torch.utils.data.dataloader as dl

    monkeypatch.delattr(dl, "T_co", raising=False)
    ensure_schnetpack_importable()
    assert dl.T_co is dl._T_co
    ensure_schnetpack_importable()  


@pytest.mark.parametrize("model", ["schnet", "painn"])
@pytest.mark.parametrize("forces,varying", [(True, False), (False, True)], ids=["forces_fixed", "energy_only_varying"])
def test_native_trainer_trains_and_calculator_evaluates(tmp_path, monkeypatch, model, forces, varying):
    _need_schnetpack()
    monkeypatch.setenv("PYTHONPATH", SRC)
    from baselines.evaluation.evaluate import evaluate_file

    data, run = _export(tmp_path, monkeypatch, forces=forces, varying=varying), str(tmp_path / "run")
    rc = sw.run(model, data, run, 0, device="cpu", epochs=2, batch_size=3, cutoff=3.0, overrides=TINY)
    assert rc == 0 and os.path.exists(sw.best_model_path(run))
    m = evaluate_file(sw.load_calculator(run), os.path.join(data, "test.xyz"), os.path.join(run, "metrics.json"))
    assert m["n_structures"] == 3 and np.isfinite(m["energy_mae_meV"])
    assert ("force_mae_meV_per_A" in m) == forces
