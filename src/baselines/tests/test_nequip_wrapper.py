from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch

pytest.importorskip("ase")
pytest.importorskip("yaml")

from baselines.export import export_splits as es  # noqa: E402
from baselines.trainers import nequip_wrapper as nw  # noqa: E402

COMPOSITIONS = ([8, 1, 1], [6, 1, 1], [8, 6, 1])


class FakeDS:
    def __init__(self, n, forces, varying, seed):
        g = np.random.default_rng(seed)
        self.items = []
        for i in range(n):
            item = {
                "x": torch.tensor(g.normal(scale=0.8, size=(3, 3)), dtype=torch.float32),
                "atomic_numbers": torch.tensor(COMPOSITIONS[i % 3] if varying else COMPOSITIONS[0]),
                "energy": torch.tensor(float(-10 + g.normal()), dtype=torch.float64),
            }
            if forces:
                item["forces"] = torch.tensor(g.normal(size=(3, 3)), dtype=torch.float32)
            self.items.append(item)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def _export(tmp_path, monkeypatch, forces=True, varying=False):
    ds = [FakeDS(n, forces, varying, i) for i, n in enumerate((9, 3, 3))]
    monkeypatch.setattr(es, "load_splits", lambda *a: tuple(ds))
    out = str(tmp_path / "data")
    es.export_splits("x", "s", "root", out)
    return out


TINY = {
    "nequip": dict(num_layers=2, l_max=1, num_features=4, invariant_neurons=8, num_basis=4),
    "allegro": dict(
        num_layers=1, l_max=1, num_tensor_features=4, radial_basis_num_bessels_per_basis=4, two_body_latent_mlp_latent_dimensions=[8, 16],
        latent_mlp_latent_dimensions=[16], edge_eng_mlp_latent_dimensions=[8],
    ),
}


def test_config_maps_exchange_keys_and_builders(tmp_path, monkeypatch):
    d = _export(tmp_path, monkeypatch)
    n = nw.build_config("nequip", d, str(tmp_path / "r"), 3)
    assert n["dataset_key_mapping"] == {"REF_energy": "total_energy", "REF_forces": "forces"}
    assert n["chemical_symbols"] == ["H", "O"] and n["seed"] == 3 and n["append"] is True
    assert "ForceOutput" in n["model_builders"] and "StressForceOutput" not in n["model_builders"]
    assert (n["n_train"], n["n_val"]) == (9, 3)
    a = nw.build_config("allegro", d, str(tmp_path / "r"), 0)
    assert a["model_builders"][0] == "allegro.model.Allegro"


def test_energy_only_data_drops_forces_from_loss_and_mapping(tmp_path, monkeypatch):
    d = _export(tmp_path, monkeypatch, forces=False)
    c = nw.build_config("nequip", d, str(tmp_path / "r"), 0)
    assert "forces" not in c["loss_coeffs"] and "REF_forces" not in c["dataset_key_mapping"]
    assert list(c["dataset_include_keys"]) == ["REF_energy"]
    # ForceOutput stays in the model, so the force-RMS default scale would be requested; it must be overridden.
    assert c["global_rescale_scale"] == "dataset_total_energy_std"
    assert "global_rescale_scale" not in nw.build_config("nequip", _export(tmp_path / "f", monkeypatch), str(tmp_path / "r"), 0)


def test_per_species_shift_only_for_varying_composition(tmp_path, monkeypatch):
    fixed = nw.build_config("nequip", _export(tmp_path / "a", monkeypatch, varying=False), str(tmp_path / "r"), 0)
    varying = nw.build_config("nequip", _export(tmp_path / "b", monkeypatch, varying=True), str(tmp_path / "r"), 0)
    assert "per_species_rescale_shifts" not in fixed
    assert varying["per_species_rescale_shifts"] == "dataset_per_species_total_energy_mean"


def test_unknown_model_and_missing_record_fields(tmp_path):
    with pytest.raises(ValueError, match="model must be"):
        nw.build_config("mace", "D", "R", 0)
    (tmp_path / "split.json").write_text(json.dumps({"splits": {}}))
    with pytest.raises(KeyError, match="re-export"):
        nw.read_split_record(str(tmp_path))


def test_dry_run_writes_config_only(tmp_path, monkeypatch, capsys):
    d = _export(tmp_path, monkeypatch)
    assert nw.run("nequip", d, str(tmp_path / "r"), 0, dry_run=True) == 0
    assert os.path.exists(tmp_path / "r" / "config.yaml") and not os.path.exists(tmp_path / "r" / nw.TRAIN_DONE)
    assert "nequip.scripts.train" in capsys.readouterr().out


def test_finished_run_is_a_noop(tmp_path):
    (tmp_path / nw.DEPLOYED).write_bytes(b"x")
    assert nw.run("nequip", "D", str(tmp_path), 0) == 0


@pytest.mark.parametrize("model", ["nequip", "allegro"])
@pytest.mark.parametrize("forces,varying", [(True, False), (False, True)], ids=["forces_fixed", "energy_only_varying"])
def test_native_trainer_deploys_and_evaluates(tmp_path, monkeypatch, model, forces, varying):
    pytest.importorskip("nequip")
    if model == "allegro":
        pytest.importorskip("allegro")
    from baselines.evaluation.evaluate import evaluate_file

    data, run = _export(tmp_path, monkeypatch, forces=forces, varying=varying), str(tmp_path / "run")
    rc = nw.run(model, data, run, 0, device="cpu", epochs=2, batch_size=3, r_max=3.0, model_dtype="float64", overrides=TINY[model])
    assert rc == 0 and os.path.exists(nw.deployed_path(run)) and os.path.exists(os.path.join(run, nw.TRAIN_DONE))
    rec = nw.read_split_record(data)
    calc = nw.load_calculator(nw.deployed_path(run), rec["species"])
    m = evaluate_file(calc, os.path.join(data, "test.xyz"), os.path.join(run, "metrics.json"))
    assert m["n_structures"] == 3 and np.isfinite(m["energy_mae_meV"])
    assert ("force_mae_meV_per_A" in m) == forces


@pytest.mark.parametrize("model", ["nequip", "allegro"])
def test_every_config_key_is_consumed_by_the_library(tmp_path, monkeypatch, model):
    import subprocess
    import sys

    pytest.importorskip("nequip")
    if model == "allegro":
        pytest.importorskip("allegro")
    data = _export(tmp_path, monkeypatch, varying=True)

    def unused(extra):
        cfg = nw.build_config(model, data, str(tmp_path / f"r{len(extra)}"), 0, device="cpu", epochs=1, batch_size=3, r_max=3.0,
                              model_dtype="float64", overrides={**TINY[model], "warn_unused": True, **extra})
        out = subprocess.run([sys.executable, "-m", "nequip.scripts.train", nw.write_config(cfg, cfg["root"])], capture_output=True, text=True)
        assert out.returncode == 0, out.stderr[-2000:]
        return [ln for ln in (out.stdout + out.stderr).splitlines() if "were not used" in ln]

    assert unused({}) == []
    assert any("bogus_key_xyz" in ln for ln in unused({"bogus_key_xyz": 1}))
