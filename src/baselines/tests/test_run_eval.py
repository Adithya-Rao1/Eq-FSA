from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("ase")

from ase.calculators.calculator import Calculator, all_changes  # noqa: E402

from baselines.evaluation import run_eval  # noqa: E402
from baselines.export.extxyz import write_extxyz  # noqa: E402


class Zero(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {"energy": 0.0, "forces": np.zeros((len(atoms), 3))}


def _export(tmp_path, n_val=2):
    g = np.random.default_rng(0)
    rec = {"dataset": "rmd17:x", "split": "s", "species": ["C", "H"], "splits": {}}
    for name, n in (("train", 3), ("val", n_val), ("test", 4)):
        rows = [{"atomic_numbers": [6, 1], "positions": g.normal(size=(2, 3)), "energy": -1.0, "forces": g.normal(size=(2, 3))} for _ in range(n)]
        write_extxyz(str(tmp_path / f"{name}.xyz"), rows) if n else (tmp_path / f"{name}.xyz").write_text("")
        rec["splits"][name] = {"file": f"{name}.xyz", "n": n}
    (tmp_path / "split.json").write_text(json.dumps(rec))


def test_evaluates_val_and_test_and_writes_metrics(tmp_path, monkeypatch):
    _export(tmp_path)
    seen = {}
    monkeypatch.setattr(run_eval, "build_calculator", lambda model, record, run_dir, device: seen.update(model=model, species=record["species"]) or Zero())
    out = tmp_path / "run" / "metrics.json"
    result = run_eval.evaluate_run("mace", str(tmp_path), str(tmp_path / "run"), "cpu", seed=3, out_path=str(out))
    assert seen == {"model": "mace", "species": ["C", "H"]}
    assert result["val"]["n_structures"] == 2 and result["test"]["n_structures"] == 4
    assert result["test"]["energy_mae_meV"] == pytest.approx(1000.0, abs=1e-6)  # prediction 0 vs reference -1 eV
    assert json.loads(out.read_text()) == result and result["seed"] == 3 and result["dataset"] == "rmd17:x"
    assert not list(out.parent.glob("*.tmp"))


def test_empty_validation_split_is_recorded_as_none_not_a_crash(tmp_path, monkeypatch):
    _export(tmp_path, n_val=0)  # SPICE has no validation set
    monkeypatch.setattr(run_eval, "build_calculator", lambda *a: Zero())
    result = run_eval.evaluate_run("mace", str(tmp_path), str(tmp_path / "run"), "cpu")
    assert result["val"] is None and result["test"]["n_structures"] == 4


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="model must be"):
        run_eval.build_calculator("equiformer", {}, "/r", "cpu")


def test_cli_prints_mae_summary(tmp_path, monkeypatch, capsys):
    _export(tmp_path)
    monkeypatch.setattr(run_eval, "build_calculator", lambda *a: Zero())
    assert run_eval.main(["--model", "mace", "--data-dir", str(tmp_path), "--run-dir", str(tmp_path / "r"), "--out", str(tmp_path / "r" / "m.json"), "--device", "cpu"]) == 0
    assert "energy_mae_meV" in capsys.readouterr().out
