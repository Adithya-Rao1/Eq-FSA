from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("ase")

from ase.calculators.calculator import Calculator, all_changes  # noqa: E402

from baselines.evaluation import evaluate as ev  # noqa: E402
from baselines.export.extxyz import ENERGY_KEY, FORCES_KEY, read_extxyz, write_extxyz  # noqa: E402


class OffsetCalc(Calculator):
    # Returns reference energy + de_eV and reference forces + df_eV_A, looked up by the atoms' first coordinate.
    implemented_properties = ["energy", "forces"]

    def __init__(self, truth, de=0.0, df=0.0, **kw):
        super().__init__(**kw)
        self.truth, self.de, self.df = truth, de, df

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        e, f = self.truth[round(float(atoms.positions[0, 0]), 6)]
        self.results = {"energy": e + self.de, "forces": f + self.df}


def _frames(tmp_path, n=5, forces=True):
    g = np.random.default_rng(0)
    rows, truth = [], {}
    for i in range(n):
        pos = g.normal(size=(3, 3))
        pos[0, 0] = i + 10.0  # unique key
        e, f = -1000.0 - i, g.normal(size=(3, 3))
        truth[round(pos[0, 0], 6)] = (e, f)
        rows.append({"atomic_numbers": [6, 1, 8], "positions": pos, "energy": e, "forces": f if forces else None})
    p = str(tmp_path / "t.xyz")
    write_extxyz(p, rows)
    return p, read_extxyz(p), truth


def test_perfect_calculator_scores_zero(tmp_path):
    _, frames, truth = _frames(tmp_path)
    m = ev.evaluate_calculator(OffsetCalc(truth), frames)
    assert m["energy_mae_meV"] == pytest.approx(0, abs=1e-6)
    assert m["force_mae_meV_per_A"] == pytest.approx(0, abs=1e-6)  # forces are stored with 10 decimals -> <= 5e-11 eV/A = 5e-8 meV/A


def test_constant_energy_offset_gives_exact_meV_and_per_atom(tmp_path):
    _, frames, truth = _frames(tmp_path)
    m = ev.evaluate_calculator(OffsetCalc(truth, de=0.002), frames)  # +2 meV on 3 atoms
    assert m["energy_mae_meV"] == pytest.approx(2.0, abs=1e-6)
    assert m["energy_rmse_meV"] == pytest.approx(2.0, abs=1e-6)
    assert m["energy_mae_meV_per_atom"] == pytest.approx(2.0 / 3, abs=1e-6)


def test_constant_force_offset_gives_exact_meV_per_A(tmp_path):
    _, frames, truth = _frames(tmp_path)
    m = ev.evaluate_calculator(OffsetCalc(truth, df=0.05), frames)
    assert m["force_mae_meV_per_A"] == pytest.approx(50.0, abs=1e-5)
    assert m["force_rmse_meV_per_A"] == pytest.approx(50.0, abs=1e-5)
    assert m["n_force_components"] == 5 * 9


def test_rmse_exceeds_mae_for_uneven_errors():
    from ase import Atoms

    frames = []
    for e in (0.0, 0.0):
        a = Atoms("H", positions=[[0, 0, 0]])
        a.info[ENERGY_KEY] = e
        frames.append(a)
    m = ev.score(frames, np.array([0.001, 0.003]), None)
    assert m["energy_mae_meV"] == pytest.approx(2.0)
    assert m["energy_rmse_meV"] == pytest.approx(np.sqrt(5.0))


def test_relative_energy_removes_constant_offset_only(tmp_path):
    _, frames, truth = _frames(tmp_path)
    m = ev.evaluate_calculator(OffsetCalc(truth, de=7.0), frames, relative=True)  # a 7 eV zero shift, like a classical force field
    assert m["energy_mae_meV"] == pytest.approx(7000.0, abs=1e-6)
    assert m["energy_relative_mae_meV"] == pytest.approx(0, abs=1e-6)
    assert m["energy_offset_meV"] == pytest.approx(7000.0, abs=1e-6)


def test_relative_energy_still_sees_non_constant_error():
    from ase import Atoms

    frames = []
    for _ in range(2):
        a = Atoms("H", positions=[[0, 0, 0]])
        a.info[ENERGY_KEY] = 0.0
        frames.append(a)
    m = ev.score(frames, np.array([5.0, 5.002]), None, relative=True)
    assert m["energy_relative_mae_meV"] == pytest.approx(1.0, abs=1e-6)


def _grouped_frames(shifts):
    from ase import Atoms

    frames, pred = [], []
    for label, shift, errs in shifts:
        for e in errs:
            a = Atoms("H", positions=[[0, 0, 0]])
            a.info[ENERGY_KEY] = 0.0
            a.info["smiles"] = label
            frames.append(a)
            pred.append(shift + e)
    return frames, np.array(pred)


def test_group_key_removes_one_offset_per_group():
    # Two molecules with different zeros (+5 eV, -3 eV) and +-1 meV scatter around each.
    frames, pred = _grouped_frames([("a", 5.0, (-0.001, 0.001, -0.001, 0.001)), ("b", -3.0, (0.001, -0.001))])
    m = ev.score(frames, pred, None, relative=True, group_key="smiles")
    assert m["energy_relative_mae_meV"] == pytest.approx(1.0, abs=1e-9)
    assert m["n_relative_groups"] == 2 and m["n_relative_structures"] == 6
    pooled = ev.score(frames, pred, None, relative=True)
    assert pooled["energy_relative_mae_meV"] > 1000.0  # one pooled offset cannot absorb +5 and -3 eV


def test_singleton_groups_are_excluded_from_relative_stats():
    frames, pred = _grouped_frames([("a", 5.0, (-0.002, 0.002)), ("lonely", 9.0, (0.0,))])
    m = ev.score(frames, pred, None, relative=True, group_key="smiles")
    assert m["n_relative_groups"] == 1 and m["n_relative_structures"] == 2
    assert m["energy_relative_mae_meV"] == pytest.approx(2.0, abs=1e-9)
    assert m["n_structures"] == 3  # the absolute metrics still cover every frame


def test_all_singleton_groups_raise():
    frames, pred = _grouped_frames([("a", 1.0, (0.0,)), ("b", 2.0, (0.0,))])
    with pytest.raises(ValueError, match="two or more"):
        ev.score(frames, pred, None, relative=True, group_key="smiles")


def test_no_reference_forces_skips_force_metrics(tmp_path):
    _, frames, truth = _frames(tmp_path, forces=False)
    assert FORCES_KEY not in frames[0].arrays
    m = ev.evaluate_calculator(OffsetCalc(truth), frames)
    assert "force_mae_meV_per_A" not in m and "energy_mae_meV" in m


def test_input_frames_are_not_mutated(tmp_path):
    _, frames, truth = _frames(tmp_path)
    ev.evaluate_calculator(OffsetCalc(truth, de=1.0), frames)
    assert all(a.calc is None for a in frames)
    assert frames[0].info[ENERGY_KEY] == truth[round(frames[0].positions[0, 0], 6)][0]


def test_empty_test_set_raises():
    with pytest.raises(ValueError, match="empty"):
        ev.score([], np.array([]), None)


def test_evaluate_file_writes_metrics_json(tmp_path):
    p, _, truth = _frames(tmp_path)
    out = tmp_path / "sub" / "metrics.json"
    m = ev.evaluate_file(OffsetCalc(truth, de=0.001), p, str(out), extra={"model": "toy", "seed": 3})
    on_disk = json.loads(out.read_text())
    assert on_disk == m and on_disk["model"] == "toy" and on_disk["seed"] == 3


def test_cli_loads_calculator_by_module_path(tmp_path, monkeypatch, capsys):
    p, _, _ = _frames(tmp_path)
    (tmp_path / "toycalc.py").write_text(
        "from ase.calculators.calculator import Calculator, all_changes\n"
        "import numpy as np\n"
        "class C(Calculator):\n"
        "    implemented_properties=['energy','forces']\n"
        "    def calculate(self, atoms=None, properties=('energy',), system_changes=all_changes):\n"
        "        super().calculate(atoms, properties, system_changes)\n"
        "        self.results={'energy':0.0,'forces':np.zeros((len(atoms),3))}\n"
        "def make(): return C()\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    out = tmp_path / "m.json"
    assert ev.main(["--test-file", p, "--out", str(out), "--calculator", "toycalc:make"]) == 0
    assert json.loads(out.read_text())["n_structures"] == 5
    with pytest.raises(SystemExit):
        ev.main(["--test-file", p, "--out", str(out), "--calculator", "nofunction"])
