from __future__ import annotations

import json

import numpy as np
import pytest
import torch

pytest.importorskip("ase")

from baselines.export import export_splits as es  # noqa: E402
from baselines.export.extxyz import ENERGY_KEY, FORCES_KEY, format_frame, read_extxyz, write_extxyz  # noqa: E402


class FakeDataset:
    def __init__(self, n, with_forces=True, seed=0, offset=0.0):
        g = np.random.default_rng(seed)
        self.items = []
        for i in range(n):
            item = {
                "x": torch.tensor(g.normal(size=(3, 3)), dtype=torch.float32),
                "atomic_numbers": torch.tensor([6, 1, 8]),
                # ~1e4 eV total with sub-meV digits: a float32 or fixed-point round trip would lose them.
                "energy": torch.tensor(-12345.678901234567 + i + offset, dtype=torch.float64),
            }
            if with_forces:
                item["forces"] = torch.tensor(g.normal(size=(3, 3)), dtype=torch.float32)
            self.items.append(item)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def _patch(monkeypatch, **sizes):
    ds = {k: FakeDataset(n, with_forces=sizes.get("forces", True), seed=i) for i, (k, n) in enumerate(
        (("train", sizes.get("train", 10)), ("val", sizes.get("val", 4)), ("test", sizes.get("test", 6))))}
    monkeypatch.setattr(es, "load_splits", lambda dataset, split, root: (ds["train"], ds["val"], ds["test"]))
    return ds


def test_energy_round_trips_exactly_in_float64(tmp_path):
    e = -12345.678901234567
    path = tmp_path / "a.xyz"
    write_extxyz(str(path), [{"atomic_numbers": [6, 1], "positions": np.zeros((2, 3)), "energy": e}])
    atoms = read_extxyz(str(path))[0]
    assert atoms.info[ENERGY_KEY] == e  # exact, not approximate


def test_forces_and_positions_round_trip(tmp_path):
    g = np.random.default_rng(1)
    pos, frc = g.normal(size=(4, 3)), g.normal(size=(4, 3))
    path = tmp_path / "a.xyz"
    write_extxyz(str(path), [{"atomic_numbers": [6, 1, 1, 8], "positions": pos, "energy": 1.0, "forces": frc}])
    atoms = read_extxyz(str(path))[0]
    assert atoms.get_chemical_symbols() == ["C", "H", "H", "O"]
    # Positions and forces are written with 10 decimals, so the round-trip error is bounded by 5e-11.
    np.testing.assert_allclose(atoms.get_positions(), pos, atol=5e-11, rtol=0)
    np.testing.assert_allclose(atoms.arrays[FORCES_KEY], frc, atol=5e-11, rtol=0)
    assert not atoms.pbc.any()


def test_forces_are_optional_for_qm9_style_frames(tmp_path):
    path = tmp_path / "a.xyz"
    write_extxyz(str(path), [{"atomic_numbers": [6, 1], "positions": np.zeros((2, 3)), "energy": 2.5}])
    atoms = read_extxyz(str(path))[0]
    assert FORCES_KEY not in atoms.arrays


def test_extra_tags_are_written_to_the_header():
    text = format_frame([1], np.zeros((1, 3)), 0.0, tags={"idx": "7"})
    assert "idx=7" in text.splitlines()[1]


@pytest.mark.parametrize("bad", [0, 119, -1])
def test_atomic_number_outside_periodic_table_raises(bad):
    with pytest.raises(ValueError, match="periodic table"):
        format_frame([bad], np.zeros((1, 3)), 0.0)


def test_shape_mismatches_raise():
    with pytest.raises(ValueError, match="positions"):
        format_frame([1, 1], np.zeros((1, 3)), 0.0)
    with pytest.raises(ValueError, match="forces"):
        format_frame([1], np.zeros((1, 3)), 0.0, forces=np.zeros((2, 3)))


def test_export_writes_files_and_split_json(tmp_path, monkeypatch):
    ds = _patch(monkeypatch)
    rec = es.export_splits("rmd17:aspirin", "s", "root", str(tmp_path))
    on_disk = json.loads((tmp_path / "split.json").read_text())
    assert on_disk == rec
    assert on_disk["keys"] == {"energy": ENERGY_KEY, "forces": FORCES_KEY}
    assert on_disk["units"]["energy"] == "eV"
    for name, n in (("train", 10), ("val", 4), ("test", 6)):
        info = on_disk["splits"][name]
        assert (info["n"], info["n_source"], info["indices"], info["has_forces"]) == (n, n, "all", True)
        frames = read_extxyz(str(tmp_path / info["file"]))
        assert len(frames) == n
        assert frames[0].info[ENERGY_KEY] == float(ds[name][0]["energy"])


def test_sha256_matches_file_contents(tmp_path, monkeypatch):
    import hashlib

    _patch(monkeypatch)
    rec = es.export_splits("qm9", "s", "root", str(tmp_path))
    for info in rec["splits"].values():
        assert info["sha256"] == hashlib.sha256((tmp_path / info["file"]).read_bytes()).hexdigest()


def test_qm9_style_dataset_has_no_forces(tmp_path, monkeypatch):
    _patch(monkeypatch, forces=False)
    rec = es.export_splits("qm9", "s", "root", str(tmp_path))
    assert all(not info["has_forces"] for info in rec["splits"].values())
    assert FORCES_KEY not in read_extxyz(str(tmp_path / "train.xyz"))[0].arrays


def test_subsample_is_seeded_sorted_and_capped():
    a = es._subsample(100, 10, seed=3)
    assert a == es._subsample(100, 10, seed=3)
    assert a != es._subsample(100, 10, seed=4)
    assert a == sorted(a) and len(set(a)) == 10 and all(0 <= i < 100 for i in a)
    assert es._subsample(5, None, 0) == [0, 1, 2, 3, 4]
    assert es._subsample(5, 50, 0) == [0, 1, 2, 3, 4]


def test_caps_keep_the_same_indices_for_every_caller(tmp_path, monkeypatch):
    # The ANI-1x requirement: every model (and EqFSA) trains on the identical subset.
    ds = _patch(monkeypatch, train=50)
    r1 = es.export_splits("ani1x", "s", "root", str(tmp_path / "a"), max_train=8, subset_seed=5)
    r2 = es.export_splits("ani1x", "s", "root", str(tmp_path / "b"), max_train=8, subset_seed=5)
    assert r1["splits"]["train"]["indices"] == r2["splits"]["train"]["indices"]
    assert r1["splits"]["train"]["sha256"] == r2["splits"]["train"]["sha256"]
    info = r1["splits"]["train"]
    assert info["n"] == 8 and info["n_source"] == 50 and info["max_n"] == 8
    frames = read_extxyz(str(tmp_path / "a" / "train.xyz"))
    assert [f.info[ENERGY_KEY] for f in frames] == [float(ds["train"][i]["energy"]) for i in info["indices"]]
    # Uncapped splits are unaffected.
    assert r1["splits"]["val"]["indices"] == "all"


def test_cli_main_prints_and_writes(tmp_path, monkeypatch, capsys):
    _patch(monkeypatch)
    rc = es.main(["--dataset", "qm9", "--split", "s", "--data-root", "root", "--out-dir", str(tmp_path), "--max-train", "3"])
    assert rc == 0
    assert "train: 3 of 10" in capsys.readouterr().out


def test_split_json_records_species_and_fixed_composition(tmp_path, monkeypatch):
    _patch(monkeypatch)  # every fake frame is O, H, H... atomic numbers [6, 1, 8]
    rec = es.export_splits("rmd17:aspirin", "s", "root", str(tmp_path))
    assert rec["atomic_numbers"] == [1, 6, 8] and rec["species"] == ["H", "C", "O"]
    assert rec["fixed_composition"] is True


def test_varying_composition_is_detected(tmp_path, monkeypatch):
    ds = _patch(monkeypatch)
    ds["val"].items[0]["atomic_numbers"] = torch.tensor([6, 1, 7])
    rec = es.export_splits("qm9", "s", "root", str(tmp_path))
    assert rec["fixed_composition"] is False and rec["species"] == ["H", "C", "N", "O"]
