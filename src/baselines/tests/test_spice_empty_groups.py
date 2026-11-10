# The real SPICE file (server copy, 113,986 groups) has one group, '54X VAL' in the Amino Acid Ligand subset, whose datasets are all empty (mbis_charges
# shape (0,), conformations shape (0,)), and 121 groups with no mbis_charges. Neither may crash SPICEDataset or leak into the index.
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
pytest.importorskip("torch")
pytest.importorskip("rdkit")


def _write_group(f, name, *, n_conf=2, n_atoms=3, mbis=True, empty=False):
    g = f.create_group(name)
    shape = (0,) if empty else None
    g["atomic_numbers"] = np.array([8, 1, 1][:n_atoms] if not empty else [], dtype=np.int16)
    g["smiles"] = np.array(["[O:1]([H:2])[H:3]"], dtype=h5py.string_dtype())
    g["subset"] = np.array(["SPICE Amino Acid Ligand v1.0"], dtype=h5py.string_dtype())
    g["conformations"] = np.zeros(shape) if empty else np.random.default_rng(0).normal(size=(n_conf, n_atoms, 3)).astype(np.float32)
    g["formation_energy"] = np.zeros(shape) if empty else np.linspace(-0.1, -0.2, n_conf)
    g["dft_total_gradient"] = np.zeros(shape) if empty else np.full((n_conf, n_atoms, 3), 0.01, dtype=np.float32)
    if mbis:
        g["mbis_charges"] = np.zeros(shape) if empty else np.zeros((n_conf, n_atoms, 1), dtype=np.float32)  # net charge 0 per conformation


def test_empty_and_mbis_less_groups_are_skipped_not_fatal(tmp_path):
    from stage_1.spice_dataset import SPICEDataset

    path = str(tmp_path / "spice.hdf5")
    with h5py.File(path, "w") as f:
        _write_group(f, "good a")
        _write_group(f, "54X VAL", empty=True)
        _write_group(f, "no mbis", mbis=False)
        _write_group(f, "good b", n_conf=3)
    ds = SPICEDataset(path)
    assert len(ds) == 2 + 3
    assert {ds.identity_for(i)[1] for i in range(len(ds))} == {"good a", "good b"}
