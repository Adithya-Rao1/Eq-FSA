from __future__ import annotations

import csv
import os
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from rdkit import Chem

from utils.mappings import ATOM_TYPE_MAP
from utils.calc_props import calc_rdkit_mol_props_smiles
from utils.pw_bases import compute_molecular_box
from stage_1.spice_dataset import build_rdkit_edge_features

KCAL_PER_MOL_TO_EV = 0.0433641

_PERIODIC_TABLE = Chem.GetPeriodicTable()

MOLECULE_SMILES: Dict[str, str] = {
    "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "ethanol": "CCO",
    "malonaldehyde": "O=CCC=O",
    "benzene": "c1ccccc1",
    "toluene": "Cc1ccccc1",
    "naphthalene": "c1ccc2ccccc2c1",
    "salicylic acid": "O=C(O)c1ccccc1O",
    "uracil": "O=c1cc[nH]c(=O)[nH]1",
    "paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "azobenzene": "c1ccc(/N=N/c2ccccc2)cc1",
}


_RMD17_URL_FIXED = "https://archive.materialscloud.org/api/records/pfffs-fff86/files/rmd17.tar.bz2/content"


def get_pyg_md17_class():
    from torch_geometric.datasets import MD17 as PyGMD17

    if PyGMD17.revised_url != _RMD17_URL_FIXED:
        PyGMD17.revised_url = _RMD17_URL_FIXED
    return PyGMD17


def _split_dir(root: str) -> str:
    return os.path.join(root, "raw", "rmd17", "splits")


def _read_split_indices(root: str, split_idx: int, train: bool) -> List[int]:
    kind = "train" if train else "test"
    path = os.path.join(_split_dir(root), f"index_{kind}_{split_idx + 1:02d}.csv")
    with open(path, newline="") as fh:
        return [int(row[0]) for row in csv.reader(fh) if row]


class MD17Dataset(Dataset):
    def __init__(
        self,
        molecule: str,
        root: str,
        split_idx: int,
        train: bool,
        box_padding: float = 12.0,
        edge_cutoff: float = 5.0,
    ):
        if molecule not in MOLECULE_SMILES:
            raise ValueError(
                f"Unknown molecule '{molecule}'; add its canonical SMILES to "
                f"MOLECULE_SMILES first. Known: {sorted(MOLECULE_SMILES)}"
            )
        if not 0 <= split_idx <= 4:
            raise ValueError(f"split_idx must be 0-4 (rMD17 ships 5 official splits), got {split_idx}")

        PyGMD17 = get_pyg_md17_class()

        full = PyGMD17(root=root, name=f"revised {molecule}")

        indices = _read_split_indices(root, split_idx, train)
        self._data = [full[i] for i in indices]

        self.molecule = molecule
        self.smiles = MOLECULE_SMILES[molecule]
        self._box_padding = box_padding
        self._edge_cutoff = edge_cutoff

        atomic_numbers = self._data[0].z
        self._n_atoms = atomic_numbers.shape[0]
        self._types = torch.tensor(
            [ATOM_TYPE_MAP.get(_PERIODIC_TABLE.GetElementSymbol(int(z)), 0) for z in atomic_numbers],
            dtype=torch.long,
        )
        self._is_solvent = torch.zeros(self._n_atoms, dtype=torch.bool)
        self._mol_props = torch.tensor(calc_rdkit_mol_props_smiles(self.smiles), dtype=torch.float32)
        # NOTE: unlike SPICE's SMILES, MOLECULE_SMILES above carries no atom map, so structural-prior signal is weaker here than for SPICE until we get the true per-molecule atom-index permutation
        self._edge_feats = build_rdkit_edge_features(self.smiles, self._n_atoms)

    def __len__(self) -> int:
        return len(self._data)

    def energy_force_stats(self, indices=None):
        idx = range(len(self._data)) if indices is None else indices
        energies = torch.tensor([float(self._data[i].energy) * KCAL_PER_MOL_TO_EV for i in idx])
        force_sumsq = torch.tensor(
            [float((self._data[i].force * KCAL_PER_MOL_TO_EV).pow(2).sum()) for i in idx]
        )
        energy_mean = float(energies.mean())
        energy_std = float(energies.std())
        force_rms = float((force_sumsq / (self._n_atoms * 3)).mean().sqrt())
        return energy_mean, energy_std, force_rms

    def __getitem__(self, idx: int) -> Dict[str, object]:
        d = self._data[idx]
        x = d.pos.float()
        energy = float(d.energy) * KCAL_PER_MOL_TO_EV
        forces = d.force.float() * KCAL_PER_MOL_TO_EV

        V = compute_molecular_box(x, self._box_padding, self._box_padding, self._edge_cutoff)

        return {
            "idx": idx,
            "x": x,
            "types": self._types,
            "is_solvent": self._is_solvent,
            "energy": energy,
            "forces": forces,
            "mol_props": self._mol_props,
            "edge_feats": self._edge_feats,
            "smiles": self.smiles,
            "subset": self.molecule,
            "V": V,
        }
