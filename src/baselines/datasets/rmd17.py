# rMD17 loader with the paper's fixed 1000/1000 per-molecule split.
# Christensen & von Lilienfeld, Mach. Learn.: Sci. Technol. 1, 045018 (2020), arXiv:2007.09593
# data: figshare.com/articles/dataset/Revised_MD17_dataset_rMD17_/12672038. 
from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from baselines.datasets.splits import RMD17_MOLECULES, rmd17_fixed_indices
from baselines.datasets.units_check import to_ev_angstrom


class RMD17Dataset(Dataset):
    def __init__(
        self,
        positions: torch.Tensor,
        atomic_numbers: torch.Tensor,
        energies: torch.Tensor,
        forces: torch.Tensor,
        indices: list[int],
    ) -> None:
        self._positions = positions
        self._atomic_numbers = atomic_numbers
        self._energies = energies
        self._forces = forces
        self._indices = indices

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        frame = self._indices[i]
        return {
            "x": self._positions[frame],
            "atomic_numbers": self._atomic_numbers,
            "energy": self._energies[frame],
            "forces": self._forces[frame],
        }


def _rmd17_npz_path(root: str, molecule: str) -> str:
    return os.path.join(root, "npz_data", f"rmd17_{molecule}.npz")


def load_rmd17(root: str, molecule: str, split_index: int = 0) -> tuple[Dataset, Dataset, Dataset]:
    if molecule not in RMD17_MOLECULES:
        raise ValueError(f"unknown rMD17 molecule {molecule!r}; expected one of {RMD17_MOLECULES}")

    npz_path = _rmd17_npz_path(root, molecule)
    if not os.path.exists(npz_path):
        # TODO(real data): download the rMD17 release's per-molecule npz archive into `npz_path`
        raise NotImplementedError(
            f"rMD17 archive not found: {npz_path!r}. Download from "
            "figshare.com/articles/dataset/Revised_MD17_dataset_rMD17_/12672038 into "
            f"{os.path.join(root, 'npz_data')!r} -- figshare.com and ndownloader.figshare.com both "
            "returned HTTP 403 when probed from this environment."
        )

    data = np.load(npz_path)
    positions = torch.from_numpy(np.asarray(data["R"])).float()           
    atomic_numbers = torch.from_numpy(np.asarray(data["z"])).long()
    energies_raw = torch.from_numpy(np.asarray(data["E"])).float().reshape(-1)  
    forces_raw = torch.from_numpy(np.asarray(data["F"])).float()  

    energies, forces = to_ev_angstrom(energies_raw, forces_raw, "kcal_mol_angstrom")

    train_idx, test_idx = rmd17_fixed_indices(molecule, split_index, root=root)
    train = RMD17Dataset(positions, atomic_numbers, energies, forces, train_idx)
    # The paper's release ships only a fixed 1000/1000 train/test split per molecule --> no val split.
    val = RMD17Dataset(positions, atomic_numbers, energies, forces, [])
    test = RMD17Dataset(positions, atomic_numbers, energies, forces, test_idx)
    return train, val, test
