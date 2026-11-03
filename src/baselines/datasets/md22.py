# MD22 loader. Chmiela et al., "Accurate global machine learning force fields for molecules with hundreds of atoms," Sci. Adv. 9, eadf0873 (2023) / arXiv:2209.14865.
# Used paper training set sizes here as a seeded random draw of that many frames per molecule; everything else is split 50/50 val/test.
from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from baselines.datasets.units_check import to_ev_angstrom

MD22_MOLECULES: tuple[str, ...] = (
    "Ac-Ala3-NHMe", "DHA", "stachyose", "AT-AT", "AT-AT-CG-CG", "buckyball-catcher", "double-walled_nanotube",
)

MD22_TRAIN_SIZE: dict[str, int] = {
    "Ac-Ala3-NHMe": 6000,
    "DHA": 8000,
    "stachyose": 8000,
    "AT-AT": 3000,
    "AT-AT-CG-CG": 2000,
    "buckyball-catcher": 600,
    "double-walled_nanotube": 800,
}
MD22_SPLIT_SEED = 0


class MD22Dataset(Dataset):
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


def _md22_npz_path(root: str, molecule: str) -> str:
    return os.path.join(root, f"md22_{molecule}.npz")


def load_md22(root: str, molecule: str, seed: int = MD22_SPLIT_SEED) -> tuple[Dataset, Dataset, Dataset]:
    if molecule not in MD22_MOLECULES:
        raise ValueError(f"unknown MD22 molecule {molecule!r}; expected one of {MD22_MOLECULES}")

    npz_path = _md22_npz_path(root, molecule)
    if not os.path.exists(npz_path):
        # TODO(real data): download the real MD22 archive into `npz_path`
        raise NotImplementedError(
            f"MD22 archive not found: {npz_path!r}. Download md22_{molecule}.npz from sgdml.org/#datasets "
            f"into {root!r}."
        )

    data = np.load(npz_path)
    positions = torch.from_numpy(np.asarray(data["R"])).float()                
    atomic_numbers = torch.from_numpy(np.asarray(data["z"])).long()
    energies_raw = torch.from_numpy(np.asarray(data["E"])).float().reshape(-1)   
    forces_raw = torch.from_numpy(np.asarray(data["F"])).float()            

    energies, forces = to_ev_angstrom(energies_raw, forces_raw, "kcal_mol_angstrom")

    n = positions.shape[0]
    n_train = min(MD22_TRAIN_SIZE[molecule], n)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    train_idx = perm[:n_train]
    remainder = perm[n_train:]
    half = len(remainder) // 2
    val_idx, test_idx = remainder[:half], remainder[half:]

    train = MD22Dataset(positions, atomic_numbers, energies, forces, train_idx)
    val = MD22Dataset(positions, atomic_numbers, energies, forces, val_idx)
    test = MD22Dataset(positions, atomic_numbers, energies, forces, test_idx)
    return train, val, test
