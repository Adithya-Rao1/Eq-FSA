# ANI-1x / ANI-1ccx loader. Smith et al., J. Chem. Phys. 148, 241733 (2018) (ANI-1x) / Sci. Data 7, 134 (2020) (ANI-1ccx).
# Split is by HDF5 group (= chemical formula)
# Every conformer of a molecule lands on one side and sequence-aware models cannot leak across splits --> whole formulas held out for stricter OOD testing
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterator

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from baselines.config.units import BOHR_TO_ANGSTROM
from baselines.datasets.splits import molecule_level_split
from baselines.datasets.units_check import to_ev_angstrom

ANI1X_DEFAULT_ENERGY_KEY = "wb97x_dz.energy"
ANI1X_DEFAULT_FORCES_KEY = "wb97x_dz.forces"
ANI1CCX_ENERGY_KEY = "ccsd(t)_cbs.energy"  # ANI-1ccx: energy-only coupled-cluster resample of ~10% of ANI-1x

ANI1X_TRAIN_FRAC = 0.8
ANI1X_VAL_FRAC = 0.1
# No official split exists for ANI-1x/1ccx in the literature
ANI1X_SPLIT_SEED = 0


@dataclass(frozen=True)
class ANI1xSample:
    positions: torch.Tensor        
    atomic_numbers: torch.Tensor   
    energy: torch.Tensor           
    forces: torch.Tensor | None    


def _iter_groups(
    h5_path: str, energy_key: str, forces_key: str | None,
) -> Iterator[tuple[str, np.ndarray, np.ndarray, float, np.ndarray | None]]:
    with h5py.File(h5_path, "r") as fh:
        for group_name, grp in fh.items():
            if energy_key not in grp or "coordinates" not in grp or "atomic_numbers" not in grp:
                continue
            atomic_numbers = np.asarray(grp["atomic_numbers"])
            coords = np.asarray(grp["coordinates"])
            energies = np.asarray(grp[energy_key])    
            forces_all = (
                np.asarray(grp[forces_key]) if forces_key is not None and forces_key in grp else None
            )
            for c in range(coords.shape[0]):
                e = energies[c]
                if not np.isfinite(e) or not np.isfinite(coords[c]).all():
                    continue
                if forces_key is not None:
                    if forces_all is None or not np.isfinite(forces_all[c]).all():
                        continue
                    f = forces_all[c]
                else:
                    f = None
                yield group_name, atomic_numbers, coords[c], float(e), f


class ANI1xDataset(Dataset):
    def __init__(self, samples: list[ANI1xSample]) -> None:
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = self._samples[idx]
        item = {"x": s.positions, "atomic_numbers": s.atomic_numbers, "energy": s.energy}
        if s.forces is not None:
            item["forces"] = s.forces
        return item


def load_ani1x(
    h5_path: str,
    energy_key: str = ANI1X_DEFAULT_ENERGY_KEY,
    forces_key: str | None = ANI1X_DEFAULT_FORCES_KEY,
    seed: int = ANI1X_SPLIT_SEED,
) -> tuple[Dataset, Dataset, Dataset]:
    if not os.path.exists(h5_path):
        raise FileNotFoundError(
            f"{h5_path!r} not found. Download the ANI-1x/1ccx release HDF5 there and point load_ani1x "
            "at it; this repo does not fetch multi-GB datasets in this environment."
        )

    samples: list[ANI1xSample] = []
    group_names: list[str] = []
    for group_name, atomic_numbers, coords, energy_hartree, force_hartree_ang in _iter_groups(h5_path, energy_key, forces_key):
        positions = torch.from_numpy(coords).float()  # already Angstrom

        if force_hartree_ang is not None:
            # ANI's native force units are Hartree and Angstrom (not Bohr) --> Pre-scale by BOHR_TO_ANGSTROM here so division in bohr_to_angstrom cancels
            forces_raw = torch.from_numpy(force_hartree_ang).float() * BOHR_TO_ANGSTROM
        else:
            forces_raw = torch.zeros_like(positions)
        energy_ev, forces_ev = to_ev_angstrom(torch.tensor(energy_hartree), forces_raw, "hartree_bohr")

        group_names.append(group_name)
        samples.append(
            ANI1xSample(
                positions=positions,
                atomic_numbers=torch.from_numpy(atomic_numbers).long(),
                energy=energy_ev.float(),
                forces=forces_ev.float() if force_hartree_ang is not None else None,
            )
        )

    train_groups, val_groups, test_groups = molecule_level_split(group_names, ANI1X_TRAIN_FRAC, ANI1X_VAL_FRAC, seed)
    train = ANI1xDataset([s for s, g in zip(samples, group_names) if g in train_groups])
    val = ANI1xDataset([s for s, g in zip(samples, group_names) if g in val_groups])
    test = ANI1xDataset([s for s, g in zip(samples, group_names) if g in test_groups])
    return train, val, test
