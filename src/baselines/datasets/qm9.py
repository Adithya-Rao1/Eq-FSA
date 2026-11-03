# Ramakrishnan, Dral, Rupp, von Lilienfeld, "Quantum chemistry structures and properties of 134 kilo molecules," Sci. Data 1, 140022 (2014). 
# Standard 110,000/10,000/10,831 split: Gilmer et al., ICML 2017 (arXiv:1704.01212), reused by Schütt et al.'s SchNet paper
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from baselines.datasets.units_check import to_ev_angstrom

QM9_PROPERTY_NAMES: tuple[str, ...] = (
    "A", "B", "C", "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "U0", "U", "H", "G", "Cv",
)
QM9_U0_INDEX = QM9_PROPERTY_NAMES.index("U0")  # Hartree; internal energy at 0K, used as the energy target

QM9_FULL_SIZE_AFTER_EXCLUSION = 130_831  # 134k raw minus the 3,054 "uncharacterized" molecules
QM9_TRAIN_SIZE = 110_000
QM9_VAL_SIZE = 10_000
QM9_TEST_SIZE = 10_831
QM9_SPLIT_SEED = 0

_ELEMENT_TO_Z = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}  # QM9's only allowed elements


@dataclass(frozen=True)
class QM9Molecule:
    positions: torch.Tensor       
    atomic_numbers: torch.Tensor 
    energy: torch.Tensor        


def parse_qm9_molecule_file(path: str) -> QM9Molecule:
    with open(path) as f:
        lines = f.read().splitlines()
    n_atoms = int(lines[0].strip())
    scalar_props = lines[1].split()[2:]  # drop the "gdb9"/"gdb" tag and the molecule index
    if len(scalar_props) != len(QM9_PROPERTY_NAMES):
        raise ValueError(
            f"{path}: expected {len(QM9_PROPERTY_NAMES)} scalar properties on line 2, got {len(scalar_props)}"
        )
    u0_hartree = float(scalar_props[QM9_U0_INDEX])

    atomic_numbers: list[int] = []
    positions: list[list[float]] = []
    for line in lines[2 : 2 + n_atoms]:
        fields = line.split()
        element = fields[0]
        # QM9's xyz files write small numbers in Fortran "D" exponent notation (e.g. "1.23*^-5") so shifting to Python's e convention
        coords = [float(v.replace("*^", "e")) for v in fields[1:4]]
        atomic_numbers.append(_ELEMENT_TO_Z[element])
        positions.append(coords)

    # QM9 has no forces --> dummy zero tensor is fed through so energy still goes through the project-wide convention instead of calling hartree_to_ev directly.
    energy_ev, _ = to_ev_angstrom(torch.tensor(u0_hartree), torch.zeros(n_atoms, 3), "hartree_bohr")

    return QM9Molecule(
        positions=torch.tensor(positions, dtype=torch.float32),
        atomic_numbers=torch.tensor(atomic_numbers, dtype=torch.long),
        energy=energy_ev.float(),
    )


class QM9Dataset(Dataset):
    def __init__(self, molecules: list[QM9Molecule]) -> None:
        self._molecules = molecules

    def __len__(self) -> int:
        return len(self._molecules)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        mol = self._molecules[idx]
        return {"x": mol.positions, "atomic_numbers": mol.atomic_numbers, "energy": mol.energy}


def _discover_molecule_files(root: str) -> list[str]:
    paths = sorted(glob.glob(os.path.join(root, "*.xyz")))
    if not paths:
        raise FileNotFoundError(
            f"No QM9 '*.xyz' files found under {root!r}. Expected the extracted dsgdb9nsd_*.xyz files "
            "from the GDB-9 release (figshare, via Ramakrishnan et al. 2014); figshare.com and "
            "ndownloader.figshare.com both returned HTTP 403 when probed from this environment, so no "
            "real data is bundled here."
        )
    return paths


def qm9_fixed_permutation(n: int, seed: int = QM9_SPLIT_SEED) -> list[int]:
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(n, generator=generator).tolist()


def load_qm9(root: str, seed: int = QM9_SPLIT_SEED) -> tuple[Dataset, Dataset, Dataset]:
    paths = _discover_molecule_files(root)
    molecules = [parse_qm9_molecule_file(p) for p in paths]
    n = len(molecules)

    if n == QM9_FULL_SIZE_AFTER_EXCLUSION:
        n_train, n_val = QM9_TRAIN_SIZE, QM9_VAL_SIZE
    else:
        n_train = round(n * QM9_TRAIN_SIZE / QM9_FULL_SIZE_AFTER_EXCLUSION)
        n_val = round(n * QM9_VAL_SIZE / QM9_FULL_SIZE_AFTER_EXCLUSION)

    perm = qm9_fixed_permutation(n, seed=seed)
    train = QM9Dataset([molecules[i] for i in perm[:n_train]])
    val = QM9Dataset([molecules[i] for i in perm[n_train : n_train + n_val]])
    test = QM9Dataset([molecules[i] for i in perm[n_train + n_val :]])
    return train, val, test
