# MACE-OFF23 (Kovacs et al., arXiv:2312.15211) uses molecule-level 95/5 train/test (no val split)
from __future__ import annotations

import os
import sys

from torch.utils.data import Dataset, Subset

_SRC_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

from stage_1.spice_dataset import SPICEDataset, spice_collate_fn  # noqa: E402  (re-exported for callers)

from baselines.datasets.splits import molecule_level_split

SPICE_TRAIN_FRAC = 0.95
SPICE_VAL_FRAC = 0.0  
SPICE_SPLIT_SEED = 0

__all__ = ["load_spice_molecule_split", "SPICEDataset", "spice_collate_fn"]


def _molecule_ids(dataset: SPICEDataset) -> list[tuple[str, str]]:
    return [dataset.identity_for(i)[:2] for i in range(len(dataset))]


def load_spice_molecule_split(
    hdf5_paths: str | list[str],
    subsets: list[str] | None = None,
    train_frac: float = SPICE_TRAIN_FRAC,
    val_frac: float = SPICE_VAL_FRAC,
    seed: int = SPICE_SPLIT_SEED,
    val_from_train_frac: float = 0.0,
    **spice_dataset_kwargs: object,
) -> tuple[Dataset, Dataset, Dataset]:
    full = SPICEDataset(hdf5_paths, subsets=subsets, **spice_dataset_kwargs)
    molecule_ids = _molecule_ids(full)

    train_mols, val_mols, test_mols = molecule_level_split(molecule_ids, train_frac, val_frac, seed)

    if val_from_train_frac:
        if val_frac:
            raise ValueError("val_from_train_frac and val_frac are mutually exclusive")
        train_mols, val_mols, _ = molecule_level_split(sorted(train_mols, key=repr), 1.0 - val_from_train_frac, val_from_train_frac, seed + 1)

    train_idx = [i for i, m in enumerate(molecule_ids) if m in train_mols]
    val_idx = [i for i, m in enumerate(molecule_ids) if m in val_mols]
    test_idx = [i for i, m in enumerate(molecule_ids) if m in test_mols]

    return Subset(full, train_idx), Subset(full, val_idx), Subset(full, test_idx)
