# Shared dataset dispatch for the tier 0/1 adapters: maps (--dataset, --split) onto the existing
# per-dataset loaders in baselines.datasets. Every loader returns a (train, val, test) tuple.
#
# --dataset: "qm9" | "ani1x" | "spice" | "rmd17:<molecule>" | "md22:<molecule>"
# --data-root: dataset root dir (qm9/rmd17/md22) or file path (ani1x .h5, spice .hdf5)
from __future__ import annotations

import re
from functools import lru_cache

from torch.utils.data import Dataset

_SPLIT_INDEX = {"train": 0, "val": 1, "test": 2}


@lru_cache(maxsize=4)
def _load_all(dataset: str, split: str, data_root: str) -> tuple[Dataset, Dataset, Dataset]:
    name, _, molecule = dataset.partition(":")
    if name == "qm9":
        from baselines.datasets.qm9 import load_qm9
        return load_qm9(data_root)
    if name == "ani1x":
        from baselines.datasets.ani1x import load_ani1x
        return load_ani1x(data_root)
    if name == "spice":
        from baselines.datasets.spice_adapter import load_spice_molecule_split
        match = re.fullmatch(r"molecule_95_5_val(\d+)", split)
        if match:  
            return load_spice_molecule_split(data_root, val_from_train_frac=int(match.group(1)) / 100)
        return load_spice_molecule_split(data_root)
    if name in ("rmd17", "md22"):
        if not molecule:
            raise ValueError(f"--dataset {dataset!r} needs a molecule, e.g. '{name}:<molecule>'")
        if name == "md22":
            from baselines.datasets.md22 import load_md22
            return load_md22(data_root, molecule)
        from baselines.datasets.rmd17 import load_rmd17
        match = re.fullmatch(r"rmd17_split_(\d+)(?:_val(\d+))?", split)
        train, val, test = load_rmd17(data_root, molecule, int(match.group(1)) if match else 0)
        if match and match.group(2):  
            from baselines.datasets.splits import carve_validation_frames
            train, val = carve_validation_frames(train, int(match.group(2)))
        return train, val, test
    raise ValueError(f"unknown dataset {dataset!r}; expected qm9, ani1x, spice, rmd17:<mol> or md22:<mol>")


def load_split(dataset: str, split: str, which: str, data_root: str | None) -> Dataset:
    if which not in _SPLIT_INDEX:
        raise ValueError(f"which must be one of {sorted(_SPLIT_INDEX)}, got {which!r}")
    if not data_root:
        raise ValueError("--data-root is required to load a dataset")
    return _load_all(dataset, split, data_root)[_SPLIT_INDEX[which]]


def load_splits(dataset: str, split: str, data_root: str | None) -> tuple[Dataset, Dataset, Dataset]:
    if not data_root:
        raise ValueError("--data-root is required to load a dataset")
    return _load_all(dataset, split, data_root)
