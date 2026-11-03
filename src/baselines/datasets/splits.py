from __future__ import annotations

import os
import random
from typing import Hashable, Sequence

RMD17_MOLECULES: tuple[str, ...] = (
    "aspirin", "benzene", "ethanol", "malonaldehyde", "naphthalene",
    "salicylic_acid", "toluene", "uracil", "paracetamol", "azobenzene",
)
RMD17_NUM_SPLITS = 5
RMD17_TRAIN_SIZE = 1000
RMD17_TEST_SIZE = 1000
RMD17_DEFAULT_ROOT = os.environ.get("RMD17_ROOT", os.path.join("data", "rmd17"))


def molecule_level_split(
    molecule_ids: Sequence[Hashable], train_frac: float, val_frac: float, seed: int,
) -> tuple[set, set, set]:
    if not (0.0 <= train_frac <= 1.0 and 0.0 <= val_frac <= 1.0 and train_frac + val_frac <= 1.0 + 1e-9):
        raise ValueError(f"train_frac={train_frac}, val_frac={val_frac} must each be in [0,1] and sum to <= 1")

    # Sort by repr() first so the shuffle below is deterministic regardless of the caller input order or hash randomization of set/dict iteration order.
    unique_ids = sorted(set(molecule_ids), key=repr)
    random.Random(seed).shuffle(unique_ids)

    n = len(unique_ids)
    n_train = round(train_frac * n)
    n_val = round(val_frac * n)

    train_ids = unique_ids[:n_train]
    val_ids = unique_ids[n_train : n_train + n_val]
    test_ids = unique_ids[n_train + n_val :]
    return set(train_ids), set(val_ids), set(test_ids)


def _rmd17_split_paths(root: str, split_index: int) -> tuple[str, str]:
    split_dir = os.path.join(root, "splits")
    one_based = split_index + 1  # the paper's release numbers its 5 fixed splits 01..05
    train_path = os.path.join(split_dir, f"index_train_{one_based:02d}.csv")
    test_path = os.path.join(split_dir, f"index_test_{one_based:02d}.csv")
    return train_path, test_path


def _read_index_file(path: str) -> list[int]:
    with open(path) as f:
        return [int(line.strip()) for line in f if line.strip()]


def rmd17_fixed_indices(
    molecule: str, split_index: int = 0, root: str | None = None,
) -> tuple[list[int], list[int]]:
    if molecule not in RMD17_MOLECULES:
        raise ValueError(f"unknown rMD17 molecule {molecule!r}; expected one of {RMD17_MOLECULES}")
    if not (0 <= split_index < RMD17_NUM_SPLITS):
        raise ValueError(
            f"rMD17 ships exactly {RMD17_NUM_SPLITS} fixed splits (0-{RMD17_NUM_SPLITS - 1}), "
            f"got split_index={split_index}"
        )

    root = RMD17_DEFAULT_ROOT if root is None else root
    train_path, test_path = _rmd17_split_paths(root, split_index)
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        # TODO(real data): download the rMD17 release's `splits/` directory into `root` 
        raise NotImplementedError(
            f"rMD17 fixed split files not found: {train_path!r}, {test_path!r}. Download the release "
            "from figshare.com/articles/dataset/Revised_MD17_dataset_rMD17_/12672038 into "
            f"{root!r} (expects a 'splits/index_{{train,test}}_01..05.csv' layout) -- both figshare.com "
            "and ndownloader.figshare.com returned HTTP 403 when probed from this environment."
        )

    train_idx = _read_index_file(train_path)
    test_idx = _read_index_file(test_path)
    if len(train_idx) != RMD17_TRAIN_SIZE or len(test_idx) != RMD17_TEST_SIZE:
        raise ValueError(
            f"{train_path!r}/{test_path!r} should contain {RMD17_TRAIN_SIZE}/{RMD17_TEST_SIZE} indices "
            f"per the paper's fixed splits; got {len(train_idx)}/{len(test_idx)}"
        )
    return train_idx, test_idx
