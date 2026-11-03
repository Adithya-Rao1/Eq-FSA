# Tests for src/baselines/datasets/splits.py: molecule-level split leakage/determinism, and rMD17's fixed-index reader.
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import pytest

from baselines.datasets.splits import (
    RMD17_MOLECULES,
    RMD17_TEST_SIZE,
    RMD17_TRAIN_SIZE,
    molecule_level_split,
    rmd17_fixed_indices,
)

MOLECULE_IDS = [f"mol_{i}" for i in range(1000)]
CONFORMERS_PER_MOLECULE = 5
ROW_MOLECULE_IDS = [mol for mol in MOLECULE_IDS for _ in range(CONFORMERS_PER_MOLECULE)]


def test_molecule_level_split_no_leakage_across_splits():
    train, val, test = molecule_level_split(ROW_MOLECULE_IDS, train_frac=0.8, val_frac=0.1, seed=0)
    assert train & val == set()
    assert train & test == set()
    assert val & test == set()
    assert train | val | test == set(MOLECULE_IDS)


def test_molecule_level_split_sizes_match_requested_fractions():
    train, val, test = molecule_level_split(MOLECULE_IDS, train_frac=0.8, val_frac=0.1, seed=0)
    n = len(MOLECULE_IDS)
    assert abs(len(train) - 0.8 * n) <= 1
    assert abs(len(val) - 0.1 * n) <= 1
    assert abs(len(test) - 0.1 * n) <= 1
    assert len(train) + len(val) + len(test) == n


def test_molecule_level_split_deterministic_given_seed():
    split_a = molecule_level_split(MOLECULE_IDS, train_frac=0.8, val_frac=0.1, seed=42)
    split_b = molecule_level_split(MOLECULE_IDS, train_frac=0.8, val_frac=0.1, seed=42)
    assert split_a == split_b


def test_molecule_level_split_different_seeds_differ():
    split_a = molecule_level_split(MOLECULE_IDS, train_frac=0.8, val_frac=0.1, seed=0)
    split_b = molecule_level_split(MOLECULE_IDS, train_frac=0.8, val_frac=0.1, seed=1)
    assert split_a != split_b


def test_molecule_level_split_rejects_invalid_fractions():
    with pytest.raises(ValueError):
        molecule_level_split(MOLECULE_IDS, train_frac=0.9, val_frac=0.3, seed=0)


def _write_fake_rmd17_split(root: str, split_index: int, train_idx: list[int], test_idx: list[int]) -> None:
    split_dir = os.path.join(root, "splits")
    os.makedirs(split_dir, exist_ok=True)
    one_based = split_index + 1
    with open(os.path.join(split_dir, f"index_train_{one_based:02d}.csv"), "w") as f:
        f.write("\n".join(str(i) for i in train_idx) + "\n")
    with open(os.path.join(split_dir, f"index_test_{one_based:02d}.csv"), "w") as f:
        f.write("\n".join(str(i) for i in test_idx) + "\n")


def test_rmd17_fixed_indices_reads_fake_split_deterministically():
    fake_train = list(range(RMD17_TRAIN_SIZE))
    fake_test = list(range(RMD17_TRAIN_SIZE, RMD17_TRAIN_SIZE + RMD17_TEST_SIZE))
    with tempfile.TemporaryDirectory() as root:
        _write_fake_rmd17_split(root, split_index=0, train_idx=fake_train, test_idx=fake_test)
        result_a = rmd17_fixed_indices("aspirin", split_index=0, root=root)
        result_b = rmd17_fixed_indices("aspirin", split_index=0, root=root)
        assert result_a == result_b == (fake_train, fake_test)


def test_rmd17_fixed_indices_distinguishes_split_index():
    fake_train_0 = list(range(RMD17_TRAIN_SIZE))
    fake_test_0 = list(range(RMD17_TRAIN_SIZE, RMD17_TRAIN_SIZE + RMD17_TEST_SIZE))
    fake_train_1 = list(range(2000, 2000 + RMD17_TRAIN_SIZE))
    fake_test_1 = list(range(2000 + RMD17_TRAIN_SIZE, 2000 + RMD17_TRAIN_SIZE + RMD17_TEST_SIZE))
    with tempfile.TemporaryDirectory() as root:
        _write_fake_rmd17_split(root, split_index=0, train_idx=fake_train_0, test_idx=fake_test_0)
        _write_fake_rmd17_split(root, split_index=1, train_idx=fake_train_1, test_idx=fake_test_1)
        assert rmd17_fixed_indices("aspirin", split_index=0, root=root) == (fake_train_0, fake_test_0)
        assert rmd17_fixed_indices("aspirin", split_index=1, root=root) == (fake_train_1, fake_test_1)


def test_rmd17_fixed_indices_raises_not_implemented_when_real_data_missing():
    with tempfile.TemporaryDirectory() as root:
        with pytest.raises(NotImplementedError):
            rmd17_fixed_indices("aspirin", split_index=0, root=root)


def test_rmd17_fixed_indices_rejects_unknown_molecule():
    with pytest.raises(ValueError):
        rmd17_fixed_indices("not_a_real_molecule", split_index=0)


def test_rmd17_fixed_indices_rejects_out_of_range_split_index():
    with pytest.raises(ValueError):
        rmd17_fixed_indices("aspirin", split_index=5)


def test_rmd17_molecules_has_ten_entries():
    assert len(RMD17_MOLECULES) == 10
