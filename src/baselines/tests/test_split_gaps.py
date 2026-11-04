# Tests for src/baselines/datasets/spice_adapter.py molecule-level splitting
# reference: MACE-OFF23 (Kovacs et al., arXiv:2312.15211), molecule-level 95/5 train/test.
# ANI-1x and qm9_fixed_permutation splits are covered in test_dataset_loaders.py.
from __future__ import annotations

import contextlib
import importlib
import sys
import types
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import pytest
from torch.utils.data import Subset

from baselines.datasets.splits import molecule_level_split

ADAPTER_MODULE = "baselines.datasets.spice_adapter"
FAKE_SPICE_MODULE = "stage_1.spice_dataset"
N_MOLECULES = 100
CONFORMERS_PER_MOLECULE = 3


class _StubSPICEDataset:
    identities: list[tuple[str, str, int]] = []
    constructed: list[tuple] = []

    def __init__(self, hdf5_paths, subsets=None, **kwargs) -> None:
        type(self).constructed.append((hdf5_paths, subsets, kwargs))

    def __len__(self) -> int:
        return len(type(self).identities)

    def identity_for(self, idx: int) -> tuple[str, str, int]:
        return type(self).identities[idx]


def _make_stub_class(identities: list[tuple[str, str, int]]) -> type:
    return type("StubSPICEDataset", (_StubSPICEDataset,), {"identities": identities, "constructed": []})


@contextlib.contextmanager
def _spice_adapter_with_stub(identities: list[tuple[str, str, int]]):
    stub_cls = _make_stub_class(identities)
    fake_module = types.ModuleType(FAKE_SPICE_MODULE)
    fake_module.SPICEDataset = stub_cls
    fake_module.spice_collate_fn = lambda batch: batch
    package = importlib.import_module("baselines.datasets")
    previous = {name: sys.modules.get(name) for name in (FAKE_SPICE_MODULE, ADAPTER_MODULE)}
    had_attr = hasattr(package, "spice_adapter")
    previous_attr = getattr(package, "spice_adapter", None)
    try:
        sys.modules[FAKE_SPICE_MODULE] = fake_module
        sys.modules.pop(ADAPTER_MODULE, None)
        yield importlib.import_module(ADAPTER_MODULE), stub_cls
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        if had_attr:
            package.spice_adapter = previous_attr
        elif hasattr(package, "spice_adapter"):
            delattr(package, "spice_adapter")


def _interleaved_identities(n_molecules: int = N_MOLECULES, conformers: int = CONFORMERS_PER_MOLECULE, files: tuple[str, ...] = ("spice.hdf5",)):
    # Conformer-major order so a molecule's rows are not contiguous.
    return [(f, f"mol_{m}", c) for c in range(conformers) for f in files for m in range(n_molecules)]


def _molecules_in(subset: Subset, identities: list[tuple[str, str, int]]) -> list[tuple[str, str]]:
    return [identities[i][:2] for i in subset.indices]


def test_default_spice_split_constants():
    with _spice_adapter_with_stub(_interleaved_identities()) as (adapter, _):
        assert adapter.SPICE_TRAIN_FRAC == 0.95
        assert adapter.SPICE_VAL_FRAC == 0.0
        assert adapter.SPICE_SPLIT_SEED == 0


def test_spice_default_split_is_95_5_by_molecule_with_empty_validation():
    identities = _interleaved_identities()
    with _spice_adapter_with_stub(identities) as (adapter, _):
        train, val, test = adapter.load_spice_molecule_split("/fake/spice.hdf5")
    assert all(isinstance(s, Subset) for s in (train, val, test))
    assert (len(train), len(val), len(test)) == (95 * CONFORMERS_PER_MOLECULE, 0, 5 * CONFORMERS_PER_MOLECULE)
    assert len(train) + len(val) + len(test) == len(identities)


def test_spice_split_has_no_molecule_leakage_and_keeps_all_conformers_together():
    identities = _interleaved_identities()
    with _spice_adapter_with_stub(identities) as (adapter, _):
        train, val, test = adapter.load_spice_molecule_split("/fake/spice.hdf5")
    train_mols, test_mols = set(_molecules_in(train, identities)), set(_molecules_in(test, identities))
    assert train_mols.isdisjoint(test_mols)
    assert len(train_mols) == 95 and len(test_mols) == 5
    assert train_mols | test_mols == {(f, g) for f, g, _ in identities}
    for subset in (train, test):
        per_molecule = {}
        for mol in _molecules_in(subset, identities):
            per_molecule[mol] = per_molecule.get(mol, 0) + 1
        assert set(per_molecule.values()) == {CONFORMERS_PER_MOLECULE}
    assert sorted(train.indices + test.indices) == list(range(len(identities)))


def test_spice_split_matches_molecule_level_split_with_default_seed_zero():
    identities = _interleaved_identities()
    expected_train, expected_val, expected_test = molecule_level_split([i[:2] for i in identities], 0.95, 0.0, 0)
    with _spice_adapter_with_stub(identities) as (adapter, _):
        train, val, test = adapter.load_spice_molecule_split("/fake/spice.hdf5")
    assert set(_molecules_in(train, identities)) == expected_train
    assert set(_molecules_in(test, identities)) == expected_test
    assert expected_val == set() and len(val) == 0


def test_spice_split_indices_are_ascending_and_subsets_share_the_full_dataset():
    identities = _interleaved_identities()
    with _spice_adapter_with_stub(identities) as (adapter, stub_cls):
        train, val, test = adapter.load_spice_molecule_split("/fake/spice.hdf5")
    assert train.indices == sorted(train.indices) and test.indices == sorted(test.indices)
    assert train.dataset is val.dataset is test.dataset
    assert len(stub_cls.constructed) == 1


def test_spice_split_is_deterministic_and_seed_sensitive():
    identities = _interleaved_identities()
    with _spice_adapter_with_stub(identities) as (adapter, _):
        first = adapter.load_spice_molecule_split("/fake/spice.hdf5")
        second = adapter.load_spice_molecule_split("/fake/spice.hdf5")
        other = adapter.load_spice_molecule_split("/fake/spice.hdf5", seed=1)
    assert [s.indices for s in first] == [s.indices for s in second]
    assert first[0].indices != other[0].indices


def test_spice_split_respects_custom_fractions_with_validation():
    identities = _interleaved_identities()
    with _spice_adapter_with_stub(identities) as (adapter, _):
        train, val, test = adapter.load_spice_molecule_split("/fake/spice.hdf5", train_frac=0.8, val_frac=0.1)
    assert (len(train), len(val), len(test)) == (80 * CONFORMERS_PER_MOLECULE, 10 * CONFORMERS_PER_MOLECULE, 10 * CONFORMERS_PER_MOLECULE)
    mols = [set(_molecules_in(s, identities)) for s in (train, val, test)]
    assert mols[0].isdisjoint(mols[1]) and mols[0].isdisjoint(mols[2]) and mols[1].isdisjoint(mols[2])


def test_spice_split_treats_same_group_name_in_different_files_as_different_molecules():
    identities = _interleaved_identities(n_molecules=10, conformers=2, files=("a.hdf5", "b.hdf5"))
    with _spice_adapter_with_stub(identities) as (adapter, _):
        train, val, test = adapter.load_spice_molecule_split("/fake/spice.hdf5")
    molecules = set(_molecules_in(train, identities)) | set(_molecules_in(test, identities))
    assert len(molecules) == 20
    # round(0.95 * 20) = 19 training molecules x 2 conformers; 1 test molecule x 2 conformers.
    assert (len(train), len(test)) == (38, 2)


def test_spice_split_forwards_paths_subsets_and_dataset_kwargs_to_spice_dataset():
    with _spice_adapter_with_stub(_interleaved_identities()) as (adapter, stub_cls):
        adapter.load_spice_molecule_split(["/fake/a.hdf5", "/fake/b.hdf5"], subsets=["pubchem"], box_padding=3.5, edge_cutoff=4.0)
    assert stub_cls.constructed == [(["/fake/a.hdf5", "/fake/b.hdf5"], ["pubchem"], {"box_padding": 3.5, "edge_cutoff": 4.0})]


def test_spice_split_rejects_invalid_fractions():
    with _spice_adapter_with_stub(_interleaved_identities()) as (adapter, _):
        with pytest.raises(ValueError):
            adapter.load_spice_molecule_split("/fake/spice.hdf5", train_frac=0.9, val_frac=0.3)


def test_spice_adapter_reexports_dataset_and_collate_from_stage_1():
    with _spice_adapter_with_stub(_interleaved_identities()) as (adapter, stub_cls):
        assert adapter.SPICEDataset is stub_cls
        assert set(adapter.__all__) == {"load_spice_molecule_split", "SPICEDataset", "spice_collate_fn"}
