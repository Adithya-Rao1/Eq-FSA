# Tests for src/baselines/adapters/dataset_loading.py

from __future__ import annotations

import sys
import types
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import pytest
import torch
from torch.utils.data import TensorDataset

from baselines.adapters import dataset_loading
from baselines.adapters.dataset_loading import _load_all, load_split

WHICH_TO_INDEX = {"train": 0, "val": 1, "test": 2}
DATA_ROOT = "/fake/root"


def _fake_triple(tag: int) -> tuple[TensorDataset, TensorDataset, TensorDataset]:
    # every element of split k carries the value tag * 10 + k.
    return tuple(TensorDataset(torch.full((2,), float(tag * 10 + k))) for k in range(3))


def _tag_of(dataset: TensorDataset) -> float:
    return float(dataset[0][0])


class _Recorder:
    def __init__(self, tag: int) -> None:
        self.tag = tag
        self.calls: list[tuple] = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return _fake_triple(self.tag)


def _install_fake_module(monkeypatch, module_name: str, **attrs) -> None:
    # ani1x / spice_adapter import h5py / rdkit at module level, so replace the whole module instead of patching attribute.
    module = types.ModuleType(module_name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, module_name, module)


def _patch_all_loaders(monkeypatch) -> dict[str, _Recorder]:
    recorders = {name: _Recorder(tag) for tag, name in enumerate(["qm9", "ani1x", "spice", "rmd17", "md22"], start=1)}
    _install_fake_module(monkeypatch, "baselines.datasets.qm9", load_qm9=recorders["qm9"])
    _install_fake_module(monkeypatch, "baselines.datasets.ani1x", load_ani1x=recorders["ani1x"])
    _install_fake_module(monkeypatch, "baselines.datasets.spice_adapter", load_spice_molecule_split=recorders["spice"])
    _install_fake_module(monkeypatch, "baselines.datasets.rmd17", load_rmd17=recorders["rmd17"])
    _install_fake_module(monkeypatch, "baselines.datasets.md22", load_md22=recorders["md22"])
    _load_all.cache_clear()
    return recorders


def test_dispatch_qm9_passes_data_root_only(monkeypatch):
    recorders = _patch_all_loaders(monkeypatch)
    try:
        dataset = load_split("qm9", "any", "train", DATA_ROOT)
        assert _tag_of(dataset) == 1 * 10 + 0
        assert recorders["qm9"].calls == [((DATA_ROOT,), {})]
        assert all(not r.calls for name, r in recorders.items() if name != "qm9")
    finally:
        _load_all.cache_clear()


def test_dispatch_ani1x_passes_data_root_only(monkeypatch):
    recorders = _patch_all_loaders(monkeypatch)
    try:
        assert _tag_of(load_split("ani1x", "any", "val", DATA_ROOT)) == 2 * 10 + 1
        assert recorders["ani1x"].calls == [((DATA_ROOT,), {})]
        assert all(not r.calls for name, r in recorders.items() if name != "ani1x")
    finally:
        _load_all.cache_clear()


def test_dispatch_spice_passes_data_root_only(monkeypatch):
    recorders = _patch_all_loaders(monkeypatch)
    try:
        assert _tag_of(load_split("spice", "any", "test", DATA_ROOT)) == 3 * 10 + 2
        assert recorders["spice"].calls == [((DATA_ROOT,), {})]
        assert all(not r.calls for name, r in recorders.items() if name != "spice")
    finally:
        _load_all.cache_clear()


def test_dispatch_rmd17_passes_root_molecule_and_default_split_index_zero(monkeypatch):
    recorders = _patch_all_loaders(monkeypatch)
    try:
        assert _tag_of(load_split("rmd17:aspirin", "rmd17_fixed_1000", "train", DATA_ROOT)) == 4 * 10 + 0
        assert recorders["rmd17"].calls == [((DATA_ROOT, "aspirin", 0), {})]
        assert all(not r.calls for name, r in recorders.items() if name != "rmd17")
    finally:
        _load_all.cache_clear()


def test_dispatch_md22_passes_root_and_molecule_and_ignores_split_name(monkeypatch):
    recorders = _patch_all_loaders(monkeypatch)
    try:
        assert _tag_of(load_split("md22:Ac-Ala3-NHMe", "rmd17_split_3", "val", DATA_ROOT)) == 5 * 10 + 1
        assert recorders["md22"].calls == [((DATA_ROOT, "Ac-Ala3-NHMe"), {})]
        assert all(not r.calls for name, r in recorders.items() if name != "md22")
    finally:
        _load_all.cache_clear()


@pytest.mark.parametrize("which", ["train", "val", "test"])
def test_which_selects_the_matching_tuple_element(monkeypatch, which):
    _patch_all_loaders(monkeypatch)
    try:
        dataset = load_split("qm9", "any", which, DATA_ROOT)
        assert _tag_of(dataset) == 1 * 10 + WHICH_TO_INDEX[which]
        assert dataset is _load_all("qm9", "any", DATA_ROOT)[WHICH_TO_INDEX[which]]
    finally:
        _load_all.cache_clear()


@pytest.mark.parametrize("split, expected_index", [
    ("rmd17_split_0", 0), ("rmd17_split_1", 1), ("rmd17_split_4", 4), ("rmd17_split_12", 12),
])
def test_rmd17_split_name_is_parsed_into_split_index(monkeypatch, split, expected_index):
    recorders = _patch_all_loaders(monkeypatch)
    try:
        load_split("rmd17:ethanol", split, "train", DATA_ROOT)
        assert recorders["rmd17"].calls == [((DATA_ROOT, "ethanol", expected_index), {})]
    finally:
        _load_all.cache_clear()


@pytest.mark.parametrize("split", ["rmd17_fixed_1000", "molecule_95_5", "rmd17_split_", "rmd17_split_x", "rmd17_split_1_extra", "xrmd17_split_2", "rmd17_split_-1"])
def test_rmd17_non_matching_split_name_falls_back_to_index_zero(monkeypatch, split):
    # re.fullmatch --> anything that is not exactly rmd17_split_<digits> silently selects split 0.
    recorders = _patch_all_loaders(monkeypatch)
    try:
        load_split("rmd17:ethanol", split, "train", DATA_ROOT)
        assert recorders["rmd17"].calls == [((DATA_ROOT, "ethanol", 0), {})]
    finally:
        _load_all.cache_clear()


def test_rmd17_split_index_is_an_int(monkeypatch):
    recorders = _patch_all_loaders(monkeypatch)
    try:
        load_split("rmd17:uracil", "rmd17_split_2", "test", DATA_ROOT)
        assert type(recorders["rmd17"].calls[0][0][2]) is int
    finally:
        _load_all.cache_clear()


@pytest.mark.parametrize("dataset", ["rmd17", "rmd17:", "md22", "md22:"])
def test_missing_molecule_for_rmd17_and_md22_raises_value_error(monkeypatch, dataset):
    recorders = _patch_all_loaders(monkeypatch)
    try:
        with pytest.raises(ValueError, match="needs a molecule"):
            load_split(dataset, "any", "train", DATA_ROOT)
        assert not recorders["rmd17"].calls and not recorders["md22"].calls
    finally:
        _load_all.cache_clear()


@pytest.mark.parametrize("dataset", ["", "cifar10", "QM9", "qm9x", "ani2x", ":aspirin", "omol25"])
def test_unknown_dataset_raises_value_error(monkeypatch, dataset):
    recorders = _patch_all_loaders(monkeypatch)
    try:
        with pytest.raises(ValueError, match="unknown dataset"):
            load_split(dataset, "any", "train", DATA_ROOT)
        assert all(not r.calls for r in recorders.values())
    finally:
        _load_all.cache_clear()


@pytest.mark.parametrize("data_root", [None, ""])
def test_missing_data_root_raises_value_error_before_loading(monkeypatch, data_root):
    recorders = _patch_all_loaders(monkeypatch)
    try:
        with pytest.raises(ValueError, match="--data-root"):
            load_split("qm9", "any", "train", data_root)
        assert all(not r.calls for r in recorders.values())
    finally:
        _load_all.cache_clear()


@pytest.mark.parametrize("which", ["", "training", "Train", "valid", "validation", "all", None])
def test_invalid_which_raises_value_error_before_loading(monkeypatch, which):
    recorders = _patch_all_loaders(monkeypatch)
    try:
        with pytest.raises(ValueError, match="which must be one of"):
            load_split("qm9", "any", which, DATA_ROOT)
        assert all(not r.calls for r in recorders.values())
    finally:
        _load_all.cache_clear()


def test_invalid_which_is_reported_even_without_data_root():
    with pytest.raises(ValueError, match="which must be one of"):
        load_split("qm9", "any", "bogus", None)


def test_repeated_calls_hit_the_cache_and_return_the_same_object(monkeypatch):
    recorders = _patch_all_loaders(monkeypatch)
    try:
        first = load_split("qm9", "any", "train", DATA_ROOT)
        second = load_split("qm9", "any", "train", DATA_ROOT)
        val = load_split("qm9", "any", "val", DATA_ROOT)
        test = load_split("qm9", "any", "test", DATA_ROOT)
        assert first is second
        assert len(recorders["qm9"].calls) == 1
        assert {_tag_of(first), _tag_of(val), _tag_of(test)} == {10.0, 11.0, 12.0}
    finally:
        _load_all.cache_clear()


def test_cache_is_keyed_on_dataset_split_and_data_root(monkeypatch):
    recorders = _patch_all_loaders(monkeypatch)
    try:
        load_split("rmd17:aspirin", "rmd17_split_0", "train", DATA_ROOT)
        load_split("rmd17:aspirin", "rmd17_split_1", "train", DATA_ROOT)
        load_split("rmd17:ethanol", "rmd17_split_0", "train", DATA_ROOT)
        load_split("rmd17:aspirin", "rmd17_split_0", "train", "/other/root")
        assert [call[0] for call in recorders["rmd17"].calls] == [
            (DATA_ROOT, "aspirin", 0), (DATA_ROOT, "aspirin", 1), (DATA_ROOT, "ethanol", 0), ("/other/root", "aspirin", 0),
        ]
    finally:
        _load_all.cache_clear()


def test_cache_clear_forces_a_reload(monkeypatch):
    recorders = _patch_all_loaders(monkeypatch)
    try:
        load_split("qm9", "any", "train", DATA_ROOT)
        _load_all.cache_clear()
        load_split("qm9", "any", "train", DATA_ROOT)
        assert len(recorders["qm9"].calls) == 2
    finally:
        _load_all.cache_clear()


def test_loader_errors_propagate_and_are_not_cached(monkeypatch):
    attempts: list[str] = []

    def flaky_loader(root: str):
        attempts.append(root)
        if len(attempts) == 1:
            raise FileNotFoundError("no xyz files")
        return _fake_triple(1)

    _patch_all_loaders(monkeypatch)
    _install_fake_module(monkeypatch, "baselines.datasets.qm9", load_qm9=flaky_loader)
    try:
        with pytest.raises(FileNotFoundError, match="no xyz files"):
            load_split("qm9", "any", "train", DATA_ROOT)
        assert _tag_of(load_split("qm9", "any", "train", DATA_ROOT)) == 10.0
        assert len(attempts) == 2
    finally:
        _load_all.cache_clear()


def test_load_split_real_qm9_and_md22_modules_are_patchable_by_attribute(monkeypatch):
    # The brief's contract --> the lazy `from baselines.datasets.<mod> import load_*` must resolve patched attributes.
    import baselines.datasets.md22 as md22_module
    import baselines.datasets.qm9 as qm9_module
    import baselines.datasets.rmd17 as rmd17_module

    qm9_recorder, md22_recorder, rmd17_recorder = _Recorder(1), _Recorder(5), _Recorder(4)
    monkeypatch.setattr(qm9_module, "load_qm9", qm9_recorder)
    monkeypatch.setattr(md22_module, "load_md22", md22_recorder)
    monkeypatch.setattr(rmd17_module, "load_rmd17", rmd17_recorder)
    _load_all.cache_clear()
    try:
        load_split("qm9", "s", "train", DATA_ROOT)
        load_split("md22:DHA", "s", "train", DATA_ROOT)
        load_split("rmd17:benzene", "rmd17_split_2", "train", DATA_ROOT)
        assert qm9_recorder.calls == [((DATA_ROOT,), {})]
        assert md22_recorder.calls == [((DATA_ROOT, "DHA"), {})]
        assert rmd17_recorder.calls == [((DATA_ROOT, "benzene", 2), {})]
    finally:
        _load_all.cache_clear()


def test_split_index_mapping_contract():
    assert dataset_loading._SPLIT_INDEX == WHICH_TO_INDEX
