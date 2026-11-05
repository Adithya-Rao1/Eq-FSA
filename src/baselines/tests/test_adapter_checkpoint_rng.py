# Tests for _save_checkpoint/_load_checkpoint/_build_dataloader in src/baselines/adapters/{tier0_legacy,tier1_mpnn}.py

from __future__ import annotations

import argparse
import dataclasses
import random
import sys
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import numpy as np
import pytest
import torch
from torch.utils.data import RandomSampler, SequentialSampler, TensorDataset

from baselines.adapters import tier0_legacy, tier1_mpnn
from baselines.config.schema import CommonConfig, RunPaths
from baselines.provenance import seeding
from baselines.provenance.seeding import set_seed

ADAPTERS = [(tier0_legacy, "schnet"), (tier1_mpnn, "nequip")]
ADAPTER_IDS = ["tier0_legacy", "tier1_mpnn"]
DATASET_SIZE = 16
EXPECTED_CHECKPOINT_KEYS = {"epoch", "global_step", "model_state_dict", "optimizer_state_dict", "config", "rng_state"}


def _parse_args(adapter, model_name: str, run_dir: Path, extra: list[str] | None = None) -> tuple[argparse.Namespace, CommonConfig]:
    argv = ["--dataset", "qm9", "--split", "molecule_95_5", "--run-dir", str(run_dir), "--model", model_name] + (extra or [])
    args = adapter.build_parser().parse_args(argv)
    return args, CommonConfig.from_args(args)


def _make_state(init_seed: int) -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    torch.manual_seed(init_seed)
    # Dropout makes the post-resume step depend on the torch RNG stream, so a missed restore changes the weights.
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(0.5), torch.nn.Linear(4, 1))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    return model, optimizer


def _step(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
    model.train()
    optimizer.zero_grad()
    model(torch.ones(8, 3)).pow(2).sum().backward()
    optimizer.step()


def _draws() -> tuple[list[float], list[float], list[float]]:
    return torch.rand(4).tolist(), np.random.rand(4).tolist(), [random.random() for _ in range(4)]


def _weights_equal(model_a: torch.nn.Module, model_b: torch.nn.Module) -> bool:
    state_a, state_b = model_a.state_dict(), model_b.state_dict()
    return state_a.keys() == state_b.keys() and all(torch.equal(state_a[k], state_b[k]) for k in state_a)


def _momentum_buffers(optimizer: torch.optim.Optimizer) -> list[torch.Tensor]:
    return [optimizer.state[p]["momentum_buffer"] for group in optimizer.param_groups for p in group["params"]]


@pytest.mark.parametrize("adapter, model_name", ADAPTERS, ids=ADAPTER_IDS)
def test_checkpoint_round_trip_restores_weights_optimizer_counters_and_returns_them(adapter, model_name, tmp_path):
    args, cfg = _parse_args(adapter, model_name, tmp_path)
    path = RunPaths(str(tmp_path)).ensure().checkpoint(f"{model_name}_epoch2")
    set_seed(0, deterministic=False)
    model, optimizer = _make_state(init_seed=1)
    _step(model, optimizer)
    _step(model, optimizer)
    adapter._save_checkpoint(path, 2, 14, model, optimizer, cfg, args)

    fresh_model, fresh_optimizer = _make_state(init_seed=2)
    assert not _weights_equal(model, fresh_model)
    start_epoch, global_step = adapter._load_checkpoint(path, fresh_model, fresh_optimizer)

    # The checkpoint was written at the end of epoch 2, so training resumes at epoch 3; global_step is already post-increment.
    assert (start_epoch, global_step) == (3, 14)
    assert _weights_equal(model, fresh_model)
    for saved, loaded in zip(_momentum_buffers(optimizer), _momentum_buffers(fresh_optimizer)):
        assert torch.equal(saved, loaded)
    assert fresh_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"] == 0.1
    assert fresh_optimizer.param_groups[0]["momentum"] == 0.9


@pytest.mark.parametrize("adapter, model_name", ADAPTERS, ids=ADAPTER_IDS)
def test_resume_never_repeats_the_checkpointed_epoch(adapter, model_name, tmp_path):
    args, cfg = _parse_args(adapter, model_name, tmp_path)
    total_epochs = 3
    for finished_epoch in range(total_epochs):
        path = RunPaths(str(tmp_path)).ensure().checkpoint(f"{model_name}_epoch{finished_epoch}")
        model, optimizer = _make_state(init_seed=1)
        adapter._save_checkpoint(path, finished_epoch, finished_epoch + 1, model, optimizer, cfg, args)
        fresh_model, fresh_optimizer = _make_state(init_seed=2)
        start_epoch, _ = adapter._load_checkpoint(path, fresh_model, fresh_optimizer)
        # Same loop shape as the training functions: range(start_epoch, epochs) must exclude every epoch up to the checkpointed one.
        ran_after = list(range(start_epoch, total_epochs))
        assert ran_after == list(range(finished_epoch + 1, total_epochs))
        assert not set(ran_after) & set(range(finished_epoch + 1))


@pytest.mark.parametrize("adapter, model_name", ADAPTERS, ids=ADAPTER_IDS)
def test_resumed_run_draws_the_same_random_numbers_as_an_uninterrupted_run(adapter, model_name, tmp_path):
    args, cfg = _parse_args(adapter, model_name, tmp_path)
    path = RunPaths(str(tmp_path)).ensure().checkpoint("x")

    set_seed(0, deterministic=False)
    model, optimizer = _make_state(init_seed=1)
    _step(model, optimizer)
    set_seed(7, deterministic=False)
    adapter._save_checkpoint(path, 0, 1, model, optimizer, cfg, args)
    # Uninterrupted continuation: one more dropout step, then fresh draws from every stream.
    _step(model, optimizer)
    uninterrupted_draws = _draws()

    set_seed(12345, deterministic=False)  # a resumed process starts from an unrelated RNG state
    resumed_model, resumed_optimizer = _make_state(init_seed=99)
    adapter._load_checkpoint(path, resumed_model, resumed_optimizer)
    _step(resumed_model, resumed_optimizer)
    resumed_draws = _draws()

    assert _weights_equal(model, resumed_model)
    assert resumed_draws == uninterrupted_draws


@pytest.mark.parametrize("adapter, model_name", ADAPTERS, ids=ADAPTER_IDS)
def test_load_checkpoint_restores_each_rng_stream_individually(adapter, model_name, tmp_path):
    args, cfg = _parse_args(adapter, model_name, tmp_path)
    path = RunPaths(str(tmp_path)).ensure().checkpoint("rng")
    model, optimizer = _make_state(init_seed=1)
    set_seed(3, deterministic=False)
    adapter._save_checkpoint(path, 0, 0, model, optimizer, cfg, args)
    expected_torch, expected_numpy, expected_python = _draws()

    set_seed(4, deterministic=False)
    other_model, other_optimizer = _make_state(init_seed=5)
    set_seed(4, deterministic=False)
    adapter._load_checkpoint(path, other_model, other_optimizer)
    assert torch.rand(4).tolist() == expected_torch
    assert np.random.rand(4).tolist() == expected_numpy
    assert [random.random() for _ in range(4)] == expected_python


@pytest.mark.parametrize("adapter, model_name", ADAPTERS, ids=ADAPTER_IDS)
def test_saved_checkpoint_contents(adapter, model_name, tmp_path):
    args, cfg = _parse_args(adapter, model_name, tmp_path, extra=["--seed", "5", "--epochs", "9"])
    path = RunPaths(str(tmp_path)).ensure().checkpoint("contents")
    model, optimizer = _make_state(init_seed=1)
    _step(model, optimizer)
    adapter._save_checkpoint(path, 4, 40, model, optimizer, cfg, args)

    # weights_only=False only so the test can inspect the raw payload independently of the adapter's loader.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    assert set(ckpt) == EXPECTED_CHECKPOINT_KEYS
    assert (ckpt["epoch"], ckpt["global_step"]) == (4, 40)
    assert {"torch", "numpy", "python"} <= set(ckpt["rng_state"])
    assert ("cuda" in ckpt["rng_state"]) == torch.cuda.is_available()
    assert ckpt["config"] == {**dataclasses.asdict(cfg), **vars(args)}
    assert ckpt["config"]["seed"] == 5 and ckpt["config"]["epochs"] == 9
    assert ckpt["config"]["model"] == model_name and ckpt["config"]["dataset"] == "qm9"
    assert set(ckpt["model_state_dict"]) == set(model.state_dict())
    assert "state" in ckpt["optimizer_state_dict"] and "param_groups" in ckpt["optimizer_state_dict"]


@pytest.mark.parametrize("adapter, model_name", ADAPTERS, ids=ADAPTER_IDS)
def test_save_checkpoint_snapshot_is_independent_of_later_training(adapter, model_name, tmp_path):
    args, cfg = _parse_args(adapter, model_name, tmp_path)
    path = RunPaths(str(tmp_path)).ensure().checkpoint("snap")
    model, optimizer = _make_state(init_seed=1)
    adapter._save_checkpoint(path, 0, 0, model, optimizer, cfg, args)
    saved_weights = {k: v.clone() for k, v in model.state_dict().items()}
    _step(model, optimizer)
    _step(model, optimizer)
    restored_model, restored_optimizer = _make_state(init_seed=2)
    adapter._load_checkpoint(path, restored_model, restored_optimizer)
    assert all(torch.equal(saved_weights[k], v) for k, v in restored_model.state_dict().items())


@pytest.mark.parametrize("adapter, model_name", ADAPTERS, ids=ADAPTER_IDS)
def test_load_checkpoint_missing_file_raises_file_not_found(adapter, model_name, tmp_path):
    model, optimizer = _make_state(init_seed=1)
    with pytest.raises(FileNotFoundError):
        adapter._load_checkpoint(str(tmp_path / "nope.pt"), model, optimizer)


@pytest.mark.parametrize("adapter, model_name", ADAPTERS, ids=ADAPTER_IDS)
def test_load_checkpoint_into_mismatched_model_raises_instead_of_silently_loading(adapter, model_name, tmp_path):
    args, cfg = _parse_args(adapter, model_name, tmp_path)
    path = RunPaths(str(tmp_path)).ensure().checkpoint("mismatch")
    model, optimizer = _make_state(init_seed=1)
    adapter._save_checkpoint(path, 0, 0, model, optimizer, cfg, args)
    wrong_model = torch.nn.Linear(5, 5)
    wrong_optimizer = torch.optim.SGD(wrong_model.parameters(), lr=0.1)
    with pytest.raises(RuntimeError):
        adapter._load_checkpoint(path, wrong_model, wrong_optimizer)


@pytest.mark.parametrize("adapter, model_name", ADAPTERS, ids=ADAPTER_IDS)
def test_adapters_use_the_shared_seeding_helpers(adapter, model_name):
    assert adapter.capture_rng_state is seeding.capture_rng_state
    assert adapter.restore_rng_state is seeding.restore_rng_state
    assert adapter.seed_worker is seeding.seed_worker
    assert adapter.dataloader_generator is seeding.dataloader_generator


def _patch_load_split(monkeypatch) -> list[tuple]:
    calls: list[tuple] = []
    dataset = TensorDataset(torch.arange(DATASET_SIZE))

    def fake_load_split(*args, **kwargs):
        calls.append((args, kwargs))
        return dataset

    monkeypatch.setattr("baselines.adapters.dataset_loading.load_split", fake_load_split)
    return calls


def _config(seed: int = 3, num_workers: int = 0, batch_size: int = 4) -> CommonConfig:
    return CommonConfig(
        seed=seed, dataset="qm9", split="molecule_95_5", run_dir="unused", data_root="/fake/root",
        device="cpu", num_workers=num_workers, batch_size=batch_size,
    )


def _epoch_orders(loader, epochs: int) -> list[list[int]]:
    return [torch.cat([batch[0] for batch in loader]).tolist() for _ in range(epochs)]


@pytest.mark.parametrize("adapter, model_name", ADAPTERS, ids=ADAPTER_IDS)
def test_build_dataloader_forwards_config_to_load_split(adapter, model_name, monkeypatch):
    calls = _patch_load_split(monkeypatch)
    adapter._build_dataloader(_config(), "val")
    assert calls == [(("qm9", "molecule_95_5", "val", "/fake/root"), {})]


@pytest.mark.parametrize("adapter, model_name", ADAPTERS, ids=ADAPTER_IDS)
def test_build_dataloader_shuffles_only_the_train_split(adapter, model_name, monkeypatch):
    _patch_load_split(monkeypatch)
    assert isinstance(adapter._build_dataloader(_config(), "train").sampler, RandomSampler)
    for split in ("val", "test"):
        loader = adapter._build_dataloader(_config(), split)
        assert isinstance(loader.sampler, SequentialSampler)
        assert _epoch_orders(loader, 2) == [list(range(DATASET_SIZE))] * 2


@pytest.mark.parametrize("adapter, model_name", ADAPTERS, ids=ADAPTER_IDS)
def test_build_dataloader_train_order_is_reproducible_from_the_config_seed(adapter, model_name, monkeypatch):
    _patch_load_split(monkeypatch)
    torch.manual_seed(111)
    first = _epoch_orders(adapter._build_dataloader(_config(seed=3), "train"), 3)
    torch.manual_seed(222)
    second = _epoch_orders(adapter._build_dataloader(_config(seed=3), "train"), 3)
    other_seed = _epoch_orders(adapter._build_dataloader(_config(seed=4), "train"), 3)
    assert first == second
    assert first != other_seed
    assert all(sorted(order) == list(range(DATASET_SIZE)) for order in first)
    assert first[0] != list(range(DATASET_SIZE))
    assert first[0] != first[1]


@pytest.mark.parametrize("adapter, model_name", ADAPTERS, ids=ADAPTER_IDS)
def test_build_dataloader_attaches_seeded_generator_and_worker_init(adapter, model_name, monkeypatch):
    _patch_load_split(monkeypatch)
    loader = adapter._build_dataloader(_config(seed=17), "train")
    assert loader.generator.initial_seed() == 17
    assert loader.worker_init_fn is seeding.seed_worker


@pytest.mark.parametrize("adapter, model_name", ADAPTERS, ids=ADAPTER_IDS)
def test_build_dataloader_zero_workers_sets_no_worker_only_options(adapter, model_name, monkeypatch):
    _patch_load_split(monkeypatch)
    loader = adapter._build_dataloader(_config(num_workers=0), "train")
    assert loader.num_workers == 0
    assert loader.persistent_workers is False
    assert loader.prefetch_factor is None
    assert loader.batch_size == 4
    assert [batch[0].shape[0] for batch in loader] == [4, 4, 4, 4]


@pytest.mark.parametrize("adapter, model_name", ADAPTERS, ids=ADAPTER_IDS)
def test_build_dataloader_with_workers_enables_persistent_workers_and_prefetch(adapter, model_name, monkeypatch):
    _patch_load_split(monkeypatch)
    loader = adapter._build_dataloader(_config(num_workers=2), "train")
    assert loader.num_workers == 2
    assert loader.persistent_workers is True
    assert loader.prefetch_factor == 2


@pytest.mark.parametrize("adapter, model_name", ADAPTERS, ids=ADAPTER_IDS)
def test_build_dataloader_batch_size_comes_from_config(adapter, model_name, monkeypatch):
    _patch_load_split(monkeypatch)
    loader = adapter._build_dataloader(_config(batch_size=5), "test")
    assert loader.batch_size == 5
    assert [batch[0].shape[0] for batch in loader] == [5, 5, 5, 1]
