# Tests for src/baselines/provenance/seeding.py
# PyTorch reproducibility notes, pytorch.org/docs/stable/notes/randomness.html#dataloader
# (worker seed = base_seed + worker_id --> inside a worker as torch.initial_seed()).

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from baselines.provenance.seeding import (
    capture_rng_state,
    dataloader_generator,
    restore_rng_state,
    seed_worker,
    set_seed,
)

N_DRAWS = 5
DATASET_SIZE = 20


def _draws() -> tuple[list[float], list[float], list[float]]:
    return (
        torch.rand(N_DRAWS).tolist(),
        np.random.rand(N_DRAWS).tolist(),
        [random.random() for _ in range(N_DRAWS)],
    )


def _snapshot_torch_flags() -> tuple[bool, bool, bool, bool]:
    return (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
    )


def _restore_torch_flags(snapshot: tuple[bool, bool, bool, bool]) -> None:
    torch.use_deterministic_algorithms(snapshot[0], warn_only=snapshot[1])
    torch.backends.cudnn.deterministic = snapshot[2]
    torch.backends.cudnn.benchmark = snapshot[3]


class _WorkerStreamDataset(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, i: int) -> torch.Tensor:
        info = torch.utils.data.get_worker_info()
        return torch.tensor([float(info.id), float(np.random.rand()), random.random()], dtype=torch.float64)


def _worker_rows(seed: int) -> torch.Tensor:
    # batch_size=1 with 2 workers --> batch i is produced by worker i % 2, so rows alternate worker ids 0, 1, 0, 1.
    loader = DataLoader(
        _WorkerStreamDataset(), batch_size=1, num_workers=2,
        worker_init_fn=seed_worker, generator=dataloader_generator(seed),
    )
    return torch.cat(list(loader), dim=0)


def _epoch_orders(loader: DataLoader, epochs: int) -> list[list[int]]:
    return [torch.cat([b[0] for b in loader]).tolist() for _ in range(epochs)]


def test_set_seed_makes_two_runs_bit_identical():
    set_seed(123, deterministic=False)
    run_a = _draws()
    set_seed(123, deterministic=False)
    run_b = _draws()
    assert run_a == run_b


def test_set_seed_different_seeds_differ_in_every_stream():
    set_seed(0, deterministic=False)
    run_a = _draws()
    set_seed(1, deterministic=False)
    run_b = _draws()
    for stream_a, stream_b in zip(run_a, run_b):
        assert stream_a != stream_b


def test_set_seed_streams_match_direct_library_seeding():
    set_seed(77, deterministic=False)
    torch_draw, numpy_draw, python_draw = _draws()
    torch.manual_seed(77)
    assert torch_draw == torch.rand(N_DRAWS).tolist()
    np.random.seed(77)
    assert numpy_draw == np.random.rand(N_DRAWS).tolist()
    random.seed(77)
    assert python_draw == [random.random() for _ in range(N_DRAWS)]


def test_set_seed_deterministic_true_sets_flags_and_cublas_default(monkeypatch):
    # setenv-then-delenv so monkeypatch records the pre-test state and undoes whatever set_seed writes.
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "placeholder")
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG")
    snapshot = _snapshot_torch_flags()
    try:
        set_seed(0, deterministic=True)
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    finally:
        _restore_torch_flags(snapshot)


def test_set_seed_does_not_override_existing_cublas_config(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    snapshot = _snapshot_torch_flags()
    try:
        set_seed(0, deterministic=True)
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"
    finally:
        _restore_torch_flags(snapshot)


def test_set_seed_deterministic_false_leaves_flags_and_env_untouched(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "placeholder")
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG")
    before = _snapshot_torch_flags()
    set_seed(0, deterministic=False)
    assert _snapshot_torch_flags() == before
    assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ


def test_capture_restore_round_trip_reproduces_next_draws():
    set_seed(5, deterministic=False)
    _draws()  # advance every stream so the captured state is not just the freshly seeded one
    state = capture_rng_state()
    expected = _draws()
    _draws()  # scramble further
    restore_rng_state(state)
    assert _draws() == expected


def test_restore_rng_state_can_be_applied_repeatedly_from_one_capture():
    set_seed(5, deterministic=False)
    state = capture_rng_state()
    restore_rng_state(state)
    first = _draws()
    restore_rng_state(state)
    second = _draws()
    assert first == second


def test_capture_rng_state_is_a_snapshot_not_a_view():
    set_seed(9, deterministic=False)
    state = capture_rng_state()
    _draws()
    restore_rng_state(state)
    after_restore = _draws()
    set_seed(9, deterministic=False)
    assert after_restore == _draws()


def test_capture_rng_state_keys_cpu_streams_always_present():
    state = capture_rng_state()
    assert {"torch", "numpy", "python"} <= set(state)
    assert ("cuda" in state) == torch.cuda.is_available()


def test_restore_rng_state_without_cuda_key_is_accepted():
    set_seed(2, deterministic=False)
    state = capture_rng_state()
    state.pop("cuda", None)
    expected = _draws()
    restore_rng_state(state)
    assert _draws() == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA RNG round-trip needs a GPU")
def test_capture_restore_round_trip_cuda():
    set_seed(5, deterministic=False)
    state = capture_rng_state()
    expected = torch.rand(N_DRAWS, device="cuda").cpu().tolist()
    torch.rand(N_DRAWS, device="cuda")
    restore_rng_state(state)
    assert torch.rand(N_DRAWS, device="cuda").cpu().tolist() == expected


def test_dataloader_generator_is_seeded_torch_generator():
    gen = dataloader_generator(11)
    assert isinstance(gen, torch.Generator)
    assert gen.initial_seed() == 11


def test_dataloader_generator_independent_of_global_rng():
    gen_a = dataloader_generator(4)
    torch.manual_seed(1000)
    torch.rand(3)
    gen_b = dataloader_generator(4)
    assert torch.randperm(DATASET_SIZE, generator=gen_a).tolist() == torch.randperm(DATASET_SIZE, generator=gen_b).tolist()


def test_dataloader_generator_gives_identical_shuffle_order_across_loaders():
    dataset = TensorDataset(torch.arange(DATASET_SIZE))
    # Different global seeds prove the order comes only from the explicit generator.
    torch.manual_seed(111)
    loader_a = DataLoader(dataset, batch_size=4, shuffle=True, generator=dataloader_generator(3))
    torch.manual_seed(222)
    loader_b = DataLoader(dataset, batch_size=4, shuffle=True, generator=dataloader_generator(3))
    order_a, order_b = _epoch_orders(loader_a, 2), _epoch_orders(loader_b, 2)
    assert order_a == order_b
    assert sorted(order_a[0]) == list(range(DATASET_SIZE))
    assert order_a[0] != list(range(DATASET_SIZE))


def test_dataloader_generator_different_seeds_give_different_order():
    dataset = TensorDataset(torch.arange(DATASET_SIZE))
    loader_a = DataLoader(dataset, batch_size=4, shuffle=True, generator=dataloader_generator(0))
    loader_b = DataLoader(dataset, batch_size=4, shuffle=True, generator=dataloader_generator(1))
    assert _epoch_orders(loader_a, 1) != _epoch_orders(loader_b, 1)


def test_dataloader_generator_advances_between_epochs():
    dataset = TensorDataset(torch.arange(DATASET_SIZE))
    loader = DataLoader(dataset, batch_size=4, shuffle=True, generator=dataloader_generator(0))
    epoch_0, epoch_1 = _epoch_orders(loader, 2)
    assert epoch_0 != epoch_1


def test_seed_worker_seeds_numpy_and_python_from_torch_initial_seed():
    torch.manual_seed(1234)
    seed_worker(0)
    numpy_draw, python_draw = np.random.rand(N_DRAWS).tolist(), [random.random() for _ in range(N_DRAWS)]
    np.random.seed(1234)
    random.seed(1234)
    assert numpy_draw == np.random.rand(N_DRAWS).tolist()
    assert python_draw == [random.random() for _ in range(N_DRAWS)]


def test_seed_worker_reduces_seed_modulo_2_pow_32_for_numpy():
    # np.random.seed rejects values >= 2**32, and 2**40 % 2**32 == 0, so this should seed numpy/python with 9.
    torch.manual_seed(2**40 + 9)
    seed_worker(0)
    numpy_draw, python_draw = np.random.rand(N_DRAWS).tolist(), [random.random() for _ in range(N_DRAWS)]
    np.random.seed(9)
    random.seed(9)
    assert numpy_draw == np.random.rand(N_DRAWS).tolist()
    assert python_draw == [random.random() for _ in range(N_DRAWS)]


def test_seed_worker_distinct_torch_seeds_give_distinct_streams():
    # Inside a real worker, torch.initial_seed() is base_seed + worker_id; emulate two workers of one loader.
    base_seed = 50
    draws = []
    for worker_id in (0, 1):
        torch.manual_seed(base_seed + worker_id)
        seed_worker(worker_id)
        draws.append((np.random.rand(N_DRAWS).tolist(), [random.random() for _ in range(N_DRAWS)]))
    assert draws[0][0] != draws[1][0]
    assert draws[0][1] != draws[1][1]


def test_seed_worker_with_real_multiprocess_dataloader_is_reproducible_and_distinct_per_worker():
    rows_a, rows_b = _worker_rows(seed=7), _worker_rows(seed=7)
    assert rows_a.tolist() == rows_b.tolist()
    assert rows_a[:, 0].tolist() == [0.0, 1.0, 0.0, 1.0]
    # Row 0 / row 1 are each worker's first draw; row 2 / row 3 are their second.
    assert rows_a[0, 1] != rows_a[1, 1]
    assert rows_a[0, 2] != rows_a[1, 2]
    assert rows_a[0, 1] != rows_a[2, 1]
    assert rows_a[1, 2] != rows_a[3, 2]


def test_seed_worker_with_real_multiprocess_dataloader_changes_with_loader_seed():
    assert _worker_rows(seed=7).tolist() != _worker_rows(seed=8).tolist()
