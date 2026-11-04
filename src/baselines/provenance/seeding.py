from __future__ import annotations

# Worker-seeding pattern follows PyTorch's reproducibility docs at pytorch.org/docs/stable/notes/randomness.html#dataloader.

import os
import random
from typing import Any

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # deterministic kernels for every op
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def dataloader_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def capture_rng_state() -> dict[str, Any]:
    # numpy's state holds an ndarray, which torch.load(weights_only=True) (the default since torch 2.6) refuses to unpickle, so it is stored as builtin types; np.random.set_state restores the list form bit-exactly.
    algorithm, keys, pos, has_gauss, cached_gaussian = np.random.get_state()
    state: dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "numpy": (algorithm, keys.tolist(), int(pos), int(has_gauss), float(cached_gaussian)),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
