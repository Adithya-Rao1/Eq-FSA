from __future__ import annotations

import torch

from stage_1.experiments.neurreps.lr_sr.dataset import (
    BOX_MAX, BOX_MIN, MIN_SEPARATION, N_ATOMS, N_NEGATIVE, N_POSITIVE, NEGATIVE_TYPE_IDX,
    POSITIVE_TYPE_IDX, generate_configuration, lr_sr_collate_fn,
)


def _min_pairwise_separation(x: torch.Tensor, box: float) -> float:
    d = x.unsqueeze(0) - x.unsqueeze(1)
    d = d - box * torch.round(d / box)
    r = d.norm(dim=-1)
    r.fill_diagonal_(999.0)
    return r.min().item()


def test_generated_configs_respect_min_separation_and_charge_balance():
    for seed in range(10):
        cfg = generate_configuration(seed=seed)
        assert BOX_MIN <= cfg["box"] <= BOX_MAX
        assert cfg["x"].shape == (N_ATOMS, 3)
        assert _min_pairwise_separation(cfg["x"], cfg["box"]) >= MIN_SEPARATION - 1e-4
        n_pos = (cfg["types"] == POSITIVE_TYPE_IDX).sum().item()
        n_neg = (cfg["types"] == NEGATIVE_TYPE_IDX).sum().item()
        assert n_pos == N_POSITIVE and n_neg == N_NEGATIVE
        assert torch.isfinite(cfg["energy"]).all()
        assert torch.isfinite(cfg["forces"]).all()


def test_collate_fn_shapes():
    batch = [generate_configuration(seed=i) for i in range(4)]
    out = lr_sr_collate_fn(batch)
    assert out["x"].shape == (4, N_ATOMS, 3)
    assert out["types"].shape == (4, N_ATOMS)
    assert out["V"].shape == (4, 3, 3)
    assert out["energy"].shape == (4,)
    assert out["forces"].shape == (4, N_ATOMS, 3)
    assert out["mask"].all()
