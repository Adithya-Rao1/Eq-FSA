from __future__ import annotations

import torch

from utils.pw_bases import make_wavevectors

DEFAULT_BOX = 20.0


def make_wavevector_grid(n_max: int, box: float = DEFAULT_BOX):
    V = (torch.eye(3) * box).unsqueeze(0)
    kvecs, indices, mode_mask = make_wavevectors(V, n_max, e_cutoff=None)
    nonzero = indices.abs().sum(dim=-1) > 0
    return kvecs[0][nonzero]


def generate_batch(
    batch_size: int, n_atoms: int, kvecs: torch.Tensor | None, sigmas: torch.Tensor,
    box: float = DEFAULT_BOX, box_range: tuple[float, float] | None = None,
    n_max: int | None = None, generator: torch.Generator | None = None,
):
    n_pos = n_atoms // 2
    charges_1d = torch.cat([torch.ones(n_pos), -torch.ones(n_atoms - n_pos)])
    charges = charges_1d.unsqueeze(0).expand(batch_size, -1).clone()
    for b in range(batch_size):
        perm = torch.randperm(n_atoms, generator=generator)
        charges[b] = charges_1d[perm]

    mask = torch.ones(batch_size, n_atoms, dtype=torch.bool)

    if box_range is not None:
        assert n_max is not None, "generate_batch: n_max is required when box_range is given"
        box_min, box_max = box_range
        box_per_sample = torch.rand(batch_size, generator=generator) * (box_max - box_min) + box_min
        positions = torch.rand(batch_size, n_atoms, 3, generator=generator) * box_per_sample.view(-1, 1, 1)
        V = torch.eye(3).unsqueeze(0) * box_per_sample.view(-1, 1, 1)

        kvecs_b, indices, _ = make_wavevectors(V, n_max, e_cutoff=None)
        nonzero = indices.abs().sum(dim=-1) > 0
        kvecs_b = kvecs_b[:, nonzero]  

        phase = torch.einsum("bni,bmi->bnm", positions, kvecs_b)
        S_re = torch.einsum("bn,bnm->bm", charges, torch.cos(phase))
        S_im = torch.einsum("bn,bnm->bm", charges, torch.sin(phase))
        S_k2 = S_re**2 + S_im**2  

        k2 = (kvecs_b**2).sum(dim=-1)
        weights = torch.exp(-k2.unsqueeze(1) * (sigmas.view(1, -1, 1) ** 2) / 2.0)  
        E_LR = torch.einsum("bkm,bm->bk", weights, S_k2)  
        return charges, positions, V, mask, E_LR

    positions = torch.rand(batch_size, n_atoms, 3, generator=generator) * box
    V = (torch.eye(3) * box).unsqueeze(0).expand(batch_size, 3, 3).contiguous()

    phase = torch.einsum("bni,mi->bnm", positions, kvecs)
    S_re = torch.einsum("bn,bnm->bm", charges, torch.cos(phase))
    S_im = torch.einsum("bn,bnm->bm", charges, torch.sin(phase))
    S_k2 = S_re**2 + S_im**2  

    k2 = (kvecs**2).sum(dim=-1)
    # w_k(sigma) = exp(-k^2 sigma^2 / 2), summed over modes
    weights = torch.exp(-k2.unsqueeze(0) * (sigmas.unsqueeze(1) ** 2) / 2.0) 
    E_LR = torch.einsum("km,bm->bk", weights, S_k2)  

    return charges, positions, V, mask, E_LR
