"""Data generation for the RSA-vs-EqFSA-LR encoder comparison: random point charges, target is
the Gaussian-damped reciprocal-space energy E_LR(sigma) at several smearing widths sigma (not raw
per-mode |S(k)|^2, which is shot-noise-dominated -- std/mean ~1 regardless of N, confirmed
empirically in this session's earlier capacity experiment). Summing over many k-modes at each
sigma averages out that shot noise, giving a well-conditioned, physically meaningful target (this
is literally the family of quantities RSA's own Eq. 5 computes, just probed at several lengthscales).
"""
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
    batch_size: int, n_atoms: int, kvecs: torch.Tensor, sigmas: torch.Tensor,
    box: float = DEFAULT_BOX, generator: torch.Generator | None = None,
):
    """Returns charges (B,N), positions (B,N,3), V (B,3,3), mask (B,N), and E_LR (B,K) -- the
    Gaussian-damped reciprocal-space energy at each of K sigma values."""
    n_pos = n_atoms // 2
    charges_1d = torch.cat([torch.ones(n_pos), -torch.ones(n_atoms - n_pos)])
    charges = charges_1d.unsqueeze(0).expand(batch_size, -1).clone()
    for b in range(batch_size):
        perm = torch.randperm(n_atoms, generator=generator)
        charges[b] = charges_1d[perm]

    positions = torch.rand(batch_size, n_atoms, 3, generator=generator) * box
    V = (torch.eye(3) * box).unsqueeze(0).expand(batch_size, 3, 3).contiguous()
    mask = torch.ones(batch_size, n_atoms, dtype=torch.bool)

    phase = torch.einsum("bni,mi->bnm", positions, kvecs)
    S_re = torch.einsum("bn,bnm->bm", charges, torch.cos(phase))
    S_im = torch.einsum("bn,bnm->bm", charges, torch.sin(phase))
    S_k2 = S_re**2 + S_im**2  # (B,M)

    k2 = (kvecs**2).sum(dim=-1)  # (M,)
    # w_k(sigma) = exp(-k^2 sigma^2 / 2), summed over modes -> (B,K)
    weights = torch.exp(-k2.unsqueeze(0) * (sigmas.unsqueeze(1) ** 2) / 2.0)  # (K,M)
    E_LR = torch.einsum("km,bm->bk", weights, S_k2)  # (B,K)

    return charges, positions, V, mask, E_LR
