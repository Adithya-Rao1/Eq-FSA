from __future__ import annotations

import math

import torch

from utils.pw_bases import make_wavevectors, minimum_image

COULOMB_KE = 14.399645351950548

def ewald_real_space(
    positions: torch.Tensor, charges: torch.Tensor, V: torch.Tensor,
    alpha: float, real_cutoff: float,
) -> torch.Tensor:
    B, N, _ = positions.shape
    dx = positions.unsqueeze(1) - positions.unsqueeze(2)
    dx_mic = minimum_image(dx.reshape(B, N * N, 3), V).reshape(B, N, N, 3)
    r = dx_mic.norm(dim=-1)

    self_mask = ~torch.eye(N, dtype=torch.bool, device=positions.device).unsqueeze(0)
    within_cutoff = (r <= real_cutoff) & self_mask
    r_safe = r.clamp(min=1e-6)

    qq = charges.unsqueeze(2) * charges.unsqueeze(1)
    contrib = torch.where(
        within_cutoff, qq * torch.erfc(alpha * r_safe) / r_safe, torch.zeros_like(r_safe),
    )
    return COULOMB_KE * 0.5 * contrib.sum(dim=(1, 2))


def ewald_reciprocal_space(
    positions: torch.Tensor, charges: torch.Tensor, V: torch.Tensor,
    alpha: float, n_max: int,
) -> torch.Tensor:
    B = positions.shape[0]
    kvecs, indices, mode_mask = make_wavevectors(V, n_max, e_cutoff=None)  # (B,M,3), (M,3), (B,M)

    k2 = (kvecs ** 2).sum(dim=-1)
    nonzero = (indices.abs().sum(dim=-1) > 0).unsqueeze(0).expand(B, -1)  # exclude the k=0 mode
    active = nonzero & mode_mask

    phase = torch.einsum("bni,bmi->bnm", positions, kvecs)
    S_re = torch.einsum("bn,bnm->bm", charges, torch.cos(phase))
    S_im = torch.einsum("bn,bnm->bm", charges, torch.sin(phase))
    S2 = S_re ** 2 + S_im ** 2

    k2_safe = k2.clamp(min=1e-12)
    weight = torch.exp(-k2_safe / (4.0 * alpha ** 2)) / k2_safe
    weight = torch.where(active, weight, torch.zeros_like(weight))

    volume = torch.linalg.det(V).abs()
    return COULOMB_KE * (2.0 * math.pi / volume) * (weight * S2).sum(dim=-1)


def ewald_self_energy(charges: torch.Tensor, alpha: float) -> torch.Tensor:
    return -COULOMB_KE * (alpha / math.sqrt(math.pi)) * (charges ** 2).sum(dim=-1)


def ewald_energy(
    positions: torch.Tensor, charges: torch.Tensor, V: torch.Tensor,
    alpha: float = 0.6, real_cutoff: float = 5.5, n_max: int = 15,
) -> torch.Tensor:
    return (
        ewald_real_space(positions, charges, V, alpha, real_cutoff)
        + ewald_reciprocal_space(positions, charges, V, alpha, n_max)
        + ewald_self_energy(charges, alpha)
    )


def wca_energy(
    positions: torch.Tensor, V: torch.Tensor, sigma: float = 3.0, epsilon: float = 1.0,
) -> torch.Tensor:
    B, N, _ = positions.shape
    r_min = (2.0 ** (1.0 / 6.0)) * sigma

    dx = positions.unsqueeze(1) - positions.unsqueeze(2)
    dx_mic = minimum_image(dx.reshape(B, N * N, 3), V).reshape(B, N, N, 3)
    r = dx_mic.norm(dim=-1)

    self_mask = ~torch.eye(N, dtype=torch.bool, device=positions.device).unsqueeze(0)
    within_cutoff = (r <= r_min) & self_mask
    r_safe = r.clamp(min=1e-6)

    sr6 = (sigma / r_safe) ** 6
    lj = 4.0 * epsilon * (sr6 ** 2 - sr6) + epsilon
    contrib = torch.where(within_cutoff, lj, torch.zeros_like(lj))
    return 0.5 * contrib.sum(dim=(1, 2))


def total_energy(
    positions: torch.Tensor, charges: torch.Tensor, V: torch.Tensor,
    alpha: float = 0.6, real_cutoff: float = 5.5, n_max: int = 15,
    sigma: float = 3.0, epsilon: float = 1.0,
) -> torch.Tensor:
    return (
        ewald_energy(positions, charges, V, alpha=alpha, real_cutoff=real_cutoff, n_max=n_max)
        + wca_energy(positions, V, sigma=sigma, epsilon=epsilon)
    )


def compute_energy_and_forces(
    positions: torch.Tensor, charges: torch.Tensor, V: torch.Tensor, **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = positions.clone().requires_grad_(True)
    U = total_energy(x, charges, V, **kwargs)
    (grad_x,) = torch.autograd.grad(U.sum(), x)
    return U.detach(), -grad_x.detach()
