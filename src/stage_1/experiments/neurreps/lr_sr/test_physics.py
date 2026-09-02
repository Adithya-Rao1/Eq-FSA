from __future__ import annotations

import math

import torch

from stage_1.experiments.neurreps.lr_sr.physics import (
    COULOMB_KE, compute_energy_and_forces, ewald_energy, total_energy, wca_energy,
)

DTYPE = torch.double
torch.set_default_dtype(DTYPE)


def _random_config(n=16, box=16.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    pos = (torch.rand(1, n, 3, generator=g, dtype=DTYPE) - 0.5) * box
    charges = torch.cat([torch.ones(n // 2, dtype=DTYPE), -torch.ones(n - n // 2, dtype=DTYPE)])[None]
    V = (torch.eye(3, dtype=DTYPE) * box).unsqueeze(0)
    return pos, charges, V


def test_ewald_translation_invariance():
    pos, q, V = _random_config()
    shift = torch.randn(1, 1, 3, dtype=DTYPE) * 5.0
    e0 = ewald_energy(pos, q, V)
    e1 = ewald_energy(pos + shift, q, V)
    torch.testing.assert_close(e0, e1, atol=1e-8, rtol=1e-8)


def test_wca_translation_invariance():
    pos, _, V = _random_config()
    shift = torch.randn(1, 1, 3, dtype=DTYPE) * 5.0
    e0 = wca_energy(pos, V)
    e1 = wca_energy(pos + shift, V)
    torch.testing.assert_close(e0, e1, atol=1e-10, rtol=1e-10)


def test_ewald_periodic_image_invariance():
    pos, q, V = _random_config()
    pos2 = pos.clone()
    pos2[0, 0] = pos2[0, 0] + V[0, 0]  
    e0 = ewald_energy(pos, q, V)
    e1 = ewald_energy(pos2, q, V)
    torch.testing.assert_close(e0, e1, atol=1e-7, rtol=1e-7)


def test_force_matches_finite_difference():
    pos, q, V = _random_config(n=8, seed=3)
    _, F = compute_energy_and_forces(pos, q, V)

    eps = 1e-5
    atom, dim = 2, 1
    pos_plus = pos.clone(); pos_plus[0, atom, dim] += eps
    pos_minus = pos.clone(); pos_minus[0, atom, dim] -= eps
    e_plus = total_energy(pos_plus, q, V)
    e_minus = total_energy(pos_minus, q, V)
    fd_force = -(e_plus - e_minus) / (2 * eps)

    torch.testing.assert_close(F[0, atom, dim], fd_force[0], atol=1e-4, rtol=1e-3)


def test_net_force_is_zero():
    pos, q, V = _random_config(n=32, seed=5)
    _, F = compute_energy_and_forces(pos, q, V)
    net_force = F.sum(dim=1)
    torch.testing.assert_close(net_force, torch.zeros_like(net_force), atol=1e-6, rtol=0)


def test_ewald_approaches_non_periodic_coulomb_for_large_box():
    r = 4.0
    pos = torch.tensor([[[0.0, 0.0, 0.0], [r, 0.0, 0.0]]], dtype=DTYPE)
    q = torch.tensor([[1.0, -1.0]], dtype=DTYPE)
    box = 60.0
    V = (torch.eye(3, dtype=DTYPE) * box).unsqueeze(0)
    e = ewald_energy(pos, q, V, alpha=0.3, real_cutoff=box / 2 - 0.5, n_max=25)
    expected = -COULOMB_KE / r
    rel_err = abs((e.item() - expected) / expected)
    assert rel_err < 0.02, f"relative error {rel_err} too large vs. non-periodic Coulomb limit"


def test_wca_purely_repulsive_and_zero_beyond_rmin():
    sigma, epsilon = 3.0, 1.0
    r_min = (2.0 ** (1.0 / 6.0)) * sigma
    box = 40.0
    V = (torch.eye(3, dtype=DTYPE) * box).unsqueeze(0)

    for r, expect_positive in [(sigma * 0.9, True), (r_min - 0.1, True), (r_min + 0.1, False)]:
        pos = torch.tensor([[[0.0, 0.0, 0.0], [r, 0.0, 0.0]]], dtype=DTYPE)
        e = wca_energy(pos, V, sigma=sigma, epsilon=epsilon)
        if expect_positive:
            assert e.item() > 0, f"expected positive repulsive energy at r={r}, got {e.item()}"
        else:
            torch.testing.assert_close(e, torch.zeros_like(e), atol=1e-10, rtol=0)


def test_wca_matches_known_lj_repulsive_formula():
    sigma, epsilon, r = 2.0, 1.5, 2.2
    box = 40.0
    V = (torch.eye(3, dtype=DTYPE) * box).unsqueeze(0)
    pos = torch.tensor([[[0.0, 0.0, 0.0], [r, 0.0, 0.0]]], dtype=DTYPE)
    e = wca_energy(pos, V, sigma=sigma, epsilon=epsilon)
    sr6 = (sigma / r) ** 6
    expected = 4.0 * epsilon * (sr6 ** 2 - sr6) + epsilon
    torch.testing.assert_close(e, torch.tensor([expected], dtype=DTYPE), atol=1e-10, rtol=1e-10)
