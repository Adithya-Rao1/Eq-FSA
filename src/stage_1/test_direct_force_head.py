import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest
import torch

torch.manual_seed(0)
torch.set_default_dtype(torch.double)

from base_tests import (
    DTYPE, NUM_ATOM_TYPES,
    random_rotation_matrix, rotate_x, rotate_V,
    make_full_species, make_lattice, make_mol_props, make_edge_feats,
)
from utils.se3_utils import RadialNetwork, get_spherical_harmonics
from stage_1.spice_dataset import _EDGE_FEAT_DIM
from stage_1.model import SH1_TO_CARTESIAN
from stage_1.train_spice import SPICEEnergyForceModel, compute_energy_and_forces, spice_loss

FEATURE_DIM = 8
HIDDEN_DIM = 16
MAX_DEGREE = 2
NUM_LAYERS = 2
N_MASTER_NODES = 2
EDGE_CUTOFF = 100.0
N_MAX = 1

ATOL = 1e-6
RTOL = 1e-6


def build_model(seed: int = 123) -> SPICEEnergyForceModel:
    torch.manual_seed(seed)
    m = SPICEEnergyForceModel(
        radial_net=RadialNetwork,
        num_atom_types=NUM_ATOM_TYPES,
        max_degree=MAX_DEGREE,
        num_layers=NUM_LAYERS,
        feature_dim=FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
        edge_cutoff=EDGE_CUTOFF,
        n_max=N_MAX,
        n_master_nodes=N_MASTER_NODES,
        backbone="lr_restructure",
        num_broadcast_heads=2,
        edge_feat_dim=_EDGE_FEAT_DIM,
    )
    return m.to(DTYPE)


def make_batch(B: int, N: int, seed: int = 42):
    g = torch.Generator().manual_seed(seed)
    x, types_, masses, mask = make_full_species(B, N, g)
    V = make_lattice(B, g)
    mol_props = make_mol_props(B, g)
    edge_feats = make_edge_feats(B, N, g)
    return x, types_, masses, mask, V, mol_props, edge_feats


def test_sh1_to_cartesian_basis_convention():
    n = torch.randn(2000, 3, dtype=DTYPE)
    n = n / n.norm(dim=-1, keepdim=True)
    Y1 = get_spherical_harmonics(1, n)
    recovered = torch.einsum("ij,nj->ni", SH1_TO_CARTESIAN.to(DTYPE), Y1)
    torch.testing.assert_close(recovered, n, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("full", [False, True])
def test_direct_force_rotation_equivariance(full):
    model = build_model()
    model.eval()
    B, N = 2, 6
    x, types_, masses, mask, V, mol_props, edge_feats = make_batch(B, N)

    R = random_rotation_matrix()
    x_rot = rotate_x(x, R)
    V_rot = rotate_V(V, R)

    with torch.no_grad():
        _, F_raw = model(x, types_, masses, mask, V, mol_props, edge_feats, force_mode="direct")
        _, F_rot = model(x_rot, types_, masses, mask, V_rot, mol_props, edge_feats, force_mode="direct")

    F_expected = torch.einsum("ij,bnj->bni", R, F_raw)
    m = mask.unsqueeze(-1).to(DTYPE)
    torch.testing.assert_close(F_rot * m, F_expected * m, atol=ATOL, rtol=RTOL)


def test_net_force_is_zero():
    model = build_model()
    model.eval()
    B, N = 3, 7
    x, types_, masses, mask, V, mol_props, edge_feats = make_batch(B, N, seed=7)
    with torch.no_grad():
        _, F = model(x, types_, masses, mask, V, mol_props, edge_feats, force_mode="direct")
    net = (F * mask.unsqueeze(-1).to(DTYPE)).sum(dim=1)
    torch.testing.assert_close(net, torch.zeros_like(net), atol=1e-8, rtol=0.0)


def _net_torque(x, F, masses, mask, V):
    from stage_1.model import masked_weighted_com
    r_com, _ = masked_weighted_com(x, masses, mask, V=None)
    r_rel = (x - r_com.unsqueeze(1)) * mask.unsqueeze(-1).to(DTYPE)
    return (torch.cross(r_rel, F, dim=-1) * mask.unsqueeze(-1).to(DTYPE)).sum(dim=1)


def test_net_torque_is_zero_random_geometry():
    model = build_model()
    model.eval()
    B, N = 3, 7
    x, types_, masses, mask, V, mol_props, edge_feats = make_batch(B, N, seed=11)
    with torch.no_grad():
        _, F = model(x, types_, masses, mask, V, mol_props, edge_feats, force_mode="direct")
    tau = _net_torque(x, F, masses, mask, V)
    torch.testing.assert_close(tau, torch.zeros_like(tau), atol=5e-4, rtol=0.0)


def test_net_torque_is_zero_near_linear_geometry():
    model = build_model()
    model.eval()
    B, N = 1, 5
    g = torch.Generator().manual_seed(99)
    _, types_, masses, mask, V, mol_props, edge_feats = make_batch(B, N, seed=99)
    offsets = torch.linspace(-2.0, 2.0, N, dtype=DTYPE).view(1, N, 1)
    x = torch.zeros(B, N, 3, dtype=DTYPE)
    x[..., 0:1] = offsets
    x = x + 1e-5 * torch.randn(B, N, 3, generator=g, dtype=DTYPE) 

    with torch.no_grad():
        _, F = model(x, types_, masses, mask, V, mol_props, edge_feats, force_mode="direct")
    assert torch.isfinite(F).all()
    tau = _net_torque(x, F, masses, mask, V)
    torch.testing.assert_close(tau, torch.zeros_like(tau), atol=1e-4, rtol=0.0)


def test_direct_mode_never_calls_autograd_grad(monkeypatch):
    model = build_model()
    model.train()
    B, N = 2, 5
    x, types_, masses, mask, V, mol_props, edge_feats = make_batch(B, N, seed=5)

    calls = {"n": 0}
    orig_grad = torch.autograd.grad

    def spy_grad(*args, **kwargs):
        calls["n"] += 1
        return orig_grad(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", spy_grad)

    U, F = compute_energy_and_forces(
        model, x, types_, masses, mask, V, mol_props, edge_feats, force_mode="direct",
    )
    assert calls["n"] == 0, "force_mode='direct' must never call torch.autograd.grad"

    loss, _, _ = spice_loss(U, torch.zeros_like(U), F, torch.zeros_like(F), mask, 1.0, 1.0)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"
    assert all(torch.isfinite(g).all() for g in grads)


def test_both_mode_conservative_component_matches_conservative_mode():
    model = build_model()
    model.eval()
    B, N = 2, 5
    x, types_, masses, mask, V, mol_props, edge_feats = make_batch(B, N, seed=3)

    U_cons, F_cons_only = compute_energy_and_forces(
        model, x, types_, masses, mask, V, mol_props, edge_feats, force_mode="conservative",
    )
    U_both, F_direct_both, F_cons_both = compute_energy_and_forces(
        model, x, types_, masses, mask, V, mol_props, edge_feats, force_mode="both",
    )

    torch.testing.assert_close(U_cons, U_both, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(F_cons_only, F_cons_both, atol=ATOL, rtol=RTOL)

    # The direct component of "both" must equal plain "direct" mode on the same input.
    _, F_direct_only = compute_energy_and_forces(
        model, x, types_, masses, mask, V, mol_props, edge_feats, force_mode="direct",
    )
    torch.testing.assert_close(F_direct_only, F_direct_both, atol=ATOL, rtol=RTOL)


def test_spice_loss_extra_terms_are_noops_when_disabled():
    B, N = 2, 5
    U = torch.randn(B, dtype=DTYPE)
    U_hat = torch.randn(B, dtype=DTYPE)
    F = torch.randn(B, N, 3, dtype=DTYPE)
    F_hat = torch.randn(B, N, 3, dtype=DTYPE)
    mask = torch.ones(B, N, dtype=torch.bool)

    base = spice_loss(U, U_hat, F, F_hat, mask, 1.0, 2.0)
    extended = spice_loss(
        U, U_hat, F, F_hat, mask, 1.0, 2.0,
        F_cons=torch.randn(B, N, 3, dtype=DTYPE), F_cons_hat=None,
        lambda2_conservative=None, lambda_force_match=0.0,
    )
    for a, b in zip(base, extended):
        torch.testing.assert_close(a, b, atol=0.0, rtol=0.0)


def test_checkpoint_compatible_across_force_modes():
    model_a = build_model(seed=1)
    model_b = build_model(seed=2)
    missing, unexpected = model_b.load_state_dict(model_a.state_dict(), strict=True)
    assert not missing and not unexpected
