import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest
import torch

torch.manual_seed(0)
torch.set_default_dtype(torch.double)  

from base_tests import (
    DTYPE, NUM_ATOM_TYPES,
    random_rotation_matrix, rotate_x, rotate_V,
    make_species, make_full_species, make_lattice, make_mol_props, make_edge_feats,
    assert_f_dict_invariant, assert_f_dict_rotates,
)
from utils.se3_utils import RadialNetwork
from stage_1.spice_dataset import _EDGE_FEAT_DIM

from stage_1.lr_restructure.model import LREnergyModel

FEATURE_DIM = 8
HIDDEN_DIM = 16
MAX_DEGREE = 2
NUM_LAYERS = 2
N_MASTER_NODES = 2
EDGE_CUTOFF = 100.0
N_MAX = 1

ATOL = 1e-7
RTOL = 1e-7


def build_model(sigma=None):
    torch.manual_seed(123)
    m = LREnergyModel(
        radial_net=RadialNetwork,
        num_atom_types=NUM_ATOM_TYPES,
        max_degree=MAX_DEGREE,
        num_layers=NUM_LAYERS,
        feature_dim=FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
        edge_cutoff=EDGE_CUTOFF,
        n_max=N_MAX,
        n_master_nodes=N_MASTER_NODES,
        num_broadcast_heads=2,
        edge_feat_dim=_EDGE_FEAT_DIM,
        sigma=sigma,
    )
    return m.to(DTYPE).eval()


def run_model(model, x, types_, masses, mask, V, mol_props, edge_feats):
    with torch.no_grad():
        return model(x, types_, masses, mask, V, mol_props, edge_feats)


@pytest.mark.parametrize("sigma", [None, 8.0])
@pytest.mark.parametrize("full", [False, True])
def test_rotation_equivariance(sigma, full):
    g = torch.Generator().manual_seed(42)
    B, N = 2, 6
    builder = make_full_species if full else make_species
    x, types_, masses, mask = builder(B, N, g)
    V = make_lattice(B, g)
    mol_props = make_mol_props(B, g)
    edge_feats = make_edge_feats(B, N, g)
    model = build_model(sigma=sigma)

    R = random_rotation_matrix()
    x_rot = rotate_x(x, R)
    V_rot = rotate_V(V, R)

    f_raw = run_model(model, x, types_, masses, mask, V, mol_props, edge_feats)
    f_tf = run_model(model, x_rot, types_, masses, mask, V_rot, mol_props, edge_feats)

    assert_f_dict_rotates(f_raw, f_tf, R, mask=mask, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("sigma", [None, 8.0])
def test_periodic_invariance_whole_system(sigma):
    g = torch.Generator().manual_seed(7)
    B, N = 2, 6
    x, types_, masses, mask = make_species(B, N, g)
    V = make_lattice(B, g)
    mol_props = make_mol_props(B, g)
    edge_feats = make_edge_feats(B, N, g)
    model = build_model(sigma=sigma)

    n = torch.tensor([2.0, -1.0, 3.0], dtype=DTYPE)
    shift = torch.einsum("bij,j->bi", V, n)
    x_shift = x + shift.unsqueeze(1)

    f_raw = run_model(model, x, types_, masses, mask, V, mol_props, edge_feats)
    f_tf = run_model(model, x_shift, types_, masses, mask, V, mol_props, edge_feats)

    assert_f_dict_invariant(f_raw, f_tf, mask=mask, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("sigma", [None, 8.0])
def test_periodic_invariance_single_atom(sigma):
    g = torch.Generator().manual_seed(13)
    B, N = 2, 6
    x, types_, masses, mask = make_full_species(B, N, g)
    V = make_lattice(B, g)
    mol_props = make_mol_props(B, g)
    edge_feats = make_edge_feats(B, N, g)
    model = build_model(sigma=sigma)

    n = torch.tensor([1.0, -2.0, 1.0], dtype=DTYPE)
    shift = torch.einsum("bij,j->bi", V, n)
    x_shift = x.clone()
    x_shift[:, 0, :] = x_shift[:, 0, :] + shift

    f_raw = run_model(model, x, types_, masses, mask, V, mol_props, edge_feats)
    f_tf = run_model(model, x_shift, types_, masses, mask, V, mol_props, edge_feats)

    assert_f_dict_invariant(f_raw, f_tf, mask=mask, atol=ATOL, rtol=RTOL)


def run_model_layer_states(model, x, types_, masses, mask, V, mol_props, edge_feats):
    with torch.no_grad():
        return model(x, types_, masses, mask, V, mol_props, edge_feats, return_layer_states=True)


@pytest.mark.parametrize("sigma", [None, 8.0])
def test_layer_states_rotation_equivariance(sigma):
    g = torch.Generator().manual_seed(42)
    B, N = 2, 6
    x, types_, masses, mask = make_species(B, N, g)
    V = make_lattice(B, g)
    mol_props = make_mol_props(B, g)
    edge_feats = make_edge_feats(B, N, g)
    model = build_model(sigma=sigma)

    R = random_rotation_matrix()
    x_rot = rotate_x(x, R)
    V_rot = rotate_V(V, R)

    f_raw, states_raw = run_model_layer_states(model, x, types_, masses, mask, V, mol_props, edge_feats)
    f_tf, states_tf = run_model_layer_states(model, x_rot, types_, masses, mask, V_rot, mol_props, edge_feats)

    assert len(states_raw) == NUM_LAYERS
    assert_f_dict_rotates(f_raw, f_tf, R, mask=mask, atol=ATOL, rtol=RTOL)
    for layer_idx, (s_raw, s_tf) in enumerate(zip(states_raw, states_tf)):
        assert_f_dict_rotates(s_raw, s_tf, R, mask=mask, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("sigma", [None, 8.0])
def test_layer_states_periodic_invariance(sigma):
    g = torch.Generator().manual_seed(7)
    B, N = 2, 6
    x, types_, masses, mask = make_species(B, N, g)
    V = make_lattice(B, g)
    mol_props = make_mol_props(B, g)
    edge_feats = make_edge_feats(B, N, g)
    model = build_model(sigma=sigma)

    n = torch.tensor([2.0, -1.0, 3.0], dtype=DTYPE)
    shift = torch.einsum("bij,j->bi", V, n)
    x_shift = x + shift.unsqueeze(1)

    f_raw, states_raw = run_model_layer_states(model, x, types_, masses, mask, V, mol_props, edge_feats)
    f_tf, states_tf = run_model_layer_states(model, x_shift, types_, masses, mask, V, mol_props, edge_feats)

    assert_f_dict_invariant(f_raw, f_tf, mask=mask, atol=ATOL, rtol=RTOL)
    for s_raw, s_tf in zip(states_raw, states_tf):
        assert_f_dict_invariant(s_raw, s_tf, mask=mask, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("sigma", [None, 8.0])
def test_layer_states_forward_backward_finite(sigma):
    g = torch.Generator().manual_seed(99)
    B, N = 2, 6
    x, types_, masses, mask = make_species(B, N, g)
    V = make_lattice(B, g)
    mol_props = make_mol_props(B, g)
    edge_feats = make_edge_feats(B, N, g)
    model = build_model(sigma=sigma)
    model.train()

    f, layer_states = model(x, types_, masses, mask, V, mol_props, edge_feats, return_layer_states=True)
    assert len(layer_states) == NUM_LAYERS
    loss = sum(v.pow(2).sum() for v in f.values())
    for layer_f in layer_states:
        loss = loss + sum(v.pow(2).sum() for v in layer_f.values())
    assert torch.isfinite(loss)
    loss.backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"


@pytest.mark.parametrize("sigma", [None, 8.0])
def test_forward_backward_finite(sigma):
    g = torch.Generator().manual_seed(99)
    B, N = 2, 6
    x, types_, masses, mask = make_species(B, N, g)
    V = make_lattice(B, g)
    mol_props = make_mol_props(B, g)
    edge_feats = make_edge_feats(B, N, g)
    model = build_model(sigma=sigma)
    model.train()

    f = model(x, types_, masses, mask, V, mol_props, edge_feats)
    loss = sum(v.pow(2).sum() for v in f.values())
    assert torch.isfinite(loss)
    loss.backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"
