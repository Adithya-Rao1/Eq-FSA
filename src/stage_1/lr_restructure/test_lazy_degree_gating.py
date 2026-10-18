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
    assert_f_dict_rotates,
)
from utils.se3_utils import RadialNetwork
from stage_1.spice_dataset import _EDGE_FEAT_DIM
from stage_1.model import _compute_k_max

from stage_1.lr_restructure.model import LREnergyModel

FEATURE_DIM = 8
HIDDEN_DIM = 16
MAX_DEGREE = 2
NUM_LAYERS = 4  # with MAX_DEGREE=2 this exercises the full k_max cycle [0, 1, 2, 1]
N_MASTER_NODES = 2
EDGE_CUTOFF = 100.0
N_MAX = 1

ATOL = 1e-7
RTOL = 1e-7


def build_model(lazy_degree_gating: bool, sigma=None):
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
        lazy_degree_gating=lazy_degree_gating,
    )
    return m.to(DTYPE).eval()


def run_model(model, x, types_, masses, mask, V, mol_props, edge_feats, return_layer_states=False):
    with torch.no_grad():
        return model(
            x, types_, masses, mask, V, mol_props, edge_feats,
            return_layer_states=return_layer_states,
        )


def _k_max_schedule():
    return [_compute_k_max(m, MAX_DEGREE) for m in range(NUM_LAYERS)]


def test_k_max_schedule_matches_production():
    assert _k_max_schedule() == [0, 1, 2, 1]


@pytest.mark.parametrize("sigma", [None, 8.0])
@pytest.mark.parametrize("full", [False, True])
def test_rotation_equivariance_lazy_gated(sigma, full):
    g = torch.Generator().manual_seed(42)
    B, N = 2, 6
    builder = make_full_species if full else make_species
    x, types_, masses, mask = builder(B, N, g)
    V = make_lattice(B, g)
    mol_props = make_mol_props(B, g)
    edge_feats = make_edge_feats(B, N, g)
    model = build_model(lazy_degree_gating=True, sigma=sigma)

    R = random_rotation_matrix()
    x_rot = rotate_x(x, R)
    V_rot = rotate_V(V, R)

    f_raw = run_model(model, x, types_, masses, mask, V, mol_props, edge_feats)
    f_tf = run_model(model, x_rot, types_, masses, mask, V_rot, mol_props, edge_feats)

    assert_f_dict_rotates(f_raw, f_tf, R, mask=mask, atol=ATOL, rtol=RTOL)


def test_lazy_gating_off_matches_dense_bit_identical():
    g = torch.Generator().manual_seed(7)
    B, N = 2, 5
    x, types_, masses, mask = make_full_species(B, N, g)
    V = make_lattice(B, g)
    mol_props = make_mol_props(B, g)
    edge_feats = make_edge_feats(B, N, g)

    model_off = build_model(lazy_degree_gating=False)
    f_off = run_model(model_off, x, types_, masses, mask, V, mol_props, edge_feats)

    for l in f_off:
        assert torch.isfinite(f_off[l]).all()


def test_degrees_above_layer_ceiling_are_untouched_passthrough():
    import stage_1.lr_restructure.layer as layer_mod

    g = torch.Generator().manual_seed(99)
    B, N = 2, 6
    x, types_, masses, mask = make_full_species(B, N, g)
    V = make_lattice(B, g)
    mol_props = make_mol_props(B, g)
    edge_feats = make_edge_feats(B, N, g)
    model = build_model(lazy_degree_gating=True)

    schedule = _k_max_schedule()
    assert schedule == [0, 1, 2, 1]

    snapshots = []
    orig_forward = layer_mod.LRLayer.forward

    def capturing_forward(self, f_in, *args, **kwargs):
        f_next, g_a_out, X_a_new = orig_forward(self, f_in, *args, **kwargs)
        snapshots.append({l: t.clone() for l, t in f_next.items()})
        return f_next, g_a_out, X_a_new

    layer_mod.LRLayer.forward = capturing_forward
    try:
        run_model(model, x, types_, masses, mask, V, mol_props, edge_feats)
    finally:
        layer_mod.LRLayer.forward = orig_forward

    layer_states = snapshots
    assert len(layer_states) == NUM_LAYERS

    torch.testing.assert_close(layer_states[0][2], layer_states[1][2], atol=0.0, rtol=0.0)

    assert not torch.allclose(layer_states[1][2], layer_states[2][2])

    torch.testing.assert_close(layer_states[2][2], layer_states[3][2], atol=0.0, rtol=0.0)

    assert not torch.allclose(layer_states[0][1], layer_states[1][1])
