import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest
import torch

from utils.cue_backend import cuequivariance_available

if not cuequivariance_available():
    pytest.skip(
        "cuequivariance-torch not installed or no CUDA device -- run on the remote GPU "
        "box after `pip install -e '.[cuequivariance]'` to actually exercise this file.",
        allow_module_level=True,
    )

from base_tests import (
    DTYPE, NUM_ATOM_TYPES,
    random_rotation_matrix, rotate_x, rotate_V,
    make_species, make_full_species, make_lattice, make_mol_props, make_edge_feats,
    assert_f_dict_rotates,
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

ATOL = 1e-6
RTOL = 1e-6


def build_model(use_cuequivariance: bool):
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
        use_cuequivariance=use_cuequivariance,
    )
    return m.to(device="cuda", dtype=DTYPE).eval()


def run_model(model, x, types_, masses, mask, V, mol_props, edge_feats):
    with torch.no_grad():
        return model(x, types_, masses, mask, V, mol_props, edge_feats)


def test_cue_path_actually_engages():
    model = build_model(use_cuequivariance=True)
    assert model.layers[0].sr_attn.direct_sum_w._cue_ready
    assert model.layers[0].broadcast._cue_ready


@pytest.mark.parametrize("full", [False, True])
def test_cue_path_rotation_equivariant(full):
    g = torch.Generator().manual_seed(42)
    B, N = 2, 6
    builder = make_full_species if full else make_species
    x, types_, masses, mask = builder(B, N, g)
    V = make_lattice(B, g)
    mol_props = make_mol_props(B, g)
    edge_feats = make_edge_feats(B, N, g)
    device = torch.device("cuda")
    x, types_, masses, mask, V, mol_props, edge_feats = (
        t.to(device) for t in (x, types_, masses, mask, V, mol_props, edge_feats)
    )
    model = build_model(use_cuequivariance=True)

    R = random_rotation_matrix().to(device)
    x_rot = rotate_x(x, R)
    V_rot = rotate_V(V, R)

    f_raw = run_model(model, x, types_, masses, mask, V, mol_props, edge_feats)
    f_tf = run_model(model, x_rot, types_, masses, mask, V_rot, mol_props, edge_feats)

    assert_f_dict_rotates(f_raw, f_tf, R, mask=mask, atol=ATOL, rtol=RTOL)


def test_cue_path_output_shape_matches_reference():
    g = torch.Generator().manual_seed(7)
    B, N = 2, 6
    x, types_, masses, mask = make_species(B, N, g)
    V = make_lattice(B, g)
    mol_props = make_mol_props(B, g)
    edge_feats = make_edge_feats(B, N, g)
    device = torch.device("cuda")
    x, types_, masses, mask, V, mol_props, edge_feats = (
        t.to(device) for t in (x, types_, masses, mask, V, mol_props, edge_feats)
    )

    ref = run_model(build_model(False), x, types_, masses, mask, V, mol_props, edge_feats)
    cue = run_model(build_model(True), x, types_, masses, mask, V, mol_props, edge_feats)

    assert set(ref.keys()) == set(cue.keys())
    for l in ref:
        assert ref[l].shape == cue[l].shape
        assert torch.isfinite(cue[l]).all()
