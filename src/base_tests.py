import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest
import torch

from stage_1.model import (
    EnergyModel, SolventLayer,
    seed_equivariant_features,
    masked_weighted_com,
    masked_mean,
    SH1_TO_CARTESIAN,
    SH2_TO_CARTESIAN,
)
from utils.se3_utils import RadialNetwork, flat_to_traceless_symmetric
from utils.pw_bases import minimum_image
from utils.calc_props import _MOL_FEAT_DIM
from stage_1.spice_dataset import _EDGE_FEAT_DIM

torch.manual_seed(0)

ATOL = 1e-9
RTOL = 1e-9

MODEL_ATOL = 1e-9
MODEL_RTOL = 1e-9

DTYPE = torch.double
NUM_ATOM_TYPES = 5
FEATURE_DIM = 8
HIDDEN_DIM = 16
MAX_DEGREE = 2
NUM_LAYERS = 2
EDGE_CUTOFF = 100.0
N_MAX = 1


def random_rotation_matrix(dtype=DTYPE, generator=None) -> torch.Tensor:
    A = torch.randn(3, 3, dtype=torch.double, generator=generator)
    Q, R = torch.linalg.qr(A)
    d = torch.diagonal(R).sign()
    Q = Q * d
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q.to(dtype)


def make_species(B, N, generator, dtype=DTYPE):
    x = torch.randn(B, N, 3, generator=generator, dtype=dtype)
    types_ = torch.randint(0, NUM_ATOM_TYPES, (B, N), generator=generator)
    masses = torch.empty(B, N, dtype=dtype).uniform_(1.0, 16.0, generator=generator)
    n_real = torch.randint(1, N + 1, (B,), generator=generator)
    arange = torch.arange(N).unsqueeze(0)
    mask = arange < n_real.unsqueeze(1)
    return x, types_, masses, mask


def make_full_species(B, N, generator, dtype=DTYPE):
    x = torch.randn(B, N, 3, generator=generator, dtype=dtype)
    types_ = torch.randint(0, NUM_ATOM_TYPES, (B, N), generator=generator)
    masses = torch.empty(B, N, dtype=dtype).uniform_(1.0, 16.0, generator=generator)
    mask = torch.ones(B, N, dtype=torch.bool)
    return x, types_, masses, mask


def make_lattice(B, generator, dtype=DTYPE):
    return torch.eye(3, dtype=dtype).unsqueeze(0).repeat(B, 1, 1) * 5.0 \
        + 0.1 * torch.randn(B, 3, 3, generator=generator, dtype=dtype)


def make_mol_props(B, generator, dtype=DTYPE):
    return torch.randn(B, _MOL_FEAT_DIM, generator=generator, dtype=dtype)


def make_edge_feats(B, N, generator, dtype=DTYPE):
    return torch.rand(B, N, N, _EDGE_FEAT_DIM, generator=generator, dtype=dtype)


def rotate_x(x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    return x @ R.T


def rotate_V(V: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    return torch.einsum("ij,bjk->bik", R, V)


def to_cartesian_vector(out_sh: torch.Tensor) -> torch.Tensor:
    U = SH1_TO_CARTESIAN.to(dtype=out_sh.dtype, device=out_sh.device)
    return torch.einsum("ij,...j->...i", U, out_sh)


def to_cartesian_tensor(out_sh: torch.Tensor) -> torch.Tensor:
    U = SH2_TO_CARTESIAN.to(dtype=out_sh.dtype, device=out_sh.device)
    flat = torch.einsum("ij,...j->...i", U, out_sh)
    return flat_to_traceless_symmetric(flat)


def assert_scalar_invariant(out_raw, out_tf, atol=ATOL, rtol=RTOL):
    torch.testing.assert_close(out_raw, out_tf, atol=atol, rtol=rtol)


def assert_vector_rotates(out_raw_sh, out_tf_sh, R, atol=ATOL, rtol=RTOL):
    cart_raw = to_cartesian_vector(out_raw_sh)
    cart_tf = to_cartesian_vector(out_tf_sh)
    expected = torch.einsum("ij,...j->...i", R, cart_raw)
    torch.testing.assert_close(cart_tf, expected, atol=atol, rtol=rtol)


def assert_tensor_rotates(out_raw_sh, out_tf_sh, R, atol=ATOL, rtol=RTOL):
    Q_raw = to_cartesian_tensor(out_raw_sh)
    Q_tf = to_cartesian_tensor(out_tf_sh)
    expected = torch.einsum("ij,...jk,lk->...il", R, Q_raw, R)
    torch.testing.assert_close(Q_tf, expected, atol=atol, rtol=rtol)


def _mask_select(t: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    return t[mask] if mask is not None else t


def assert_f_dict_invariant(f_raw, f_tf, mask=None, atol=ATOL, rtol=RTOL):
    for l in f_raw:
        assert_scalar_invariant(
            _mask_select(f_raw[l], mask), _mask_select(f_tf[l], mask), atol=atol, rtol=rtol,
        )


def assert_f_dict_rotates(f_raw, f_tf, R, mask=None, atol=ATOL, rtol=RTOL):
    assert_scalar_invariant(_mask_select(f_raw[0], mask), _mask_select(f_tf[0], mask), atol=atol, rtol=rtol)
    assert_vector_rotates(
        _mask_select(f_raw[1], mask).reshape(-1, 3), _mask_select(f_tf[1], mask).reshape(-1, 3),
        R, atol=atol, rtol=rtol,
    )
    assert_tensor_rotates(
        _mask_select(f_raw[2], mask).reshape(-1, 5), _mask_select(f_tf[2], mask).reshape(-1, 5),
        R, atol=atol, rtol=rtol,
    )


def build_model(num_layers=NUM_LAYERS):
    torch.manual_seed(123)
    m = EnergyModel(
        radial_net=RadialNetwork,
        num_atom_types=NUM_ATOM_TYPES,
        max_degree=MAX_DEGREE,
        num_layers=num_layers,
        feature_dim=FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
        edge_cutoff=EDGE_CUTOFF,
        n_max=N_MAX,
        n_master_nodes=2,
        num_fusion_heads=2,
        edge_feat_dim=_EDGE_FEAT_DIM,
    )
    return m.to(DTYPE).eval()


def run_model(model, x, types_, masses, mask, V, mol_props, edge_feats):
    with torch.no_grad():
        return model(x, types_, masses, mask, V, mol_props, edge_feats)


@pytest.mark.parametrize("full", [False, True])
def test_energy_model_rotation_equivariance(full):
    g = torch.Generator().manual_seed(42)
    B, N = 2, 6
    builder = make_full_species if full else make_species
    x, types_, masses, mask = builder(B, N, g)
    V = make_lattice(B, g)
    mol_props = make_mol_props(B, g)
    edge_feats = make_edge_feats(B, N, g)
    model = build_model()

    R = random_rotation_matrix()
    x_rot = rotate_x(x, R)
    V_rot = rotate_V(V, R)

    f_raw = run_model(model, x, types_, masses, mask, V, mol_props, edge_feats)
    f_tf = run_model(model, x_rot, types_, masses, mask, V_rot, mol_props, edge_feats)

    assert_f_dict_rotates(f_raw, f_tf, R, mask=mask, atol=MODEL_ATOL, rtol=MODEL_RTOL)


def test_energy_model_periodic_invariance_whole_system():
    g = torch.Generator().manual_seed(7)
    B, N = 2, 6
    x, types_, masses, mask = make_species(B, N, g)
    V = make_lattice(B, g)
    mol_props = make_mol_props(B, g)
    edge_feats = make_edge_feats(B, N, g)
    model = build_model()

    n = torch.tensor([2.0, -1.0, 3.0], dtype=DTYPE)
    shift = torch.einsum("bij,j->bi", V, n)
    x_shift = x + shift.unsqueeze(1)

    f_raw = run_model(model, x, types_, masses, mask, V, mol_props, edge_feats)
    f_tf = run_model(model, x_shift, types_, masses, mask, V, mol_props, edge_feats)

    assert_f_dict_invariant(f_raw, f_tf, mask=mask, atol=MODEL_ATOL, rtol=MODEL_RTOL)


def test_energy_model_periodic_invariance_single_atom():
    g = torch.Generator().manual_seed(13)
    B, N = 2, 6
    x, types_, masses, mask = make_full_species(B, N, g)
    V = make_lattice(B, g)
    mol_props = make_mol_props(B, g)
    edge_feats = make_edge_feats(B, N, g)
    model = build_model()

    n = torch.tensor([1.0, -2.0, 1.0], dtype=DTYPE)
    shift = torch.einsum("bij,j->bi", V, n)
    x_shift = x.clone()
    x_shift[:, 0, :] = x_shift[:, 0, :] + shift

    f_raw = run_model(model, x, types_, masses, mask, V, mol_props, edge_feats)
    f_tf = run_model(model, x_shift, types_, masses, mask, V, mol_props, edge_feats)

    assert_f_dict_invariant(f_raw, f_tf, mask=mask, atol=MODEL_ATOL, rtol=MODEL_RTOL)


def test_energy_model_layers_exceed_max_degree_no_crash():
    g = torch.Generator().manual_seed(99)
    B, N = 2, 5
    x, types_, masses, mask = make_full_species(B, N, g)
    V = make_lattice(B, g)
    mol_props = make_mol_props(B, g)
    edge_feats = make_edge_feats(B, N, g)
    model = build_model(num_layers=MAX_DEGREE + 3)

    f = run_model(model, x, types_, masses, mask, V, mol_props, edge_feats)
    for l in f:
        assert torch.isfinite(f[l]).all()


def test_master_fusion_head_divisibility_guard():
    with pytest.raises(AssertionError):
        EnergyModel(
            radial_net=RadialNetwork,
            num_atom_types=NUM_ATOM_TYPES,
            max_degree=MAX_DEGREE,
            num_layers=NUM_LAYERS,
            feature_dim=FEATURE_DIM,
            hidden_dim=HIDDEN_DIM,
            edge_cutoff=EDGE_CUTOFF,
            n_max=N_MAX,
            n_master_nodes=2,
            num_fusion_heads=3,
            edge_feat_dim=_EDGE_FEAT_DIM,
        )


def test_energy_model_double_backward_finite():
    g = torch.Generator().manual_seed(55)
    B, N = 2, 5
    x, types_, masses, mask = make_full_species(B, N, g)
    mask = torch.zeros(B, N, dtype=torch.bool)
    mask[:, 0] = True
    V = make_lattice(B, g)
    mol_props = make_mol_props(B, g)
    edge_feats = make_edge_feats(B, N, g)
    model = build_model(num_layers=4)

    x = x.clone().requires_grad_(True)
    f = model(x, types_, masses, mask, V, mol_props, edge_feats)
    U = f[0].sum()

    (grad_x,) = torch.autograd.grad(U, x, create_graph=True)
    assert torch.isfinite(grad_x).all(), "non-finite first-order force gradient"

    loss = grad_x.pow(2).sum()
    loss.backward()

    non_finite = [
        name for name, p in model.named_parameters()
        if p.grad is not None and not torch.isfinite(p.grad).all()
    ]
    assert not non_finite, f"non-finite double-backward gradients in: {non_finite}"


def seed(model, x, types_, masses, mask, V, mol_props):
    B, N = types_.shape
    f0_raw = model.input_embedding(types_)
    f0_ffn_out = model.f0_ffn(f0_raw)
    mol_ctx = model.mol_prop_emb(mol_props).unsqueeze(1).expand(-1, N, -1)
    mol_ffn_out = model.mol_ffn(mol_ctx)
    combined_scalar = model.seed_proj(torch.cat([f0_ffn_out, mol_ffn_out], dim=-1))
    combined_scalar = model.seed_proj_norm(model.seed_proj_out(combined_scalar))
    seeded = seed_equivariant_features(combined_scalar, model.seed_radial_net, x, masses, mask, V)
    return {0: f0_raw.unsqueeze(-1), 1: seeded[1], 2: seeded[2]}


def run_layer(model, x, types_, masses, mask, V, mol_props, edge_feats):
    f_in = seed(model, x, types_, masses, mask, V, mol_props)
    x_wrapped = minimum_image(x, V)
    x_com, _ = masked_weighted_com(x_wrapped, masses, mask)
    agg = model.master_pool_agg(f_in[0], f_in[1], f_in[2]).squeeze(-1)
    pooled = model.master_pool_ffn(masked_mean(agg, mask))
    B = x.shape[0]
    g_a = model.master_expand(pooled).view(B, model.n_master_nodes, -1).unsqueeze(-1)
    with torch.no_grad():
        return model.layers[0](
            f_in=f_in, f0=f_in, g_a_in=g_a, x=x, x_com=x_com,
            edge_feats=edge_feats, node_mask=mask, V=V,
        )


def test_energy_layer_rotation_equivariance():
    g = torch.Generator().manual_seed(42)
    B, N = 2, 6
    x = torch.randn(B, N, 3, generator=g, dtype=DTYPE)
    types_ = torch.randint(0, NUM_ATOM_TYPES, (B, N), generator=g)
    masses = torch.empty(B, N, dtype=DTYPE).uniform_(1.0, 16.0, generator=g)
    mask = torch.ones(B, N, dtype=torch.bool)
    V = make_lattice(B, g)
    mol_props = make_mol_props(B, g)
    edge_feats = make_edge_feats(B, N, g)
    model = build_model()

    R = random_rotation_matrix()
    x_rot = rotate_x(x, R)
    V_rot = rotate_V(V, R)

    f_out_raw, g_a_raw = run_layer(model, x, types_, masses, mask, V, mol_props, edge_feats)
    f_out_rot, g_a_rot = run_layer(model, x_rot, types_, masses, mask, V_rot, mol_props, edge_feats)

    assert_f_dict_rotates(f_out_raw, f_out_rot, R)
    assert_scalar_invariant(g_a_raw, g_a_rot)


def test_energy_layer_periodic_invariance_whole_system():
    g = torch.Generator().manual_seed(7)
    B, N = 2, 6
    x = torch.randn(B, N, 3, generator=g, dtype=DTYPE)
    types_ = torch.randint(0, NUM_ATOM_TYPES, (B, N), generator=g)
    masses = torch.empty(B, N, dtype=DTYPE).uniform_(1.0, 16.0, generator=g)
    mask = torch.ones(B, N, dtype=torch.bool)
    V = make_lattice(B, g)
    mol_props = make_mol_props(B, g)
    edge_feats = make_edge_feats(B, N, g)
    model = build_model()

    n = torch.tensor([2.0, -1.0, 3.0], dtype=DTYPE)
    shift = torch.einsum("bij,j->bi", V, n)
    x_shift = x + shift.unsqueeze(1)

    f_out_raw, g_a_raw = run_layer(model, x, types_, masses, mask, V, mol_props, edge_feats)
    f_out_tf, g_a_tf = run_layer(model, x_shift, types_, masses, mask, V, mol_props, edge_feats)

    assert_f_dict_invariant(f_out_raw, f_out_tf)
    assert_scalar_invariant(g_a_raw, g_a_tf)


def test_energy_layer_periodic_invariance_single_atom():
    g = torch.Generator().manual_seed(13)
    B, N = 2, 6
    x = torch.randn(B, N, 3, generator=g, dtype=DTYPE)
    types_ = torch.randint(0, NUM_ATOM_TYPES, (B, N), generator=g)
    masses = torch.empty(B, N, dtype=DTYPE).uniform_(1.0, 16.0, generator=g)
    mask = torch.ones(B, N, dtype=torch.bool)
    V = make_lattice(B, g)
    mol_props = make_mol_props(B, g)
    edge_feats = make_edge_feats(B, N, g)
    model = build_model()

    n = torch.tensor([1.0, -2.0, 1.0], dtype=DTYPE)
    shift = torch.einsum("bij,j->bi", V, n)
    x_shift = x.clone()
    x_shift[:, 0, :] = x_shift[:, 0, :] + shift

    f_out_raw, g_a_raw = run_layer(model, x, types_, masses, mask, V, mol_props, edge_feats)
    f_out_tf, g_a_tf = run_layer(model, x_shift, types_, masses, mask, V, mol_props, edge_feats)

    assert_f_dict_invariant(f_out_raw, f_out_tf)
    assert_scalar_invariant(g_a_raw, g_a_tf)


SOLVENT_N1, SOLVENT_N2 = 4, 5

_SOLVENT_SEED_RADIAL_NET = RadialNetwork(num_basis=1, hidden_dim=HIDDEN_DIM).to(DTYPE)


def build_solvent_layer():
    torch.manual_seed(123)
    layer = SolventLayer(
        radial_net=RadialNetwork,
        max_degree=MAX_DEGREE,
        feature_dim=FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
        edge_cutoff=EDGE_CUTOFF,
        n_max=N_MAX,
    )
    return layer.to(DTYPE).eval()


def seed_pair(f0_i, x_i, masses_i, mask_i, f0_j, x_j, masses_j, mask_j, V):
    seeded_i = seed_equivariant_features(f0_i.squeeze(-1), _SOLVENT_SEED_RADIAL_NET, x_i, masses_i, mask_i, V)
    seeded_j = seed_equivariant_features(f0_j.squeeze(-1), _SOLVENT_SEED_RADIAL_NET, x_j, masses_j, mask_j, V)
    f_in_i = {0: f0_i, 1: seeded_i[1], 2: seeded_i[2]}
    f_in_j = {0: f0_j, 1: seeded_j[1], 2: seeded_j[2]}
    return f_in_i, f_in_j


def run_solvent_layer(layer, f0_i, x_i, masses_i, mask_i, f0_j, x_j, masses_j, mask_j, V):
    f_in_i, f_in_j = seed_pair(f0_i, x_i, masses_i, mask_i, f0_j, x_j, masses_j, mask_j, V)
    with torch.no_grad():
        return layer(f_in_i, x_i, mask_i, f_in_j, x_j, mask_j, V)


def assert_solvent_outputs_invariant(out_raw, out_tf):
    for a, b in zip(out_raw, out_tf):
        assert_scalar_invariant(a, b)


def assert_solvent_outputs_rotate(out_raw, out_tf, R):
    z_0, z_1, z_2, f_0_j, f_1_j, f_2_j = out_raw
    z_0r, z_1r, z_2r, f_0_jr, f_1_jr, f_2_jr = out_tf

    assert_scalar_invariant(z_0, z_0r)
    assert_scalar_invariant(f_0_j, f_0_jr)
    assert_vector_rotates(z_1.reshape(-1, 3), z_1r.reshape(-1, 3), R)
    assert_vector_rotates(f_1_j.reshape(-1, 3), f_1_jr.reshape(-1, 3), R)
    assert_tensor_rotates(z_2.reshape(-1, 5), z_2r.reshape(-1, 5), R)
    assert_tensor_rotates(f_2_j.reshape(-1, 5), f_2_jr.reshape(-1, 5), R)


def test_solvent_layer_rotation_equivariance():
    g = torch.Generator().manual_seed(42)
    B = 2
    x_i, types_i, masses_i, mask_i = make_full_species(B, SOLVENT_N1, g)
    x_j, types_j, masses_j, mask_j = make_full_species(B, SOLVENT_N2, g)
    V = make_lattice(B, g)
    layer = build_solvent_layer()

    f0_i = torch.randn(B, SOLVENT_N1, FEATURE_DIM, 1, generator=g, dtype=DTYPE)
    f0_j = torch.randn(B, SOLVENT_N2, FEATURE_DIM, 1, generator=g, dtype=DTYPE)

    R = random_rotation_matrix()
    x_i_rot = rotate_x(x_i, R)
    x_j_rot = rotate_x(x_j, R)
    V_rot = rotate_V(V, R)

    out_raw = run_solvent_layer(layer, f0_i, x_i, masses_i, mask_i, f0_j, x_j, masses_j, mask_j, V)
    out_tf = run_solvent_layer(layer, f0_i, x_i_rot, masses_i, mask_i, f0_j, x_j_rot, masses_j, mask_j, V_rot)

    assert_solvent_outputs_rotate(out_raw, out_tf, R)


def test_solvent_layer_periodic_invariance_whole_system():
    g = torch.Generator().manual_seed(7)
    B = 2
    x_i, types_i, masses_i, mask_i = make_full_species(B, SOLVENT_N1, g)
    x_j, types_j, masses_j, mask_j = make_full_species(B, SOLVENT_N2, g)
    V = make_lattice(B, g)
    layer = build_solvent_layer()

    f0_i = torch.randn(B, SOLVENT_N1, FEATURE_DIM, 1, generator=g, dtype=DTYPE)
    f0_j = torch.randn(B, SOLVENT_N2, FEATURE_DIM, 1, generator=g, dtype=DTYPE)

    n = torch.tensor([2.0, -1.0, 3.0], dtype=DTYPE)
    shift = torch.einsum("bij,j->bi", V, n)

    out_raw = run_solvent_layer(layer, f0_i, x_i, masses_i, mask_i, f0_j, x_j, masses_j, mask_j, V)
    out_tf = run_solvent_layer(
        layer, f0_i, x_i + shift.unsqueeze(1), masses_i, mask_i,
        f0_j, x_j + shift.unsqueeze(1), masses_j, mask_j, V,
    )

    assert_solvent_outputs_invariant(out_raw, out_tf)


def test_solvent_layer_periodic_invariance_single_ligprot_atom():
    g = torch.Generator().manual_seed(13)
    B = 2
    x_i, types_i, masses_i, mask_i = make_full_species(B, SOLVENT_N1, g)
    x_j, types_j, masses_j, mask_j = make_full_species(B, SOLVENT_N2, g)
    V = make_lattice(B, g)
    layer = build_solvent_layer()

    f0_i = torch.randn(B, SOLVENT_N1, FEATURE_DIM, 1, generator=g, dtype=DTYPE)
    f0_j = torch.randn(B, SOLVENT_N2, FEATURE_DIM, 1, generator=g, dtype=DTYPE)

    n = torch.tensor([1.0, -2.0, 1.0], dtype=DTYPE)
    shift = torch.einsum("bij,j->bi", V, n)
    x_i_shift = x_i.clone()
    x_i_shift[:, 0, :] = x_i_shift[:, 0, :] + shift

    out_raw = run_solvent_layer(layer, f0_i, x_i, masses_i, mask_i, f0_j, x_j, masses_j, mask_j, V)
    out_tf = run_solvent_layer(layer, f0_i, x_i_shift, masses_i, mask_i, f0_j, x_j, masses_j, mask_j, V)

    assert_solvent_outputs_invariant(out_raw, out_tf)


def test_solvent_layer_periodic_invariance_single_water_atom():
    g = torch.Generator().manual_seed(17)
    B = 2
    x_i, types_i, masses_i, mask_i = make_full_species(B, SOLVENT_N1, g)
    x_j, types_j, masses_j, mask_j = make_full_species(B, SOLVENT_N2, g)
    V = make_lattice(B, g)
    layer = build_solvent_layer()

    f0_i = torch.randn(B, SOLVENT_N1, FEATURE_DIM, 1, generator=g, dtype=DTYPE)
    f0_j = torch.randn(B, SOLVENT_N2, FEATURE_DIM, 1, generator=g, dtype=DTYPE)

    n = torch.tensor([1.0, -2.0, 1.0], dtype=DTYPE)
    shift = torch.einsum("bij,j->bi", V, n)
    x_j_shift = x_j.clone()
    x_j_shift[:, 0, :] = x_j_shift[:, 0, :] + shift

    out_raw = run_solvent_layer(layer, f0_i, x_i, masses_i, mask_i, f0_j, x_j, masses_j, mask_j, V)
    out_tf = run_solvent_layer(layer, f0_i, x_i, masses_i, mask_i, f0_j, x_j_shift, masses_j, mask_j, V)

    assert_solvent_outputs_invariant(out_raw, out_tf)


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(pytest.main([__file__, "-v"]))