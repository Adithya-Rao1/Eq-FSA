from __future__ import annotations

import torch

from stage_1.experiments.neurreps.egnn_baseline import EGNNModel, EGNNLayer, build_edges_from_cutoff
from stage_1.experiments.neurreps.eqfsa_classifier import EqFSAClassifier
from utils.se3_utils import RadialNetwork

DTYPE = torch.double
ATOL, RTOL = 1e-9, 1e-9
NUM_ATOM_TYPES = 6
CLASSES = 5


def random_rotation_matrix(dtype=DTYPE, generator=None) -> torch.Tensor:
    A = torch.randn(3, 3, dtype=torch.double, generator=generator)
    Q, R = torch.linalg.qr(A)
    d = torch.diagonal(R).sign()
    Q = Q * d
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q.to(dtype)


def make_batch(B, N, generator, dtype=DTYPE):
    x = torch.randn(B, N, 3, generator=generator, dtype=dtype) * 3.0
    types = torch.randint(0, NUM_ATOM_TYPES, (B, N), generator=generator)
    mask = torch.ones(B, N, dtype=torch.bool)
    target_idx = torch.randint(0, N, (B,), generator=generator)
    return x, types, mask, target_idx

def test_egnn_layer_coordinate_update_equivariance():
    g = torch.Generator().manual_seed(1)
    B, N, hidden = 2, 6, 8
    x, types, mask, _ = make_batch(B, N, g)
    emb = torch.nn.Embedding(NUM_ATOM_TYPES, hidden, dtype=DTYPE)
    h = emb(types)
    layer = EGNNLayer(hidden, cutoff=100.0).to(DTYPE)

    edge_b, edge_i, edge_j = build_edges_from_cutoff(x, mask, 100.0)
    h_out, x_out = layer(h, x, edge_b, edge_i, edge_j)

    R = random_rotation_matrix()
    t = torch.randn(3, dtype=DTYPE, generator=g)
    x_tf = x @ R.T + t
    h_out_tf, x_out_tf = layer(h, x_tf, edge_b, edge_i, edge_j)

    torch.testing.assert_close(h_out_tf, h_out, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(x_out_tf, x_out @ R.T + t, atol=ATOL, rtol=RTOL)


def test_egnn_model_logits_rotation_translation_invariant():
    g = torch.Generator().manual_seed(2)
    B, N = 2, 8
    x, types, mask, target_idx = make_batch(B, N, g)
    model = EGNNModel(
        num_node_types=NUM_ATOM_TYPES, hidden_dim=8, num_layers=3, cutoff=100.0, classes=CLASSES,
    ).to(DTYPE)
    model.eval()

    R = random_rotation_matrix()
    t = torch.randn(3, dtype=DTYPE, generator=g)
    x_tf = x @ R.T + t

    with torch.no_grad():
        logits = model(x, types, mask, target_idx)
        logits_tf = model(x_tf, types, mask, target_idx)

    torch.testing.assert_close(logits_tf, logits, atol=ATOL, rtol=RTOL)


def test_egnn_model_forward_backward_finite():
    g = torch.Generator().manual_seed(3)
    B, N = 2, 8
    x, types, mask, target_idx = make_batch(B, N, g, dtype=torch.float32)
    model = EGNNModel(num_node_types=NUM_ATOM_TYPES, hidden_dim=16, num_layers=4, cutoff=100.0, classes=CLASSES)
    logits = model(x, types, mask, target_idx)
    loss = logits.pow(2).sum()
    loss.backward()
    assert torch.isfinite(loss)
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


def _make_eqfsa_batch(B, N, generator, dtype=DTYPE):
    from utils.calc_props import _MOL_FEAT_DIM
    x, types, mask, target_idx = make_batch(B, N, generator, dtype=dtype)
    masses = torch.ones(B, N, dtype=dtype)
    V = (torch.eye(3, dtype=dtype) * 1000.0).unsqueeze(0).expand(B, 3, 3).contiguous()
    mol_props = torch.zeros(B, _MOL_FEAT_DIM, dtype=dtype)
    edge_feats = torch.zeros(B, N, N, 1, dtype=dtype)
    return x, types, masses, mask, V, mol_props, edge_feats, target_idx


def _build_eqfsa(ablate_max_degree, dtype=DTYPE):
    model = EqFSAClassifier(
        radial_net=RadialNetwork, num_atom_types=NUM_ATOM_TYPES, max_degree=2, num_layers=2,
        feature_dim=8, hidden_dim=16, edge_cutoff=100.0, n_max=1, classes=CLASSES,
        ablate_max_degree=ablate_max_degree, edge_feat_dim=1,
    )
    return model.to(dtype) if dtype == torch.double else model


def test_eqfsa_classifier_rotation_translation_invariant():
    for ablate in (None, 0):
        g = torch.Generator().manual_seed(4)
        B, N = 2, 6
        x, types, masses, mask, V, mol_props, edge_feats, target_idx = _make_eqfsa_batch(B, N, g)
        model = _build_eqfsa(ablate)
        model.eval()

        R = random_rotation_matrix()
        t = torch.randn(3, dtype=DTYPE, generator=g)
        x_tf = x @ R.T + t

        with torch.no_grad():
            logits = model(x, types, masses, mask, V, mol_props, edge_feats, target_idx)
            logits_tf = model(x_tf, types, masses, mask, V, mol_props, edge_feats, target_idx)

        torch.testing.assert_close(
            logits_tf, logits, atol=ATOL, rtol=RTOL,
        ), f"ablate={ablate}"


def test_eqfsa_classifier_forward_backward_finite():
    for ablate in (None, 0):
        g = torch.Generator().manual_seed(5)
        B, N = 2, 6
        x, types, masses, mask, V, mol_props, edge_feats, target_idx = _make_eqfsa_batch(
            B, N, g, dtype=torch.float32,
        )
        model = _build_eqfsa(ablate, dtype=torch.float32)
        logits = model(x, types, masses, mask, V, mol_props, edge_feats, target_idx)
        loss = logits.pow(2).sum()
        loss.backward()
        assert torch.isfinite(loss)
        for p in model.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all()
