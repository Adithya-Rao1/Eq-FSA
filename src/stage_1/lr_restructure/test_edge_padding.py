import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

torch.manual_seed(0)
torch.set_default_dtype(torch.double)

from utils.se3_utils import RadialNetwork
from stage_1.lr_restructure.model import _build_padded_edges, _next_pow2
from stage_1.lr_restructure.sr_attention import SRAttention
from stage_1.lr_restructure.layer import LRLayer

ATOL = 1e-9
RTOL = 1e-9


def test_next_pow2():
    assert _next_pow2(0) == 1
    assert _next_pow2(1) == 1
    assert _next_pow2(2) == 2
    assert _next_pow2(3) == 4
    assert _next_pow2(5) == 8
    assert _next_pow2(17) == 32


def test_build_padded_edges_real_prefix_matches_nonzero():
    torch.manual_seed(3)
    B, N = 2, 6
    pair_mask = torch.rand(B, N, N) > 0.6
    x_diff = torch.randn(B, N, N, 3)
    edge_feats = torch.randn(B, N, N, 4)

    edge_b0, edge_i0, edge_j0 = pair_mask.nonzero(as_tuple=True)
    E = edge_b0.shape[0]

    edge_b, edge_i, edge_j, x_rel_e, edge_feat_e, edge_valid = _build_padded_edges(
        pair_mask, x_diff, edge_feats,
    )

    assert edge_valid.sum().item() == E
    assert edge_b.shape[0] == _next_pow2(E)
    torch.testing.assert_close(edge_b[:E], edge_b0)
    torch.testing.assert_close(edge_i[:E], edge_i0)
    torch.testing.assert_close(edge_j[:E], edge_j0)
    torch.testing.assert_close(x_rel_e[:E], x_diff[edge_b0, edge_i0, edge_j0])
    torch.testing.assert_close(edge_feat_e[:E], edge_feats[edge_b0, edge_i0, edge_j0])
    if edge_b.shape[0] > E:
        assert torch.all(edge_b[E:] == edge_b0[-1])
        assert torch.all(edge_i[E:] == edge_i0[-1])
        assert torch.all(edge_j[E:] == edge_j0[-1])
        torch.testing.assert_close(x_rel_e[E:], x_diff[edge_b0[-1], edge_i0[-1], edge_j0[-1]].expand(edge_b.shape[0] - E, -1))


def test_build_padded_edges_zero_edges_no_nan():
    B, N = 1, 3
    pair_mask = torch.zeros(B, N, N, dtype=torch.bool)
    x_diff = torch.randn(B, N, N, 3)
    edge_feats = torch.randn(B, N, N, 2)

    edge_b, edge_i, edge_j, x_rel_e, edge_feat_e, edge_valid = _build_padded_edges(
        pair_mask, x_diff, edge_feats,
    )
    assert edge_valid.sum().item() == 0
    assert edge_b.shape[0] == 1
    assert torch.isfinite(x_rel_e).all()
    assert x_rel_e.norm(dim=-1).min() > 0  


def test_sr_attention_padding_is_noop_for_real_edges():
    max_degree, k_max, feature_dim, hidden_dim, edge_feat_dim = 2, 2, 5, 12, 2
    torch.manual_seed(5)
    attn = SRAttention(RadialNetwork, max_degree, feature_dim, hidden_dim, k_max, edge_feat_dim)

    g = torch.Generator().manual_seed(17)
    E, B, N, PAD = 9, 2, 5, 7
    f_in_i_e = {k: torch.randn(E, feature_dim, 2 * k + 1, generator=g, dtype=torch.double, requires_grad=True) for k in range(k_max + 1)}
    f0_j_e = {k: torch.randn(E, feature_dim, 2 * k + 1, generator=g, dtype=torch.double, requires_grad=True) for k in range(k_max + 1)}
    x_rel_e = torch.randn(E, 3, generator=g, dtype=torch.double)
    edge_feat_e = torch.randn(E, edge_feat_dim, generator=g, dtype=torch.double)
    edge_b = torch.randint(0, B, (E,), generator=g)
    edge_i = torch.randint(0, N, (E,), generator=g)

    f_in_ref = {k: v.detach().clone().requires_grad_() for k, v in f_in_i_e.items()}
    f0_ref = {k: v.detach().clone().requires_grad_() for k, v in f0_j_e.items()}
    alpha_ref, Wv_ref = attn(f_in_ref, f0_ref, x_rel_e, edge_feat_e, edge_b, edge_i, batch_size=B, num_atoms=N)

    f_in_pad = {k: torch.cat([v.detach(), v.detach()[-1:].repeat(PAD, 1, 1)], dim=0).requires_grad_() for k, v in f_in_i_e.items()}
    f0_pad = {k: torch.cat([v.detach(), v.detach()[-1:].repeat(PAD, 1, 1)], dim=0).requires_grad_() for k, v in f0_j_e.items()}
    x_rel_pad = torch.cat([x_rel_e, x_rel_e[-1:].repeat(PAD, 1)], dim=0)
    edge_feat_pad = torch.cat([edge_feat_e, edge_feat_e[-1:].repeat(PAD, 1)], dim=0)
    edge_b_pad = torch.cat([edge_b, edge_b[-1:].repeat(PAD)], dim=0)
    edge_i_pad = torch.cat([edge_i, edge_i[-1:].repeat(PAD)], dim=0)
    edge_valid = torch.cat([torch.ones(E, dtype=torch.bool), torch.zeros(PAD, dtype=torch.bool)])

    alpha_pad, Wv_pad = attn(
        f_in_pad, f0_pad, x_rel_pad, edge_feat_pad, edge_b_pad, edge_i_pad,
        batch_size=B, num_atoms=N, edge_valid=edge_valid,
    )

    torch.testing.assert_close(alpha_ref, alpha_pad[:E], atol=ATOL, rtol=RTOL)
    assert torch.all(alpha_pad[E:] == 0)
    for l in range(max_degree + 1):
        torch.testing.assert_close(Wv_ref[l], Wv_pad[l][:E], atol=ATOL, rtol=RTOL)

    loss_ref = alpha_ref.sum() + sum(v.pow(2).sum() for v in Wv_ref.values())
    loss_pad = alpha_pad.sum() + sum(v.pow(2).sum() for v in Wv_pad.values())
    loss_ref.backward()
    loss_pad.backward()
    for k in range(k_max + 1):
        torch.testing.assert_close(f_in_ref[k].grad, f_in_pad[k].grad[:E], atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(f0_ref[k].grad, f0_pad[k].grad[:E], atol=ATOL, rtol=RTOL)


def _make_layer(max_degree=1, feature_dim=4, hidden_dim=8, n_master_nodes=2, seed=21):
    torch.manual_seed(seed)
    return LRLayer(
        RadialNetwork, max_degree, feature_dim, hidden_dim, edge_cutoff=5.0, n_max=1,
        pw_e_cutoff=None, k_max=max_degree, edge_feat_dim=2, n_master_nodes=n_master_nodes,
    )


def test_layer_output_invariant_to_padding_amount():
    max_degree, feature_dim, hidden_dim, n_master_nodes = 1, 4, 8, 2
    B, N, E = 1, 4, 5

    layer_a = _make_layer(max_degree, feature_dim, hidden_dim, n_master_nodes, seed=99)
    layer_b = _make_layer(max_degree, feature_dim, hidden_dim, n_master_nodes, seed=99)
    layer_b.load_state_dict(layer_a.state_dict())

    g = torch.Generator().manual_seed(31)
    f_in = {l: torch.randn(B, N, feature_dim, 2 * l + 1, generator=g, dtype=torch.double) for l in range(max_degree + 1)}
    f0 = {l: torch.randn(B, N, feature_dim, 2 * l + 1, generator=g, dtype=torch.double) for l in range(max_degree + 1)}
    x_wrapped = torch.randn(B, N, 3, generator=g, dtype=torch.double)
    node_mask = torch.ones(B, N, dtype=torch.bool)
    edge_b = torch.randint(0, B, (E,), generator=g)
    edge_i = torch.randint(0, N, (E,), generator=g)
    edge_j = torch.randint(0, N, (E,), generator=g)
    x_rel_e = torch.randn(E, 3, generator=g, dtype=torch.double)
    edge_feat_e = torch.randn(E, 2, generator=g, dtype=torch.double)
    g_a_in = torch.randn(B, n_master_nodes, feature_dim, 1, generator=g, dtype=torch.double)
    X_a_in = torch.randn(B, n_master_nodes, 3, generator=g, dtype=torch.double)
    V = torch.eye(3, dtype=torch.double).unsqueeze(0).repeat(B, 1, 1) * 20.0
    kvecs = torch.randn(B, 7, 3, generator=g, dtype=torch.double)
    mode_mask = torch.ones(B, 7, dtype=torch.bool)

    edge_valid_none = torch.ones(E, dtype=torch.bool)
    f_a, g_a_a, X_a_a = layer_a(
        f_in=f_in, f0=f0, g_a_in=g_a_in, X_a_in=X_a_in, x_wrapped=x_wrapped, node_mask=node_mask,
        edge_b=edge_b, edge_i=edge_i, edge_j=edge_j, x_rel_e=x_rel_e, edge_feat_e=edge_feat_e,
        edge_valid=edge_valid_none, kvecs=kvecs, mode_mask=mode_mask, V=V, V_inv=None,
    )

    PAD = 11
    edge_b_pad = torch.cat([edge_b, edge_b[-1:].repeat(PAD)])
    edge_i_pad = torch.cat([edge_i, edge_i[-1:].repeat(PAD)])
    edge_j_pad = torch.cat([edge_j, edge_j[-1:].repeat(PAD)])
    x_rel_pad = torch.cat([x_rel_e, x_rel_e[-1:].repeat(PAD, 1)])
    edge_feat_pad = torch.cat([edge_feat_e, edge_feat_e[-1:].repeat(PAD, 1)])
    edge_valid_pad = torch.cat([torch.ones(E, dtype=torch.bool), torch.zeros(PAD, dtype=torch.bool)])

    f_b, g_a_b, X_a_b = layer_b(
        f_in=f_in, f0=f0, g_a_in=g_a_in, X_a_in=X_a_in, x_wrapped=x_wrapped, node_mask=node_mask,
        edge_b=edge_b_pad, edge_i=edge_i_pad, edge_j=edge_j_pad, x_rel_e=x_rel_pad,
        edge_feat_e=edge_feat_pad, edge_valid=edge_valid_pad, kvecs=kvecs, mode_mask=mode_mask,
        V=V, V_inv=None,
    )

    for l in range(max_degree + 1):
        torch.testing.assert_close(f_a[l], f_b[l], atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(g_a_a, g_a_b, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(X_a_a, X_a_b, atol=ATOL, rtol=RTOL)
