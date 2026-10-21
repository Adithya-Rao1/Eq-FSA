import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

torch.manual_seed(0)
torch.set_default_dtype(torch.double)

from utils.se3_utils import RadialNetwork, apply_direct_sum_W
from stage_1.attention import SRAttention as OrigSRAttention
from stage_1.lr_restructure.sr_attention import SRAttention as ForkedSRAttention, LRDirectSumW

ATOL = 1e-9
RTOL = 1e-9


def _make_radial_net(max_degree, k_max, hidden_dim=16, seed=1):
    torch.manual_seed(seed)
    n_total = sum(
        (l + k + 1) - abs(l - k) for l in range(max_degree + 1) for k in range(k_max + 1)
    )
    return RadialNetwork(num_basis=32, hidden_dim=hidden_dim, num_heads=n_total)


def test_direct_sum_w_matches_original():
    max_degree, k_max = 2, 2
    radial_net = _make_radial_net(max_degree, k_max)
    forked = LRDirectSumW(radial_net, max_degree, k_max)

    g = torch.Generator().manual_seed(42)
    N = 37
    x = torch.randn(N, 3, generator=g, dtype=torch.double)
    C = 5
    f = {k: torch.randn(N, C, 2 * k + 1, generator=g, dtype=torch.double, requires_grad=True) for k in range(k_max + 1)}
    f_clone = {k: v.detach().clone().requires_grad_() for k, v in f.items()}

    out_orig = apply_direct_sum_W(f=f, x=x, radial_nets=radial_net, max_degree=max_degree, k_max=k_max)
    out_new = forked(f_clone, x)

    for l in range(max_degree + 1):
        torch.testing.assert_close(out_orig[l], out_new[l], atol=ATOL, rtol=RTOL)

    loss_orig = sum(v.pow(2).sum() for v in out_orig.values())
    loss_new = sum(v.pow(2).sum() for v in out_new.values())
    loss_orig.backward()
    loss_new.backward()
    for k in range(k_max + 1):
        torch.testing.assert_close(f[k].grad, f_clone[k].grad, atol=ATOL, rtol=RTOL)
    for p_orig, p_new in zip(radial_net.parameters(), forked.radial_net.parameters()):
        torch.testing.assert_close(p_orig.grad, p_new.grad, atol=ATOL, rtol=RTOL)


def test_direct_sum_w_matches_original_zero_edges():
    max_degree, k_max = 2, 2
    radial_net = _make_radial_net(max_degree, k_max)
    forked = LRDirectSumW(radial_net, max_degree, k_max)

    N = 0
    x = torch.zeros(N, 3, dtype=torch.double)
    C = 5
    f = {k: torch.zeros(N, C, 2 * k + 1, dtype=torch.double) for k in range(k_max + 1)}

    out_orig = apply_direct_sum_W(f=f, x=x, radial_nets=radial_net, max_degree=max_degree, k_max=k_max)
    out_new = forked(f, x)
    for l in range(max_degree + 1):
        torch.testing.assert_close(out_orig[l], out_new[l], atol=ATOL, rtol=RTOL)


def test_sr_attention_matches_original():
    max_degree, k_max, feature_dim, hidden_dim, edge_feat_dim = 2, 2, 6, 16, 2

    torch.manual_seed(7)
    orig = OrigSRAttention(RadialNetwork, max_degree, feature_dim, hidden_dim, k_max, edge_feat_dim)
    torch.manual_seed(11) 
    forked = ForkedSRAttention(RadialNetwork, max_degree, feature_dim, hidden_dim, k_max, edge_feat_dim)

    forked.direct_sum_w.radial_net.load_state_dict(orig.radial_V_sr.state_dict())
    forked.pair_readout.load_state_dict(orig.pair_readout.state_dict())
    forked.edge_input_norm.load_state_dict(orig.edge_input_norm.state_dict())
    forked.psi_e.load_state_dict(orig.psi_e.state_dict())
    forked.psi_e_out.load_state_dict(orig.psi_e_out.state_dict())

    g = torch.Generator().manual_seed(99)
    E, B, N = 23, 2, 6
    f_in_i_e = {k: torch.randn(E, feature_dim, 2 * k + 1, generator=g, dtype=torch.double, requires_grad=True) for k in range(k_max + 1)}
    f0_j_e = {k: torch.randn(E, feature_dim, 2 * k + 1, generator=g, dtype=torch.double, requires_grad=True) for k in range(k_max + 1)}
    x_rel_e = torch.randn(E, 3, generator=g, dtype=torch.double)
    edge_feat_e = torch.randn(E, edge_feat_dim, generator=g, dtype=torch.double)
    edge_b = torch.randint(0, B, (E,), generator=g)
    edge_i = torch.randint(0, N, (E,), generator=g)

    f_in_i_e2 = {k: v.detach().clone().requires_grad_() for k, v in f_in_i_e.items()}
    f0_j_e2 = {k: v.detach().clone().requires_grad_() for k, v in f0_j_e.items()}

    alpha_orig, Wv_orig = orig(f_in_i_e, f0_j_e, x_rel_e, edge_feat_e, edge_b, edge_i, batch_size=B, num_atoms=N)
    alpha_new, Wv_new = forked(f_in_i_e2, f0_j_e2, x_rel_e, edge_feat_e, edge_b, edge_i, batch_size=B, num_atoms=N)

    torch.testing.assert_close(alpha_orig, alpha_new, atol=ATOL, rtol=RTOL)
    for l in range(max_degree + 1):
        torch.testing.assert_close(Wv_orig[l], Wv_new[l], atol=ATOL, rtol=RTOL)

    loss_orig = alpha_orig.sum() + sum(v.pow(2).sum() for v in Wv_orig.values())
    loss_new = alpha_new.sum() + sum(v.pow(2).sum() for v in Wv_new.values())
    loss_orig.backward()
    loss_new.backward()

    for k in range(k_max + 1):
        torch.testing.assert_close(f_in_i_e[k].grad, f_in_i_e2[k].grad, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(f0_j_e[k].grad, f0_j_e2[k].grad, atol=ATOL, rtol=RTOL)

    submodule_pairs = [
        (orig.radial_V_sr, forked.direct_sum_w.radial_net),
        (orig.pair_readout, forked.pair_readout),
        (orig.edge_input_norm, forked.edge_input_norm),
        (orig.psi_e, forked.psi_e),
        (orig.psi_e_out, forked.psi_e_out),
    ]
    for m_orig, m_new in submodule_pairs:
        for p_orig, p_new in zip(m_orig.parameters(), m_new.parameters()):
            if p_orig.grad is None and p_new.grad is None:
                continue
            torch.testing.assert_close(p_orig.grad, p_new.grad, atol=ATOL, rtol=RTOL)
