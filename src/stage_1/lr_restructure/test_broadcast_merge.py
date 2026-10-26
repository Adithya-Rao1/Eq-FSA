import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

torch.manual_seed(0)
torch.set_default_dtype(torch.double)

from utils.se3_utils import RadialNetwork
from stage_1.lr_restructure.broadcast import MasterBroadcastAttention
from stage_1.lr_restructure._broadcast_reference import MasterBroadcastAttentionReference

ATOL = 1e-9
RTOL = 1e-9


def _copy_weights(old: MasterBroadcastAttentionReference, new: MasterBroadcastAttention, degrees, hi_degrees):
    with torch.no_grad():
        for l in degrees:
            new.q_self_w[l].copy_(old.q_self_w[str(l)])
        new.readout_K_w.copy_(old.readout_K_w)
        new.readout_V_w.copy_(old.readout_V_w)
        new.radial_KV.load_state_dict(old.radial_KV.state_dict())
        new.mlp0.load_state_dict(old.mlp0.state_dict())
        for i, l in enumerate(hi_degrees):
            new.mlp_hi_w[i].copy_(old.mlp_hi[str(l)].weight.weight)
            new.gate_hi.W_gate[i].copy_(old.gate_hi[str(l)].gate_lin.weight)
            new.gate_hi.b_gate[i].copy_(old.gate_hi[str(l)].gate_lin.bias)


def test_broadcast_merge_matches_reference():
    max_degree, feature_dim, hidden_dim, num_heads = 2, 8, 16, 2
    degrees = list(range(max_degree + 1))
    hi_degrees = [l for l in degrees if l > 0]

    torch.manual_seed(3)
    old = MasterBroadcastAttentionReference(RadialNetwork, max_degree, feature_dim, hidden_dim, num_heads)
    torch.manual_seed(17)  # deliberately different -- weights must be copied, not coincide
    new = MasterBroadcastAttention(RadialNetwork, max_degree, feature_dim, hidden_dim, num_heads)
    _copy_weights(old, new, degrees, hi_degrees)

    g = torch.Generator().manual_seed(99)
    B, N, N_a = 2, 7, 3
    f_tilde = {l: torch.randn(B, N, feature_dim, 2 * l + 1, generator=g, requires_grad=True) for l in degrees}
    f_tilde2 = {l: v.detach().clone().requires_grad_() for l, v in f_tilde.items()}
    g_a = torch.randn(B, N_a, feature_dim, 1, generator=g, requires_grad=True)
    g_a2 = g_a.detach().clone().requires_grad_()
    X_a = torch.randn(B, N_a, 3, generator=g) * 3
    x = torch.randn(B, N, 3, generator=g) * 3
    node_mask = torch.ones(B, N, dtype=torch.bool)
    V = torch.eye(3).unsqueeze(0).repeat(B, 1, 1) * 20.0

    out_old = old(f_tilde, g_a, X_a, x, node_mask, V)
    out_new = new(f_tilde2, g_a2, X_a, x, node_mask, V)

    for l in degrees:
        torch.testing.assert_close(out_old[l], out_new[l], atol=ATOL, rtol=RTOL)

    loss_old = sum(v.pow(2).sum() for v in out_old.values())
    loss_new = sum(v.pow(2).sum() for v in out_new.values())
    loss_old.backward()
    loss_new.backward()

    for l in degrees:
        torch.testing.assert_close(f_tilde[l].grad, f_tilde2[l].grad, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(g_a.grad, g_a2.grad, atol=ATOL, rtol=RTOL)

    submodule_pairs = [(old.mlp0, new.mlp0), (old.radial_KV, new.radial_KV)]
    for m_old, m_new in submodule_pairs:
        for p_old, p_new in zip(m_old.parameters(), m_new.parameters()):
            if p_old.grad is None and p_new.grad is None:
                continue
            torch.testing.assert_close(p_old.grad, p_new.grad, atol=ATOL, rtol=RTOL)

    for l in degrees:
        torch.testing.assert_close(old.q_self_w[str(l)].grad, new.q_self_w.grad[l], atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(old.readout_K_w.grad, new.readout_K_w.grad, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(old.readout_V_w.grad, new.readout_V_w.grad, atol=ATOL, rtol=RTOL)
    for i, l in enumerate(hi_degrees):
        torch.testing.assert_close(old.mlp_hi[str(l)].weight.weight.grad, new.mlp_hi_w.grad[i], atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(old.gate_hi[str(l)].gate_lin.weight.grad, new.gate_hi.W_gate.grad[i], atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(old.gate_hi[str(l)].gate_lin.bias.grad, new.gate_hi.b_gate.grad[i], atol=ATOL, rtol=RTOL)


def test_broadcast_merge_zero_n_master_nodes_edge_case():
    max_degree, feature_dim, hidden_dim, num_heads = 2, 8, 16, 2
    degrees = list(range(max_degree + 1))
    hi_degrees = [l for l in degrees if l > 0]

    torch.manual_seed(3)
    old = MasterBroadcastAttentionReference(RadialNetwork, max_degree, feature_dim, hidden_dim, num_heads)
    torch.manual_seed(17)
    new = MasterBroadcastAttention(RadialNetwork, max_degree, feature_dim, hidden_dim, num_heads)
    _copy_weights(old, new, degrees, hi_degrees)

    g = torch.Generator().manual_seed(5)
    B, N, N_a = 2, 5, 1
    f_tilde = {l: torch.randn(B, N, feature_dim, 2 * l + 1, generator=g) for l in degrees}
    g_a = torch.randn(B, N_a, feature_dim, 1, generator=g)
    X_a = torch.randn(B, N_a, 3, generator=g) * 3
    x = torch.randn(B, N, 3, generator=g) * 3
    node_mask = torch.ones(B, N, dtype=torch.bool)
    V = torch.eye(3).unsqueeze(0).repeat(B, 1, 1) * 20.0

    with torch.no_grad():
        out_old = old(f_tilde, g_a, X_a, x, node_mask, V)
        out_new = new(f_tilde, g_a, X_a, x, node_mask, V)
    for l in degrees:
        torch.testing.assert_close(out_old[l], out_new[l], atol=ATOL, rtol=RTOL)
