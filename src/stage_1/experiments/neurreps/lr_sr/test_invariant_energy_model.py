from __future__ import annotations

import torch

from stage_1.experiments.neurreps.lr_sr.invariant_energy_model import EqFSAEncoder

DTYPE = torch.double
torch.set_default_dtype(DTYPE)


def _make_inputs(B=2, N=8, box=10.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    n_pos = N // 2
    charges_1d = torch.cat([torch.ones(n_pos, dtype=DTYPE), -torch.ones(N - n_pos, dtype=DTYPE)])
    charges = charges_1d.unsqueeze(0).expand(B, -1).clone()
    for b in range(B):
        perm = torch.randperm(N, generator=g)
        charges[b] = charges_1d[perm]
    x = torch.rand(B, N, 3, dtype=DTYPE, generator=g) * box
    mask = torch.ones(B, N, dtype=torch.bool)
    V = (torch.eye(3, dtype=DTYPE) * box).unsqueeze(0).expand(B, 3, 3).contiguous()
    return charges, x, mask, V


def _make_model():
    torch.manual_seed(42)
    return EqFSAEncoder(
        num_master_nodes=3, feature_dim=8, hidden_dim=16, n_max=1, num_outputs=2, num_layers=2,
        edge_cutoff=5.0,
    ).to(DTYPE)


def test_translation_invariance():
    mod = _make_model()
    mod.eval()
    charges, x, mask, V = _make_inputs()
    shift = torch.randn(1, 1, 3, dtype=DTYPE) * 3.0
    with torch.no_grad():
        out0 = mod(charges, x, mask, V)
        out1 = mod(charges, x + shift, mask, V)
    torch.testing.assert_close(out0, out1, atol=1e-6, rtol=1e-6)


def test_periodic_image_invariance():
    mod = _make_model()
    mod.eval()
    charges, x, mask, V = _make_inputs()
    x2 = x.clone()
    x2[:, 0] = x2[:, 0] + V[:, 0]
    with torch.no_grad():
        out0 = mod(charges, x, mask, V)
        out1 = mod(charges, x2, mask, V)
    torch.testing.assert_close(out0, out1, atol=1e-5, rtol=1e-5)


def test_forward_backward_finite():
    torch.manual_seed(1)
    mod = EqFSAEncoder(
        num_master_nodes=3, feature_dim=8, hidden_dim=16, n_max=1, num_outputs=2, num_layers=2,
        edge_cutoff=5.0,
    ).to(torch.float32)
    charges, x, mask, V = _make_inputs()
    charges, x, V = charges.float(), x.float(), V.float()
    out = mod(charges, x, mask, V)
    loss = out.pow(2).sum()
    loss.backward()
    assert torch.isfinite(loss)
    for p in mod.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


def test_scale_feat_disabled_matches_shape():
    torch.manual_seed(2)
    mod = EqFSAEncoder(
        num_master_nodes=3, feature_dim=8, hidden_dim=16, n_max=1, num_outputs=2, num_layers=2,
        edge_cutoff=5.0, use_scale_feat=False,
    ).to(torch.float32)
    charges, x, mask, V = _make_inputs()
    charges, x, V = charges.float(), x.float(), V.float()
    out = mod(charges, x, mask, V)
    assert out.shape == (charges.shape[0], 2)
    assert torch.isfinite(out).all()
