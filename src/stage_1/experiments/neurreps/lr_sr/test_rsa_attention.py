from __future__ import annotations

import torch

from stage_1.experiments.neurreps.lr_sr.rsa_attention import RSAAttention

DTYPE = torch.double
torch.set_default_dtype(DTYPE)


def _make_inputs(B=2, N=8, C=6, box=10.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    f_in0 = torch.randn(B, N, C, 1, dtype=DTYPE, generator=g)
    x = torch.rand(B, N, 3, dtype=DTYPE, generator=g) * box
    mask = torch.ones(B, N, dtype=torch.bool)
    V = (torch.eye(3, dtype=DTYPE) * box).unsqueeze(0).expand(B, 3, 3).contiguous()
    return f_in0, x, mask, V


def test_translation_invariance():
    torch.manual_seed(42)
    mod = RSAAttention(feature_dim=6, n_max=1, sigma=5.0, hidden_dim=16).to(DTYPE)
    mod.eval()
    f_in0, x, mask, V = _make_inputs()
    shift = torch.randn(1, 1, 3, dtype=DTYPE) * 3.0
    with torch.no_grad():
        out0 = mod(f_in0, x, mask, V)
        out1 = mod(f_in0, x + shift, mask, V)
    torch.testing.assert_close(out0, out1, atol=1e-8, rtol=1e-8)


def test_periodic_image_invariance():
    torch.manual_seed(42)
    mod = RSAAttention(feature_dim=6, n_max=1, sigma=5.0, hidden_dim=16).to(DTYPE)
    mod.eval()
    f_in0, x, mask, V = _make_inputs()
    x2 = x.clone()
    x2[:, 0] = x2[:, 0] + V[:, 0] 
    with torch.no_grad():
        out0 = mod(f_in0, x, mask, V)
        out1 = mod(f_in0, x2, mask, V)
    torch.testing.assert_close(out0, out1, atol=1e-7, rtol=1e-7)


def test_forward_backward_finite():
    torch.manual_seed(1)
    mod = RSAAttention(feature_dim=6, n_max=1, sigma=5.0, hidden_dim=16).to(torch.float32)
    f_in0, x, mask, V = _make_inputs()
    f_in0, x, V = f_in0.float(), x.float(), V.float()
    out = mod(f_in0, x, mask, V)
    loss = out.pow(2).sum()
    loss.backward()
    assert torch.isfinite(loss)
    for p in mod.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()
