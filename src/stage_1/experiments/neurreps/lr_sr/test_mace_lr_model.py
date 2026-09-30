from __future__ import annotations

import torch
import torch.nn as nn

from stage_1.experiments.neurreps.lr_sr.mace_lr_model import MaceLRModel
from stage_1.model import masked_mean, masked_weighted_com, seed_equivariant_features
from utils.pw_bases import minimum_image

DTYPE = torch.double
torch.set_default_dtype(DTYPE)


class _StubMace(nn.Module):
    def __init__(self):
        super().__init__()
        self.readouts = nn.ModuleList([nn.Linear(4, 4)])


def _make_inputs(B=2, N=6, box=10.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    n_pos = N // 2
    charges_1d = torch.cat([torch.ones(n_pos, dtype=DTYPE), -torch.ones(N - n_pos, dtype=DTYPE)])
    charges = charges_1d.unsqueeze(0).expand(B, -1).clone()
    for b in range(B):
        perm = torch.randperm(N, generator=g)
        charges[b] = charges_1d[perm]
    node_energy = torch.randn(B, N, dtype=DTYPE, generator=g)
    positions = torch.rand(B, N, 3, dtype=DTYPE, generator=g) * box
    mask = torch.ones(B, N, dtype=torch.bool)
    V = (torch.eye(3, dtype=DTYPE) * box).unsqueeze(0).expand(B, 3, 3).contiguous()
    return node_energy, charges, positions, mask, V


def test_num_lr_layers_one_matches_manual_single_shot():
    torch.manual_seed(1)
    model = MaceLRModel(
        _StubMace(), "eqfsa_lr", feature_dim=8, hidden_dim=16, n_max=1, num_master_nodes=3,
    ).to(DTYPE)
    model.eval()
    node_energy, charges, positions, mask, V = _make_inputs()

    with torch.no_grad():
        out_loop = model._lr_correction(node_energy, charges, positions, mask, V)

        from utils.pw_bases import make_wavevectors

        f_in0 = model.embed(node_energy, charges)
        B = charges.shape[0]
        masses = mask.float()
        x_com, _ = masked_weighted_com(positions, masses, mask)
        x_wrapped = minimum_image(positions, V)
        seeded = seed_equivariant_features(f_in0.squeeze(-1), model.seed_radial_net, positions, masses, mask, V)
        f_atoms = {0: f_in0, 1: seeded[1], 2: seeded[2]}

        g_a = model.g_a_init.unsqueeze(0).expand(B, -1, -1).unsqueeze(-1).clone()
        X_a = x_com.unsqueeze(1).expand(B, model.num_master_nodes, -1).clone()
        kvecs, _, mode_mask = make_wavevectors(V, model.n_max, None, indices=model.lr_miller_indices)

        pooled, X_a_new = model.lr_aggregator(g_a.squeeze(-1), f_atoms[0].squeeze(-1), x_wrapped, X_a, mask, V)
        g_a_new = model.lr_update(pooled, X_a_new, V, kvecs, mode_mask)
        g_a_out = model.master_ln(model.W_master_self(g_a) + g_a_new.unsqueeze(-1))
        fusion_out = model.lr_broadcast(f_atoms, g_a_out, X_a_new, x_wrapped, mask, V)
        fusion_scalar = model.fusion_pool(fusion_out[0], fusion_out[1], fusion_out[2]).squeeze(-1)
        fusion_pooled = masked_mean(fusion_scalar, mask)
        decoder_in = torch.cat([g_a_out.squeeze(-1).reshape(B, -1), fusion_pooled], dim=-1)
        out_manual = model.decoder(decoder_in).squeeze(-1)

    torch.testing.assert_close(out_loop, out_manual, atol=0.0, rtol=0.0)


def test_num_lr_layers_forward_backward_finite():
    torch.manual_seed(2)
    model = MaceLRModel(
        _StubMace(), "eqfsa_lr", feature_dim=8, hidden_dim=16, n_max=1, num_master_nodes=3, num_lr_layers=3,
    ).to(DTYPE)
    node_energy, charges, positions, mask, V = _make_inputs()

    out = model._lr_correction(node_energy, charges, positions, mask, V)
    assert torch.isfinite(out).all()
    loss = out.pow(2).sum()
    loss.backward()
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


def test_rsa_variant_unaffected_by_master_attn_rename():
    torch.manual_seed(3)
    model = MaceLRModel(_StubMace(), "rsa", feature_dim=8, hidden_dim=16, n_max=1, rsa_sigma=5.0).to(DTYPE)
    node_energy, charges, positions, mask, V = _make_inputs()
    out = model._lr_correction(node_energy, charges, positions, mask, V)
    assert torch.isfinite(out).all()
    assert not hasattr(model, "master_attn")
