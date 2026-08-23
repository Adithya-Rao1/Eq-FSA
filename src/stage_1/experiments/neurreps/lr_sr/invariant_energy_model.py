"""Two encoders sharing an identical charge-embedding and decoder, differing only in the
long-range attention mechanism used to produce a single global representation from N atoms:
  - EqFSAEncoder: MasterNodeAttention (compressed, content-addressable master-node bottleneck,
    A << N states, query phase at a single shared reference point)
  - RSAEncoder: RSAAttention (genuinely per-atom outputs, query phase at each atom's own
    position), pooled via a plain sum over atoms -- mirrors the standard "total energy = sum of
    atomic contributions" MLIP convention, since RSA has no native global-representation pathway.
Both reused directly (not reimplemented) from their respective modules.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from stage_1.attention import MasterNodeAttention
from stage_1.experiments.neurreps.lr_sr.rsa_attention import RSAAttention
from stage_1.model import masked_weighted_com


class _ChargeEmbedding(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, charges: torch.Tensor) -> torch.Tensor:
        return self.net(charges.unsqueeze(-1)).unsqueeze(-1)  # (B,N,C,1) -- degree-0 convention


class _Decoder(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, num_outputs: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, num_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EqFSAEncoder(nn.Module):
    def __init__(
        self, num_master_nodes: int, feature_dim: int, hidden_dim: int, n_max: int, num_outputs: int,
    ):
        super().__init__()
        self.embed = _ChargeEmbedding(feature_dim, hidden_dim)
        self.g_a_init = nn.Parameter(torch.randn(num_master_nodes, feature_dim) * 0.1)
        self.master_attn = MasterNodeAttention(feature_dim, n_max, hidden_dim=hidden_dim)
        self.decoder = _Decoder(num_master_nodes * feature_dim, hidden_dim, num_outputs)

    def forward(self, charges, positions, mask, V):
        B = charges.shape[0]
        f_in0 = self.embed(charges)
        g_a = self.g_a_init.unsqueeze(0).expand(B, -1, -1).unsqueeze(-1).contiguous()
        masses = mask.float()
        x_com, _ = masked_weighted_com(positions, masses, mask)
        g_a_out = g_a.squeeze(-1) + self.master_attn(g_a, f_in0, positions, x_com, mask, V).squeeze(-1)
        return self.decoder(g_a_out.reshape(B, -1))


class RSAEncoder(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, n_max: int, sigma: float, num_outputs: int):
        super().__init__()
        self.embed = _ChargeEmbedding(feature_dim, hidden_dim)
        self.rsa_attn = RSAAttention(feature_dim, n_max, sigma=sigma, hidden_dim=hidden_dim)
        self.decoder = _Decoder(feature_dim, hidden_dim, num_outputs)

    def forward(self, charges, positions, mask, V):
        f_in0 = self.embed(charges)
        per_atom = self.rsa_attn(f_in0, positions, mask, V).squeeze(-1)  # (B,N,feature_dim)
        pooled = (per_atom * mask.unsqueeze(-1)).sum(dim=1)  # plain sum over atoms
        return self.decoder(pooled)
