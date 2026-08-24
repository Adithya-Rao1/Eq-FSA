from __future__ import annotations

import math

import torch
import torch.nn as nn

from stage_1.attention import MasterNodeAttention
from stage_1.experiments.neurreps.lr_sr.rsa_attention import RSAAttention
from stage_1.model import masked_weighted_com


class _LearnableRescale(nn.Module):
    def __init__(self, num_channels: int, init_scale: float = 500.0):
        super().__init__()
        self.log_scale = nn.Parameter(torch.full((num_channels,), math.log(init_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.log_scale.exp()


class _ChargeEmbedding(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, charges: torch.Tensor) -> torch.Tensor:
        return self.net(charges.unsqueeze(-1)).unsqueeze(-1)  


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
        self.master_attn_norm = _LearnableRescale(feature_dim)
        self.decoder = _Decoder(feature_dim, hidden_dim, num_outputs)

    def forward(self, charges, positions, mask, V):
        B = charges.shape[0]
        f_in0 = self.embed(charges)
        g_a = self.g_a_init.unsqueeze(0).expand(B, -1, -1).unsqueeze(-1).contiguous()
        masses = mask.float()
        x_com, _ = masked_weighted_com(positions, masses, mask)
        master_out = self.master_attn_norm(self.master_attn(g_a, f_in0, positions, x_com, mask, V).squeeze(-1))
        g_a_out = g_a.squeeze(-1) + master_out
        return self.decoder(g_a_out.mean(dim=1))


class RSAEncoder(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, n_max: int, sigma: float, num_outputs: int):
        super().__init__()
        self.embed = _ChargeEmbedding(feature_dim, hidden_dim)
        self.rsa_attn = RSAAttention(feature_dim, n_max, sigma=sigma, hidden_dim=hidden_dim)
        self.decoder = _Decoder(feature_dim, hidden_dim, num_outputs)

    def forward(self, charges, positions, mask, V):
        f_in0 = self.embed(charges)
        per_atom = self.rsa_attn(f_in0, positions, mask, V).squeeze(-1)  
        pooled = (per_atom * mask.unsqueeze(-1)).sum(dim=1)  
        return self.decoder(pooled)
