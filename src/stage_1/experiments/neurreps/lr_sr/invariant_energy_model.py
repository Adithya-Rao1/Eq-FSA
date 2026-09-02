from __future__ import annotations

import math

import torch
import torch.nn as nn

from stage_1.attention import MasterNodeAttention
from stage_1.experiments.neurreps.lr_sr.rsa_attention import RSAAttention
from stage_1.model import masked_mean, masked_weighted_com
from utils.se3_utils import EquivariantRMSNorm, init_sequential_, scalar_ffn

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
    def __init__(self, feature_dim: int, hidden_dim: int, num_outputs: int, extra_input_dim: int = 0,
                 linear_only: bool = False):
        super().__init__()
        if linear_only:
            self.net = nn.Linear(feature_dim + extra_input_dim, num_outputs)
        else:
            self.net = nn.Sequential(
                nn.Linear(feature_dim + extra_input_dim, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, num_outputs),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EqFSAEncoder(nn.Module):
    def __init__(
        self, num_master_nodes: int, feature_dim: int, hidden_dim: int, n_max: int, num_outputs: int,
        use_scale_feat: bool = True, use_per_node_com: bool = True, rescale_init_scale: float = 500.0,
        alpha_floor: float = 1e-4, use_input_dependent_g_a: bool = True,
        use_node_conditioned_k: bool = True, use_linear_decoder: bool = False,
    ):
        super().__init__()
        self.num_master_nodes = num_master_nodes
        self.use_scale_feat = use_scale_feat
        self.use_per_node_com = use_per_node_com
        self.use_input_dependent_g_a = use_input_dependent_g_a
        self.use_node_conditioned_k = use_node_conditioned_k
        self.embed = _ChargeEmbedding(feature_dim, hidden_dim)
        if use_input_dependent_g_a:
            self.dist_to_com_embed = nn.Linear(1, feature_dim)
            self.master_pool_ffn = scalar_ffn(feature_dim, hidden_dim)
            self.master_expand = nn.Sequential(
                nn.RMSNorm(feature_dim), nn.LeakyReLU(),
                nn.Linear(feature_dim, num_master_nodes * feature_dim),
            )
            init_sequential_(self.master_expand, "leaky_relu")
            self.master_expand_norm = EquivariantRMSNorm(feature_dim, degree=0)
        else:
            self.g_a_init = nn.Parameter(torch.randn(num_master_nodes, feature_dim) * 0.1)
        self.master_attn = MasterNodeAttention(feature_dim, n_max, hidden_dim=hidden_dim, alpha_floor=alpha_floor)
        self.master_attn_norm = _LearnableRescale(feature_dim, init_scale=rescale_init_scale)
        self.decoder = _Decoder(
            feature_dim, hidden_dim, num_outputs, extra_input_dim=1 if use_scale_feat else 0,
            linear_only=use_linear_decoder,
        )

    def forward(self, charges, positions, mask, V, return_features: bool = False):
        B = charges.shape[0]
        f_in0 = self.embed(charges)
        masses = mask.float()
        x_com, _ = masked_weighted_com(positions, masses, mask)
        if self.use_input_dependent_g_a:
            dist_to_com = (positions - x_com.unsqueeze(1)).norm(dim=-1, keepdim=True)
            pool_in_feat = f_in0.squeeze(-1) + self.dist_to_com_embed(dist_to_com)
            pooled_in = self.master_pool_ffn(masked_mean(pool_in_feat, mask))
            g_a = self.master_expand(pooled_in).view(B, self.num_master_nodes, -1).unsqueeze(-1)
            g_a = self.master_expand_norm(g_a)
        else:
            g_a = self.g_a_init.unsqueeze(0).expand(B, -1, -1).unsqueeze(-1).contiguous()
        out = self.master_attn(
            g_a, f_in0, positions, x_com, mask, V,
            return_scale_feat=self.use_scale_feat, per_node_com=self.use_per_node_com,
            node_conditioned_k=self.use_node_conditioned_k,
        )
        lr_master, scale_feat = out if self.use_scale_feat else (out, None)
        master_out = self.master_attn_norm(lr_master.squeeze(-1))
        g_a_out = g_a.squeeze(-1) + master_out
        pooled = g_a_out.mean(dim=1)
        decoder_in = torch.cat([pooled, scale_feat.mean(dim=1, keepdim=True)], dim=-1) if self.use_scale_feat else pooled
        pred = self.decoder(decoder_in)
        return (pred, decoder_in) if return_features else pred


class RSAEncoder(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, n_max: int, sigma: float, num_outputs: int):
        super().__init__()
        self.embed = _ChargeEmbedding(feature_dim, hidden_dim)
        self.rsa_attn = RSAAttention(feature_dim, n_max, sigma=sigma, hidden_dim=hidden_dim)
        self.decoder = _Decoder(feature_dim, hidden_dim, num_outputs)

    def forward(self, charges, positions, mask, V, return_features: bool = False):
        f_in0 = self.embed(charges)
        per_atom = self.rsa_attn(f_in0, positions, mask, V).squeeze(-1)
        pooled = (per_atom * mask.unsqueeze(-1)).sum(dim=1)
        pred = self.decoder(pooled)
        return (pred, pooled) if return_features else pred
