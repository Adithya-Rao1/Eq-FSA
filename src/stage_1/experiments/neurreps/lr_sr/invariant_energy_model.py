from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from stage_1.experiments.neurreps.lr_sr.rsa_attention import RSAAttention
from stage_1.model import EnergyModel, masked_mean
from utils.calc_props import mol_props_from_atom_counts
from utils.se3_utils import EquivariantNormReadout, RadialNetwork

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
        num_layers: int = 4, edge_cutoff: float = 5.0,
        use_scale_feat: bool = True, use_per_node_com: bool = True,
        alpha_floor: float = 1e-4,
        use_node_conditioned_k: bool = True, use_linear_decoder: bool = False,
        use_normalize_z_by_n_real: bool = True,
        max_g_a_degree: int = 0,
        g_a_additive_lift: bool = False,
        g_a_init_master_residual: bool = False,
        g_a_init_layer_residual: bool = False,
        simplified_qk: bool = False,
        fixed_alpha: Optional[float] = None,
        scale_feat_alphas: Optional[List[float]] = None,
        use_direction_aware_pool_agg: bool = False,
        pool_expand_learnable_gain: bool = False,
        ablate_lr: bool = False,
    ):
        super().__init__()
        self.use_scale_feat = use_scale_feat
        self.scale_feat_alphas = scale_feat_alphas
        self.charge_embed = _ChargeEmbedding(feature_dim, hidden_dim)
        self.norm_readout1 = EquivariantNormReadout(C=feature_dim, hidden=hidden_dim, out_dim=feature_dim)
        self.norm_readout2 = EquivariantNormReadout(C=feature_dim, hidden=hidden_dim, out_dim=feature_dim)
        self.energy_model = EnergyModel(
            radial_net=RadialNetwork, num_atom_types=1, max_degree=2, num_layers=num_layers,
            feature_dim=feature_dim, hidden_dim=hidden_dim, edge_cutoff=edge_cutoff, n_max=n_max,
            n_master_nodes=num_master_nodes, edge_feat_dim=1,
            normalize_z_by_n_real=use_normalize_z_by_n_real,
            per_node_com=use_per_node_com, node_conditioned_k=use_node_conditioned_k,
            alpha_floor=alpha_floor,
            decouple_ri_norm=False, imaginary_only_ri=False,
            deterministic_pool_agg=not use_direction_aware_pool_agg,
            direction_aware_pool_agg=use_direction_aware_pool_agg,
            attention_pool=True, skip_master_expand_norm=True, content_weighted_com=False,
            per_layer_fusion=True, reduce_norms=True,
            max_g_a_degree=max_g_a_degree,
            g_a_additive_lift=g_a_additive_lift,
            g_a_init_master_residual=g_a_init_master_residual,
            g_a_init_layer_residual=g_a_init_layer_residual,
            simplified_qk=simplified_qk,
            fixed_alpha=fixed_alpha,
            scale_feat_alphas=scale_feat_alphas,
            pool_expand_learnable_gain=pool_expand_learnable_gain,
            ablate_lr=ablate_lr,
        )
        num_scale_bands = (1 + len(scale_feat_alphas)) if scale_feat_alphas else 1
        self.decoder = _Decoder(
            feature_dim, hidden_dim, num_outputs, extra_input_dim=num_scale_bands if use_scale_feat else 0,
            linear_only=use_linear_decoder,
        )

    @property
    def master_attn(self):
        return self.energy_model.layers[-1].master_attn if len(self.energy_model.layers) > 0 else None

    def forward(self, charges, positions, mask, V, return_features: bool = False):
        B, N = charges.shape
        input_features = self.charge_embed(charges).squeeze(-1) 
        node_types = torch.zeros(B, N, dtype=torch.long, device=charges.device)
        masses = mask.float()
        n_h = ((charges > 0) & mask).sum(dim=1)
        n_o = ((charges < 0) & mask).sum(dim=1)  
        mol_props = mol_props_from_atom_counts(n_h, n_o).to(device=charges.device, dtype=positions.dtype)
        edge_feats = torch.zeros(B, N, N, 1, device=charges.device, dtype=positions.dtype)

        out = self.energy_model(
            positions, node_types, masses, mask, V, mol_props, edge_feats,
            input_features=input_features,
            collect_scale_feats=self.use_scale_feat,
        )
        f, scale_feats = out if self.use_scale_feat else (out, None)

        u_vec = self.norm_readout1(f[1]) 
        u_ten = self.norm_readout2(f[2]) 
        full_vec = f[0].squeeze(-1) + u_vec + u_ten
        pooled = masked_mean(full_vec, mask)  

        if self.use_scale_feat:
            num_scale_bands = (1 + len(self.scale_feat_alphas)) if self.scale_feat_alphas else 1
            if scale_feats:
                scale = torch.stack(scale_feats, dim=0).mean(dim=(0, 2))
                if scale.dim() == 1:
                    scale = scale.unsqueeze(-1)
            else:
                scale = torch.zeros(f[0].shape[0], num_scale_bands, device=f[0].device, dtype=f[0].dtype)
            decoder_in = torch.cat([pooled, scale], dim=-1)
        else:
            decoder_in = pooled
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
