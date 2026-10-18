from typing import Dict, Optional

import torch
import torch.nn as nn

from stage_1.model import (
    Readout, masked_weighted_com, safe_minimum_image_cutoff, seed_equivariant_features,
    _compute_k_max,
)
from utils.pw_bases import minimum_image, make_wavevectors, make_miller_indices, safe_inv
from utils.se3_utils import scalar_ffn, init_sequential_, init_linear_
from utils.spherical_harm import warm_spherical_harmonics_cache
from utils.calc_props import _MOL_FEAT_DIM

from stage_1.lr_restructure.layer import LRLayer


def _next_pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


@torch._dynamo.disable()
def _build_padded_edges(pair_mask: torch.Tensor, x_diff: torch.Tensor, edge_feats: torch.Tensor):
    edge_b, edge_i, edge_j = pair_mask.nonzero(as_tuple=True)
    B, N = x_diff.shape[0], x_diff.shape[1]
    flat_idx = edge_b * (N * N) + edge_i * N + edge_j
    x_rel_e = x_diff.reshape(B * N * N, 3).index_select(0, flat_idx)
    edge_feat_e = edge_feats.reshape(B * N * N, edge_feats.shape[-1]).index_select(0, flat_idx)

    E = int(edge_b.shape[0])
    device = edge_b.device

    if E == 0:
        edge_b = torch.zeros(1, dtype=edge_b.dtype, device=device)
        edge_i = torch.zeros(1, dtype=edge_i.dtype, device=device)
        edge_j = torch.zeros(1, dtype=edge_j.dtype, device=device)
        x_rel_e = torch.zeros(1, 3, dtype=x_diff.dtype, device=device)
        x_rel_e[:, 0] = 1.0
        edge_feat_e = torch.zeros(1, edge_feats.shape[-1], dtype=edge_feats.dtype, device=device)
        edge_valid = torch.zeros(1, dtype=torch.bool, device=device)
        return edge_b, edge_i, edge_j, x_rel_e, edge_feat_e, edge_valid

    bucket = min(pair_mask.numel(), _next_pow2(E))
    pad = bucket - E
    if pad > 0:
        edge_b = torch.cat([edge_b, edge_b[-1].repeat(pad)])
        edge_i = torch.cat([edge_i, edge_i[-1].repeat(pad)])
        edge_j = torch.cat([edge_j, edge_j[-1].repeat(pad)])
        x_rel_e = torch.cat([x_rel_e, x_rel_e[-1:].repeat(pad, 1)])
        edge_feat_e = torch.cat([edge_feat_e, edge_feat_e[-1:].repeat(pad, 1)])
    edge_valid = torch.arange(bucket, device=device) < E
    return edge_b, edge_i, edge_j, x_rel_e, edge_feat_e, edge_valid


class LREnergyModel(nn.Module):
    def __init__(
        self,
        radial_net,
        num_atom_types: int,
        max_degree: int = 2,
        num_layers: int = 4,
        feature_dim: int = 32,
        hidden_dim: int = 128,
        edge_cutoff: float = 5.0,
        n_max: int = 2,
        pw_e_cutoff: Optional[float] = None,
        n_master_nodes: int = 4,
        num_broadcast_heads: int = 4,
        edge_feat_dim: int = 8,
        rbf_num_basis: int = 12,
        rbf_min: float = 0.3,
        rbf_cutoff: float = 25.0,
        sigma: Optional[float] = None,
        alpha_floor: float = 1e-4,
        use_norms: bool = True,
        lazy_degree_gating: bool = False,
    ):
        super().__init__()
        self.max_degree = max_degree
        self.n_master_nodes = n_master_nodes
        self.lazy_degree_gating = lazy_degree_gating
        self.edge_cutoff = edge_cutoff
        self.n_max = n_max
        self.pw_e_cutoff = pw_e_cutoff
        
        warm_spherical_harmonics_cache(range(2 * max_degree + 1))

        self.register_buffer("miller_indices", make_miller_indices(n_max, dtype=torch.get_default_dtype()),)

        self.input_embedding = nn.Embedding(num_atom_types, feature_dim)
        self.ln1 = nn.Identity() if not use_norms else nn.RMSNorm(feature_dim)
        self.mol_prop_emb = nn.Sequential(
            scalar_ffn(_MOL_FEAT_DIM, hidden_dim, use_norms=use_norms),
            nn.RMSNorm(_MOL_FEAT_DIM) if use_norms else nn.Identity(),
            nn.LeakyReLU(),
            nn.Linear(_MOL_FEAT_DIM, feature_dim),
        )
        init_sequential_(self.mol_prop_emb, "leaky_relu")
        self.ln2 = nn.Identity() if not use_norms else nn.RMSNorm(feature_dim)
        self.f0_ffn = scalar_ffn(feature_dim, hidden_dim, use_norms=use_norms)
        self.mol_ffn = scalar_ffn(feature_dim, hidden_dim, use_norms=use_norms)

        self.seed_proj = scalar_ffn(2 * feature_dim, hidden_dim, use_norms=use_norms)
        self.seed_proj_out = nn.Linear(2 * feature_dim, feature_dim)
        init_linear_(self.seed_proj_out, "linear")
        self.seed_proj_norm = nn.Identity() if not use_norms else nn.RMSNorm(feature_dim)
        self.seed_radial_net = radial_net(num_basis=1, hidden_dim=hidden_dim, use_norms=use_norms)

        self.g_a_init = nn.Parameter(torch.randn(n_master_nodes, feature_dim) * 0.1)

        self.layers = nn.ModuleList([
            LRLayer(
                radial_net, max_degree, feature_dim, hidden_dim, edge_cutoff, n_max, pw_e_cutoff,
                k_max=(_compute_k_max(m, max_degree) if lazy_degree_gating else max_degree),
                edge_feat_dim=edge_feat_dim, n_master_nodes=n_master_nodes,
                num_broadcast_heads=num_broadcast_heads, rbf_num_basis=rbf_num_basis,
                rbf_min=rbf_min, rbf_cutoff=rbf_cutoff, sigma=sigma, alpha_floor=alpha_floor,
                use_norms=use_norms, lazy_degree_gating=lazy_degree_gating,
            )
            for m in range(num_layers)
        ])

        self.readout = Readout(feature_dim, hidden_dim, use_norms=use_norms)

    def forward(
        self,
        x: torch.Tensor,
        node_types: torch.Tensor,
        atomic_masses: torch.Tensor,
        node_mask: torch.Tensor,
        V: torch.Tensor,
        mol_props: torch.Tensor,
        edge_feats: torch.Tensor,
        input_features: Optional[torch.Tensor] = None,
        seed_masses: Optional[torch.Tensor] = None,
        return_layer_states: bool = False,
    ) -> Dict[int, torch.Tensor]:
        node_mask = node_mask.bool()
        atomic_masses = atomic_masses * node_mask.to(atomic_masses.dtype)
        seed_masses = atomic_masses if seed_masses is None else seed_masses * node_mask.to(seed_masses.dtype)
        B, N = node_mask.shape

        V_inv = safe_inv(V)

        f0_raw = input_features if input_features is not None else self.input_embedding(node_types)
        f0_ffn_out = self.f0_ffn(self.ln1(f0_raw))
        mol_ctx = self.mol_prop_emb(mol_props).unsqueeze(1).expand(-1, N, -1).contiguous()
        mol_ffn_out = self.mol_ffn(self.ln2(mol_ctx))
        combined_scalar = self.seed_proj(torch.cat([f0_ffn_out, mol_ffn_out], dim=-1))
        combined_scalar = self.seed_proj_norm(self.seed_proj_out(combined_scalar))

        x_wrapped = minimum_image(x, V, V_inv)
        seeded = seed_equivariant_features(
            combined_scalar, self.seed_radial_net, x, seed_masses, node_mask, V,
            x_wrapped=x_wrapped, V_inv=V_inv,
        )
        f0: Dict[int, torch.Tensor] = {0: f0_raw.unsqueeze(-1), 1: seeded[1], 2: seeded[2]}
        f: Dict[int, torch.Tensor] = dict(f0)

        x_com, _ = masked_weighted_com(x_wrapped, atomic_masses, node_mask, V=V, V_inv=V_inv)

        kvecs, _, mode_mask = make_wavevectors(V, self.n_max, self.pw_e_cutoff, V_inv=V_inv, indices=self.miller_indices)

        effective_cutoff = safe_minimum_image_cutoff(V, self.edge_cutoff).view(B, 1, 1)
        x_diff_raw = x_wrapped.unsqueeze(1) - x_wrapped.unsqueeze(2)
        x_diff = minimum_image(x_diff_raw.reshape(B, N * N, 3), V, V_inv).reshape(B, N, N, 3)
        dist = x_diff.norm(dim=-1)
        self_mask = ~torch.eye(N, dtype=torch.bool, device=x.device).unsqueeze(0)
        pair_mask = node_mask.unsqueeze(2) & node_mask.unsqueeze(1) & (dist <= effective_cutoff) & self_mask
        edge_b, edge_i, edge_j, x_rel_e, edge_feat_e, edge_valid = _build_padded_edges(
            pair_mask, x_diff, edge_feats,
        )

        g_a = self.g_a_init.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, -1).clone()
        X_a = x_com.unsqueeze(1).expand(B, self.n_master_nodes, -1).clone()

        layer_states = [] if return_layer_states else None
        for layer in self.layers:
            f, g_a, X_a = layer(
                f_in=f, f0=f0, g_a_in=g_a, X_a_in=X_a, x_wrapped=x_wrapped, node_mask=node_mask,
                edge_b=edge_b, edge_i=edge_i, edge_j=edge_j, x_rel_e=x_rel_e,
                edge_feat_e=edge_feat_e, edge_valid=edge_valid,
                kvecs=kvecs, mode_mask=mode_mask, V=V, V_inv=V_inv,
            )
            if return_layer_states:
                layer_states.append(f)

        f = self.readout(f)
        if return_layer_states:
            return f, layer_states
        return f
