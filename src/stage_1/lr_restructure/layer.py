from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from stage_1.attention import SRAttention
from utils.se3_utils import (
    EquivariantLinear, EquivariantRMSNorm, IrrepLinear, BatchedEquivariantRMSNorm,
    BatchedGatedNonlinearity,
)
from stage_1.model import Readout

from stage_1.lr_restructure.aggregation import MasterNodeAggregator
from stage_1.lr_restructure.update import MasterNodeUpdate
from stage_1.lr_restructure.broadcast import MasterBroadcastAttention


class LRLayer(nn.Module):
    def __init__(
        self, radial_net, max_degree: int, feature_dim: int, hidden_dim: int,
        edge_cutoff: float, n_max: int, pw_e_cutoff: Optional[float], k_max: int,
        edge_feat_dim: int, n_master_nodes: int, num_broadcast_heads: int = 4,
        rbf_num_basis: int = 12, rbf_min: float = 0.3, rbf_cutoff: float = 25.0,
        sigma: Optional[float] = None, alpha_floor: float = 1e-4, use_norms: bool = True,
    ):
        super().__init__()
        self.max_degree = max_degree
        self.k_max = k_max

        self.sr_attn = SRAttention(
            radial_net, max_degree, feature_dim, hidden_dim, k_max=k_max,
            edge_feat_dim=edge_feat_dim, use_norms=use_norms,
        )
        self.W_V_self_linear = nn.ModuleDict({
            str(l): EquivariantLinear(feature_dim, feature_dim) for l in range(max_degree + 1)
        })
        norm_degrees = list(range(max_degree + 1))
        self.W_V_self_norm = (
            BatchedEquivariantRMSNorm(feature_dim, degrees=norm_degrees) if use_norms else nn.Identity()
        )
        self.sigma_gate = BatchedGatedNonlinearity(feature_dim, degrees=norm_degrees)
        self.layer_norm_in = (
            BatchedEquivariantRMSNorm(feature_dim, degrees=norm_degrees) if use_norms else nn.Identity()
        )
        self.layer_norm_mid = (
            BatchedEquivariantRMSNorm(feature_dim, degrees=norm_degrees) if use_norms else nn.Identity()
        )
        self.layer_norm_out = (
            BatchedEquivariantRMSNorm(feature_dim, degrees=norm_degrees) if use_norms else nn.Identity()
        )
        self.readout = Readout(feature_dim, hidden_dim, use_norms=use_norms)

        self.aggregator = MasterNodeAggregator(
            feature_dim, n_master_nodes, hidden_dim, rbf_num_basis, rbf_min, rbf_cutoff,
        )
        self.update = MasterNodeUpdate(sigma=sigma, alpha_floor=alpha_floor)
        self.W_master_self = IrrepLinear(feature_dim)
        self.master_ln = EquivariantRMSNorm(feature_dim, degree=0) if use_norms else nn.Identity()
        self.broadcast = MasterBroadcastAttention(
            radial_net, max_degree, feature_dim, hidden_dim,
            num_heads=num_broadcast_heads, use_norms=use_norms,
        )

    def forward(
        self,
        f_in: Dict[int, torch.Tensor],
        f0: Dict[int, torch.Tensor],
        g_a_in: torch.Tensor,
        X_a_in: torch.Tensor,
        x_wrapped: torch.Tensor,
        node_mask: torch.Tensor,
        edge_b: torch.Tensor,
        edge_i: torch.Tensor,
        edge_j: torch.Tensor,
        x_rel_e: torch.Tensor,
        edge_feat_e: torch.Tensor,
        kvecs: torch.Tensor,
        mode_mask: torch.Tensor,
        V: torch.Tensor,
    ) -> Tuple[Dict[int, torch.Tensor], torch.Tensor, torch.Tensor]:
        B, N, _ = x_wrapped.shape
        f_in = self.layer_norm_in(f_in)

        f_in_i_e = {k: f_in[k][edge_b, edge_i] for k in range(self.k_max + 1)}
        f0_j_e = {k: f0[k][edge_b, edge_j] for k in range(self.k_max + 1)}
        alpha_e, Wv_j_sr_flat = self.sr_attn(
            f_in_i_e, f0_j_e, x_rel_e, edge_feat_e, edge_b, edge_i, batch_size=B, num_atoms=N,
        )
        receiver_flat = edge_b * N + edge_i

        self_terms_raw = {l: self.W_V_self_linear[str(l)](f_in[l]) for l in range(self.max_degree + 1)}
        self_terms = self.W_V_self_norm(self_terms_raw)

        neighbor_raws: Dict[int, torch.Tensor] = {}
        for l in range(self.max_degree + 1):
            C = f_in[l].shape[2]
            contrib_e = alpha_e.view(-1, 1, 1) * Wv_j_sr_flat[l]
            neighbor_raws[l] = torch.zeros(
                B * N, C, 2 * l + 1, device=x_wrapped.device, dtype=x_wrapped.dtype,
            ).scatter_add(
                0, receiver_flat.view(-1, 1, 1).expand(-1, C, 2 * l + 1), contrib_e,
            ).view(B, N, C, 2 * l + 1)
        neighbor_terms = self.sigma_gate(neighbor_raws)

        pre_mid = {l: f_in[l] + self_terms[l] + neighbor_terms[l] for l in range(self.max_degree + 1)}
        f_out = self.layer_norm_mid(pre_mid)

        f_readout = self.readout(f_out)
        pre_out = {l: f_readout[l] + f_out[l] for l in range(self.max_degree + 1)}
        f_tilde = self.layer_norm_out(pre_out) 

        pooled, X_a_new = self.aggregator(
            g_a_in.squeeze(-1), f_in[0].squeeze(-1), x_wrapped, X_a_in, node_mask, V,
        )
        g_a_new = self.update(pooled, X_a_new, V, kvecs, mode_mask)
        g_a_out = self.master_ln(self.W_master_self(g_a_in) + g_a_new.unsqueeze(-1))

        f_next = self.broadcast(f_tilde, g_a_out, X_a_new, x_wrapped, node_mask, V)

        return f_next, g_a_out, X_a_new
