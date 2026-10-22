from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from utils.se3_utils import (
    EquivariantLinear, EquivariantRMSNorm,
    BatchedGatedNonlinearity,
)
from stage_1.model import Readout

from stage_1.lr_restructure.aggregation import MasterNodeAggregator
from stage_1.lr_restructure.update import MasterNodeUpdate
from stage_1.lr_restructure.broadcast import MasterBroadcastAttention
from stage_1.lr_restructure.sr_attention import SRAttention


class LRLayer(nn.Module):
    def __init__(
        self, radial_net, max_degree: int, feature_dim: int, hidden_dim: int,
        edge_cutoff: float, n_max: int, pw_e_cutoff: Optional[float], k_max: int,
        edge_feat_dim: int, n_master_nodes: int, num_broadcast_heads: int = 4,
        rbf_num_basis: int = 12, rbf_min: float = 0.3, rbf_cutoff: float = 25.0,
        sigma: Optional[float] = None, alpha_floor: float = 1e-4, use_norms: bool = True,
        use_long_range: bool = True, sr_num_basis: int = 32,
    ):
        super().__init__()
        self.max_degree = max_degree
        self.k_max = k_max
        self.use_long_range = use_long_range

        active_ceiling = k_max
        self.active_ceiling = active_ceiling
        active_degrees = list(range(active_ceiling + 1))

        self.sr_attn = SRAttention(
            radial_net, active_ceiling, feature_dim, hidden_dim, k_max=min(k_max, active_ceiling),
            edge_feat_dim=edge_feat_dim, use_norms=use_norms, num_basis=sr_num_basis,
        )
        self.W_V_self_linear = nn.ModuleDict({
            str(l): EquivariantLinear(feature_dim, feature_dim) for l in active_degrees
        })
        self.W_V_self_norm = (
            EquivariantRMSNorm(feature_dim, degrees=active_degrees) if use_norms else nn.Identity()
        )
        self.sigma_gate = BatchedGatedNonlinearity(feature_dim, degrees=active_degrees)
        self.layer_norm_in = (
            EquivariantRMSNorm(feature_dim, degrees=active_degrees) if use_norms else nn.Identity()
        )
        self.layer_norm_mid = (
            EquivariantRMSNorm(feature_dim, degrees=active_degrees) if use_norms else nn.Identity()
        )
        self.layer_norm_out = (
            EquivariantRMSNorm(feature_dim, degrees=active_degrees) if use_norms else nn.Identity()
        )
        self.readout = Readout(feature_dim, hidden_dim, use_norms=use_norms, degrees=active_degrees)

        if self.use_long_range:
            self.aggregator = MasterNodeAggregator(
                feature_dim, n_master_nodes, hidden_dim, rbf_num_basis, rbf_min, rbf_cutoff,
            )
            self.update = MasterNodeUpdate(sigma=sigma, alpha_floor=alpha_floor)
            self.W_master_self = EquivariantLinear(feature_dim, feature_dim)
            self.master_ln = EquivariantRMSNorm(feature_dim, degree=0) if use_norms else nn.Identity()
            self.broadcast = MasterBroadcastAttention(
                radial_net, active_ceiling, feature_dim, hidden_dim,
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
        edge_valid: Optional[torch.Tensor],
        kvecs: torch.Tensor,
        mode_mask: torch.Tensor,
        V: torch.Tensor,
        V_inv: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[int, torch.Tensor], torch.Tensor, torch.Tensor]:
        B, N, _ = x_wrapped.shape
        active_degrees = range(self.active_ceiling + 1)

        with torch.profiler.record_function("lr_local_sr"):
            f_in_active = self.layer_norm_in({l: f_in[l] for l in active_degrees})

            receiver_flat = edge_b * N + edge_i
            sender_flat = edge_b * N + edge_j
            f_in_i_e = {
                k: f_in_active[k].reshape(B * N, *f_in_active[k].shape[2:]).index_select(0, receiver_flat)
                for k in range(self.k_max + 1)
            }
            f0_j_e = {
                k: f0[k].reshape(B * N, *f0[k].shape[2:]).index_select(0, sender_flat)
                for k in range(self.k_max + 1)
            }
            with torch.profiler.record_function("lr_sr_attn_only"):
                alpha_e, Wv_j_sr_flat = self.sr_attn(
                    f_in_i_e, f0_j_e, x_rel_e, edge_feat_e, edge_b, edge_i, batch_size=B, num_atoms=N,
                    edge_valid=edge_valid,
                )

            self_terms_raw = {l: self.W_V_self_linear[str(l)](f_in_active[l]) for l in active_degrees}
            self_terms = self.W_V_self_norm(self_terms_raw)

            neighbor_raws: Dict[int, torch.Tensor] = {}
            for l in active_degrees:
                C = f_in_active[l].shape[2]
                contrib_e = alpha_e.view(-1, 1, 1) * Wv_j_sr_flat[l]
                neighbor_raws[l] = torch.zeros(
                    B * N, C, 2 * l + 1, device=x_wrapped.device, dtype=x_wrapped.dtype,
                ).index_add(
                    0, receiver_flat, contrib_e,
                ).view(B, N, C, 2 * l + 1)
            neighbor_terms = self.sigma_gate(neighbor_raws)

            pre_mid = {l: f_in_active[l] + self_terms[l] + neighbor_terms[l] for l in active_degrees}
            f_out_active = self.layer_norm_mid(pre_mid)

            f_readout_active = self.readout(f_out_active)
            out = {l: f_readout_active[l] + f_out_active[l] for l in active_degrees}

            f_tilde = dict(f_in)
            f_tilde.update(out)

        if not self.use_long_range:
            return f_tilde, g_a_in, X_a_in

        with torch.profiler.record_function("lr_aggregation"):
            pooled, X_a_new = self.aggregator(
                g_a_in.squeeze(-1), f_in_active[0].squeeze(-1), x_wrapped, X_a_in, node_mask, V, V_inv,
            )

        with torch.profiler.record_function("lr_update"):
            g_a_new = self.update(pooled, X_a_new, V, kvecs, mode_mask)
            g_a_out = self.master_ln(self.W_master_self(g_a_in) + g_a_new.unsqueeze(-1))

        with torch.profiler.record_function("lr_broadcast"):
            f_next_active = self.broadcast(
                {l: f_tilde[l] for l in active_degrees}, g_a_out, X_a_new, x_wrapped, node_mask, V, V_inv,
            )
            f_next = dict(f_tilde)
            f_next.update(f_next_active)

        return f_next, g_a_out, X_a_new
