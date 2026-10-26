from typing import Dict

import torch
import torch.nn as nn

from utils.pw_bases import minimum_image
from utils.se3_utils import (
    EquivariantLinear, GatedNonlinearity,
    clebsch_gordan_matrix, get_spherical_harmonics, init_linear_tensor_, scalar_ffn,
    softmax_over_neighbors,
)


class MasterBroadcastAttentionReference(nn.Module):
    def __init__(
        self, radial_net, max_degree: int, feature_dim: int, hidden_dim: int,
        num_heads: int = 4, use_norms: bool = True,
    ):
        super().__init__()
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"
        self.max_degree = max_degree
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.scale = self.head_dim ** 0.5

        degrees = list(range(max_degree + 1))

        self.q_self_w = nn.ParameterDict()
        for l in degrees:
            w = nn.Parameter(torch.empty(feature_dim, feature_dim))
            init_linear_tensor_(w, "linear")
            self.q_self_w[str(l)] = w

        self.readout_K_w = nn.Parameter(torch.empty(len(degrees), feature_dim, feature_dim))
        init_linear_tensor_(self.readout_K_w.view(-1, feature_dim), "linear")
        self.readout_V_w = nn.Parameter(torch.empty(len(degrees), feature_dim, feature_dim))
        init_linear_tensor_(self.readout_V_w.view(-1, feature_dim), "linear")

        self.radial_KV = radial_net(num_basis=32, hidden_dim=hidden_dim, num_heads=2 * len(degrees), use_norms=use_norms)

        self.mlp0 = scalar_ffn(feature_dim, hidden_dim, use_norms=use_norms)
        self.mlp_hi = nn.ModuleDict({
            str(l): EquivariantLinear(feature_dim, feature_dim) for l in degrees if l > 0
        })
        self.gate_hi = nn.ModuleDict({str(l): GatedNonlinearity(feature_dim) for l in degrees if l > 0})

    def forward(
        self,
        f_tilde: Dict[int, torch.Tensor],
        g_a: torch.Tensor,
        X_a: torch.Tensor,
        x: torch.Tensor,
        node_mask: torch.Tensor,
        V: torch.Tensor,
    ) -> Dict[int, torch.Tensor]:
        B, N, C, _ = f_tilde[0].shape
        N_a = X_a.shape[1]
        H, Ch = self.num_heads, self.head_dim

        Q: Dict[int, torch.Tensor] = {}
        for l in range(self.max_degree + 1):
            Q[l] = torch.einsum("oc,bnck->bnok", self.q_self_w[str(l)], f_tilde[l])

        x_rel_raw = x.unsqueeze(2) - X_a.unsqueeze(1)                         
        x_rel = minimum_image(x_rel_raw.reshape(B, N * N_a, 3), V).reshape(B, N, N_a, 3)
        has_dir = (x_rel.norm(dim=-1) > 1e-7).to(x.dtype).view(B, N, N_a, 1, 1)
        x_rel_flat = x_rel.reshape(B * N * N_a, 3)
        r = x_rel_flat.norm(dim=-1, keepdim=True)

        g_a_sq = g_a.squeeze(-1)                                            
        K_content_all = torch.einsum("loc,bac->balo", self.readout_K_w, g_a_sq)  
        V_content_all = torch.einsum("loc,bac->balo", self.readout_V_w, g_a_sq)

        phi_kv_all = self.radial_KV(r)  # (N, 2 * (max_degree + 1))

        K_lifted: Dict[int, torch.Tensor] = {}
        V_lifted: Dict[int, torch.Tensor] = {}
        for l in range(self.max_degree + 1):
            K_content = K_content_all[:, :, l, :].unsqueeze(-1)
            V_content = V_content_all[:, :, l, :].unsqueeze(-1)

            phi_kv = phi_kv_all[:, 2 * l:2 * l + 2].T.unsqueeze(-1)  # (2, N, 1)
            cg = clebsch_gordan_matrix(l, 0, dtype=x.dtype, device=x.device)
            Q_J = cg[l]
            Y_J = get_spherical_harmonics(l, x_rel_flat)
            QTY = torch.einsum("ji,ni->nj", Q_J, Y_J)
            W_K = (phi_kv[0] * QTY).view(B, N, N_a, 2 * l + 1, 1)
            W_V = (phi_kv[1] * QTY).view(B, N, N_a, 2 * l + 1, 1)
            if l > 0:
                W_K = W_K * has_dir
                W_V = W_V * has_dir

            K_lifted[l] = torch.einsum("bnaij,bacj->bnaci", W_K, K_content)
            V_lifted[l] = torch.einsum("bnaij,bacj->bnaci", W_V, V_content)

        Q_concat = torch.cat(
            [Q[l].view(B, N, H, Ch, 2 * l + 1) for l in range(self.max_degree + 1)], dim=-1,
        )
        Kmaster_concat = torch.cat(
            [K_lifted[l].view(B, N, N_a, H, Ch, 2 * l + 1) for l in range(self.max_degree + 1)], dim=-1,
        )

        score_master = torch.einsum("bnhcd,bnahcd->bnah", Q_concat, Kmaster_concat) / self.scale

        beta_master = softmax_over_neighbors(score_master.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)

        Vmaster_concat = torch.cat(
            [V_lifted[l].view(B, N, N_a, H, Ch, 2 * l + 1) for l in range(self.max_degree + 1)], dim=-1,
        )

        o = torch.einsum("bnah,bnahcd->bnhcd", beta_master, Vmaster_concat)
        o_flat = o.reshape(B, N, C, -1)

        out: Dict[int, torch.Tensor] = {}
        row_start = 0
        for l in range(self.max_degree + 1):
            row_end = row_start + (2 * l + 1)
            o_l = o_flat[..., row_start:row_end]
            if l == 0:
                mixed = self.mlp0(o_l.squeeze(-1)).unsqueeze(-1)
            else:
                mixed = self.gate_hi[str(l)](self.mlp_hi[str(l)](o_l))
            out[l] = mixed
            row_start = row_end
        return out
