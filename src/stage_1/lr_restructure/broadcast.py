from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.pw_bases import minimum_image
from utils.se3_utils import (BatchedGatedNonlinearity,
    clebsch_gordan_matrix, get_spherical_harmonics, init_linear_tensor_, scalar_ffn,
    softmax_over_neighbors,
)


class MasterBroadcastAttention(nn.Module):
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
        self.degrees = degrees
        self.max_width = 2 * max_degree + 1
        hi_degrees = [l for l in degrees if l > 0]

        self.q_self_w = nn.Parameter(torch.empty(len(degrees), feature_dim, feature_dim))
        init_linear_tensor_(self.q_self_w.view(-1, feature_dim), "linear")

        self.readout_K_w = nn.Parameter(torch.empty(len(degrees), feature_dim, feature_dim))
        init_linear_tensor_(self.readout_K_w.view(-1, feature_dim), "linear")
        self.readout_V_w = nn.Parameter(torch.empty(len(degrees), feature_dim, feature_dim))
        init_linear_tensor_(self.readout_V_w.view(-1, feature_dim), "linear")

        Q_lift_blocks = []
        for l in degrees:
            cg = clebsch_gordan_matrix(l, 0)
            Q_J = cg[l].view(2 * l + 1, 2 * l + 1)
            block = torch.zeros(self.max_width, self.max_width)
            block[:2 * l + 1, :2 * l + 1] = Q_J
            Q_lift_blocks.append(block)
        self.register_buffer("Q_lift", torch.stack(Q_lift_blocks, dim=0))  

        self.radial_KV = radial_net(num_basis=32, hidden_dim=hidden_dim, num_heads=2 * len(degrees), use_norms=use_norms)

        self.mlp0 = scalar_ffn(feature_dim, hidden_dim, use_norms=use_norms)
        if hi_degrees:
            self.mlp_hi_w = nn.Parameter(torch.empty(len(hi_degrees), feature_dim, feature_dim))
            init_linear_tensor_(self.mlp_hi_w.view(-1, feature_dim), "linear")
            self.gate_hi = BatchedGatedNonlinearity(feature_dim, degrees=hi_degrees)


    def forward(
        self,
        f_tilde: Dict[int, torch.Tensor],
        g_a: torch.Tensor,
        X_a: torch.Tensor,
        x: torch.Tensor,
        node_mask: torch.Tensor,
        V: torch.Tensor,
        V_inv: Optional[torch.Tensor] = None,
    ) -> Dict[int, torch.Tensor]:
        B, N, C, _ = f_tilde[0].shape
        N_a = X_a.shape[1]
        H, Ch = self.num_heads, self.head_dim
        L, W = len(self.degrees), self.max_width
        device, dtype = x.device, x.dtype

        f_tilde_stack = torch.stack([F.pad(f_tilde[l], (0, W - (2 * l + 1))) for l in self.degrees], dim=0,)
        q_stack = torch.einsum("loc,lbnck->lbnok", self.q_self_w, f_tilde_stack)
        Q_full = q_stack.view(L, B, N, H, Ch, W)

        x_rel_raw = x.unsqueeze(2) - X_a.unsqueeze(1)
        x_rel = minimum_image(x_rel_raw.reshape(B, N * N_a, 3), V, V_inv).reshape(B, N, N_a, 3)
        has_dir = (x_rel.norm(dim=-1) > 1e-7).to(dtype).view(B, N, N_a, 1, 1)
        x_rel_flat = x_rel.reshape(B * N * N_a, 3)
        r = x_rel_flat.norm(dim=-1, keepdim=True)

        g_a_sq = g_a.squeeze(-1)
        readout_KV_w = torch.stack([self.readout_K_w, self.readout_V_w], dim=0)
        content_all = torch.einsum("kloc,bac->kbalo", readout_KV_w, g_a_sq)
        content_perm = content_all.permute(3, 0, 1, 2, 4).unsqueeze(-1)

        phi_stack = self.radial_KV(r).view(-1, L, 2).permute(1, 2, 0).unsqueeze(-1)  # (L, 2, N, 1)
        Y_stack = torch.zeros(L, x_rel_flat.shape[0], W, device=device, dtype=dtype)
        for l in self.degrees:
            Y_stack[l, :, :2 * l + 1] = get_spherical_harmonics(l, x_rel_flat)
        Q_lift = self.Q_lift.to(device=device, dtype=dtype)
        QTY_stack = torch.einsum("lqi,lni->lnq", Q_lift, Y_stack).view(L, B, N, N_a, W, 1)

        degree_mask = torch.stack([torch.ones_like(has_dir) if l == 0 else has_dir for l in self.degrees], dim=0,)
        phi_kv = phi_stack.view(L, 2, B, N, N_a, 1, 1)
        W_KV_stack = phi_kv * QTY_stack.unsqueeze(1) * degree_mask.unsqueeze(1)

        KV_lifted_stack = torch.einsum("lkbnaij,lkbacj->lkbnaci", W_KV_stack, content_perm)
        Kmaster_full = KV_lifted_stack[:, 0].view(L, B, N, N_a, H, Ch, W)
        Vmaster_full = KV_lifted_stack[:, 1].view(L, B, N, N_a, H, Ch, W)

        Q = {l: Q_full[l, ..., :2 * l + 1] for l in self.degrees}
        K_lifted = {l: Kmaster_full[l, ..., :2 * l + 1] for l in self.degrees}
        V_lifted = {l: Vmaster_full[l, ..., :2 * l + 1] for l in self.degrees}

        Q_concat = torch.cat([Q[l] for l in self.degrees], dim=-1)
        Kmaster_concat = torch.cat([K_lifted[l] for l in self.degrees], dim=-1)

        score_master = torch.einsum("bnhcd,bnahcd->bnah", Q_concat, Kmaster_concat) / self.scale

        beta_master = softmax_over_neighbors(score_master.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)

        Vmaster_concat = torch.cat([V_lifted[l] for l in self.degrees], dim=-1)

        o = torch.einsum("bnah,bnahcd->bnhcd", beta_master, Vmaster_concat)
        o_flat = o.reshape(B, N, C, -1)

        mixed: Dict[int, torch.Tensor] = {}
        row_start = 0
        o_by_l = {}
        for l in self.degrees:
            row_end = row_start + (2 * l + 1)
            o_by_l[l] = o_flat[..., row_start:row_end]
            row_start = row_end
        mixed[0] = self.mlp0(o_by_l[0].squeeze(-1)).unsqueeze(-1)

        hi_degrees = [l for l in self.degrees if l > 0]
        if hi_degrees:
            o_hi_stack = torch.stack([F.pad(o_by_l[l], (0, W - (2 * l + 1))) for l in hi_degrees], dim=0)
            mixed_hi_stack = torch.einsum("loc,lbnck->lbnok", self.mlp_hi_w, o_hi_stack)
            mixed_hi_padded = self.gate_hi({l: mixed_hi_stack[i] for i, l in enumerate(hi_degrees)})
            for l in hi_degrees:
                mixed[l] = mixed_hi_padded[l][..., :2 * l + 1]

        return mixed
