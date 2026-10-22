from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.se3_utils import (
    clebsch_gordan_matrix, get_spherical_harmonics, init_linear_tensor_,
    scalar_ffn, scatter_softmax, straight_through_norm,
)


class LRDirectSumW(nn.Module):
    def __init__(self, radial_net: nn.Module, max_degree: int, k_max: int):
        super().__init__()
        self.radial_net = radial_net
        self.max_degree = max_degree
        self.k_max = k_max

        row_sizes = [2 * l + 1 for l in range(max_degree + 1)]
        col_sizes = [2 * k + 1 for k in range(k_max + 1)]
        row_starts = [sum(row_sizes[:i]) for i in range(len(row_sizes))]
        col_starts = [sum(col_sizes[:i]) for i in range(len(col_sizes))]
        total_rows, total_cols = sum(row_sizes), sum(col_sizes)
        max_width = 2 * (max_degree + k_max) + 1

        copies: List[Tuple[int, int]] = [] 
        Q_blocks = []
        for l in range(max_degree + 1):
            for k in range(k_max + 1):
                cg = clebsch_gordan_matrix(l, k)
                for J in range(abs(l - k), l + k + 1):
                    block = torch.zeros(total_rows, total_cols, max_width)
                    Q_J = cg[J].view(row_sizes[l], col_sizes[k], 2 * J + 1)
                    block[
                        row_starts[l]:row_starts[l] + row_sizes[l],
                        col_starts[k]:col_starts[k] + col_sizes[k],
                        :2 * J + 1,
                    ] = Q_J
                    Q_blocks.append(block)
                    copies.append((l, J))
        self.register_buffer("Q_GLOBAL", torch.stack(Q_blocks, dim=0))  # (n_total, R, C, W)
        self.copies = copies
        self.total_rows, self.total_cols, self.max_width = total_rows, total_cols, max_width
        self.row_starts, self.row_sizes = row_starts, row_sizes

        distinct_Js = sorted({J for _, J in copies})
        j_to_pos = {J: i for i, J in enumerate(distinct_Js)}
        self.distinct_Js = distinct_Js
        self.register_buffer(
            "copy_to_j_idx", torch.tensor([j_to_pos[J] for _, J in copies], dtype=torch.long),
            persistent=False,
        )

    def forward(self, f: Dict[int, torch.Tensor], x: torch.Tensor) -> Dict[int, torch.Tensor]:
        device, dtype = x.device, x.dtype
        N = x.shape[0]

        out: Dict[int, torch.Tensor] = {}
        if N == 0:
            C = next(iter(f.values())).shape[-2]
            for l in range(self.max_degree + 1):
                out[l] = torch.zeros(N, C, 2 * l + 1, device=device, dtype=dtype)
            return out

        r = x.norm(dim=-1, keepdim=True)
        phi_global = self.radial_net(r).T.unsqueeze(-1)  # (n_total, N, 1)

        Y_by_j = torch.stack(
            [F.pad(get_spherical_harmonics(J, x), (0, self.max_width - (2 * J + 1)))
             for J in self.distinct_Js],
            dim=0,
        )  # (n_distinct_J, N, max_width)
        Y_global = Y_by_j[self.copy_to_j_idx]  # (n_total, N, max_width)

        Q_GLOBAL = self.Q_GLOBAL.to(device=device, dtype=dtype)
        QTY_global = torch.einsum("crqi,cni->cnrq", Q_GLOBAL, Y_global)          
        W_full = (phi_global.view(-1, N, 1, 1) * QTY_global).sum(dim=0)          

        f_concat = torch.cat([f[k] for k in range(self.k_max + 1)], dim=-1)
        out_concat = f_concat @ W_full.transpose(-1, -2)

        for l in range(self.max_degree + 1):
            rs = self.row_starts[l]
            out[l] = out_concat[..., rs:rs + self.row_sizes[l]]
        return out


class PairReadout(nn.Module):
    def __init__(self, C: int, hidden: int, k_max: int, floor: float = 0.5, use_norms: bool = True):
        super().__init__()
        self.k_max = k_max
        self.floor = floor
        self.use_norms = use_norms
        self.max_width = 2 * k_max + 1

        self.W_equiv_proj = nn.Parameter(torch.empty(k_max + 1, hidden, C))
        self.W_h1 = nn.Parameter(torch.empty(k_max + 1, hidden, hidden))
        self.W_out = nn.Parameter(torch.empty(k_max + 1, 1, hidden))
        for i in range(k_max + 1):
            init_linear_tensor_(self.W_equiv_proj[i], "linear")
            init_linear_tensor_(self.W_h1[i], "silu")
            init_linear_tensor_(self.W_out[i], "linear")
        self.g_h1 = nn.Parameter(torch.ones(k_max + 1, hidden))

    def forward(self, cats: List[torch.Tensor]) -> torch.Tensor:
        cats_padded = torch.stack([F.pad(x, (0, self.max_width - x.shape[-1])) for x in cats], dim=0,)  # (k_max+1, E, C, max_width)
        v = torch.einsum("khc,kecw->kehw", self.W_equiv_proj, cats_padded)  # (k_max+1, E, hidden, max_width)
        h0 = straight_through_norm(v, dim=-1, floor=self.floor)  # (k_max+1, E, hidden)
        h1 = h0 @ self.W_h1.transpose(-1, -2)
        if self.use_norms:
            h1 = F.rms_norm(h1, (h1.shape[-1],), None, None) * self.g_h1.unsqueeze(1)
        h1 = F.silu(h1)
        out = (h1 @ self.W_out.transpose(-1, -2)).squeeze(-1)
        return out.sum(dim=0)


class SRAttention(nn.Module):
    def __init__(
        self, radial_net, max_degree: int, feature_dim: int, hidden_dim: int,
        k_max: int, edge_feat_dim: int, use_norms: bool = True, num_basis: int = 32,
    ):
        super().__init__()
        self.max_degree = max_degree
        self.k_max = k_max

        n_total = sum((l + k + 1) - abs(l - k) for l in range(max_degree + 1) for k in range(k_max + 1))
        radial_V_sr = radial_net(num_basis=num_basis, hidden_dim=hidden_dim, num_heads=n_total, use_norms=use_norms)
        self.direct_sum_w = LRDirectSumW(radial_V_sr, max_degree, k_max)

        self.pair_readout = PairReadout(2 * feature_dim, hidden_dim, k_max, use_norms=use_norms)

        self.edge_input_norm = nn.RMSNorm(1 + edge_feat_dim) if use_norms else nn.Identity()
        self.psi_e = scalar_ffn(1 + edge_feat_dim, hidden_dim, use_norms=use_norms)
        self.psi_e_out = nn.Linear(1 + edge_feat_dim, 1)
        init_linear_tensor_(self.psi_e_out.weight, "leaky_relu")
        nn.init.zeros_(self.psi_e_out.bias)

    def forward(
        self,
        f_in_i_e: Dict[int, torch.Tensor],
        f0_j_e: Dict[int, torch.Tensor],
        x_rel_e: torch.Tensor,
        edge_feat_e: torch.Tensor,
        edge_b: torch.Tensor,
        edge_i: torch.Tensor,
        batch_size: int,
        num_atoms: int,
        edge_valid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        
        cats = [torch.cat([f_in_i_e[k], f0_j_e[k]], dim=-2) for k in range(self.k_max + 1)]
        score = self.pair_readout(cats)

        dist = x_rel_e.norm(dim=-1, keepdim=True)
        edge_feats = self.edge_input_norm(torch.cat([dist, edge_feat_e], dim=-1))
        psi = self.psi_e_out(self.psi_e(edge_feats)).squeeze(-1)

        scores_e = psi * score

        num_receivers = batch_size * num_atoms
        if edge_valid is None:
            receiver_flat = edge_b * num_atoms + edge_i
            alpha_e = scatter_softmax(scores_e, receiver_flat, num_receivers)
        else:
            receiver_flat = torch.where(edge_valid, edge_b * num_atoms + edge_i, torch.full_like(edge_b, num_receivers),)
            alpha_e = scatter_softmax(scores_e, receiver_flat, num_receivers + 1)
            alpha_e = alpha_e * edge_valid.to(alpha_e.dtype)

        Wv_j_sr_flat = self.direct_sum_w(f0_j_e, x_rel_e)

        return alpha_e, Wv_j_sr_flat
