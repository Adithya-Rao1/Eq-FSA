from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.se3_utils import (
    BatchedRadialNetwork, clebsch_gordan_matrix, get_spherical_harmonics, init_linear_,
    init_sequential_, scalar_ffn, scatter_softmax, straight_through_norm,
)


class LRDirectSumW(nn.Module):
    def __init__(self, radial_nets: nn.ModuleDict, max_degree: int, k_max: int):
        super().__init__()
        self.radial_nets = radial_nets
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
        phi_global = torch.cat(
            [self.radial_nets[str(l)](r) for l in range(self.max_degree + 1)], dim=0,
        )  # (n_total, N, 1)

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
        out_concat = torch.einsum("nij,ncj->nci", W_full, f_concat)

        for l in range(self.max_degree + 1):
            rs = self.row_starts[l]
            out[l] = out_concat[..., rs:rs + self.row_sizes[l]]
        return out


class BatchedNormedPairReadout(nn.Module):
    def __init__(self, C: int, hidden: int, k_max: int, floor: float = 0.5, use_norms: bool = True):
        super().__init__()
        self.k_max = k_max
        self.floor = floor
        self.use_norms = use_norms

        self.equiv_proj = nn.ModuleList([nn.Linear(C, hidden, bias=False) for _ in range(k_max + 1)])
        for lin in self.equiv_proj:
            init_linear_(lin, "linear")

        h1_templates = [nn.Linear(hidden, hidden, bias=False) for _ in range(k_max + 1)]
        for lin in h1_templates:
            init_linear_(lin, "silu")
        out_templates = [nn.Linear(hidden, 1, bias=False) for _ in range(k_max + 1)]
        for lin in out_templates:
            init_linear_(lin, "linear")
        self.W_h1 = nn.Parameter(torch.stack([t.weight.detach().clone() for t in h1_templates]))
        self.g_h1 = nn.Parameter(torch.ones(k_max + 1, hidden))
        self.W_out = nn.Parameter(torch.stack([t.weight.detach().clone() for t in out_templates]))

    def forward(self, cats: List[torch.Tensor]) -> torch.Tensor:
        K = self.k_max + 1
        W = 2 * self.k_max + 1
        equiv_proj_w = torch.stack([lin.weight for lin in self.equiv_proj], dim=0) 
        x_padded = torch.stack([F.pad(cats[k], (0, W - (2 * k + 1))) for k in range(K)], dim=0,) 
        v = torch.einsum("koc,kecw->keow", equiv_proj_w, x_padded)  
        h0 = straight_through_norm(v, dim=-1, floor=self.floor) 
        h1 = torch.einsum("keh,kgh->keg", h0, self.W_h1)
        if self.use_norms:
            h1 = F.rms_norm(h1, (h1.shape[-1],), None, None) * self.g_h1.unsqueeze(1)
        h1 = F.silu(h1)
        out = torch.einsum("keh,koh->keo", h1, self.W_out).squeeze(-1)
        return out.sum(dim=0)


class SRAttention(nn.Module):
    def __init__(
        self, radial_net, max_degree: int, feature_dim: int, hidden_dim: int,
        k_max: int, edge_feat_dim: int, use_norms: bool = True,
    ):
        super().__init__()
        self.max_degree = max_degree
        self.k_max = k_max

        radial_V_sr = nn.ModuleDict()
        for l in range(max_degree + 1):
            n_copies = sum((l + k + 1) - abs(l - k) for k in range(k_max + 1))
            radial_V_sr[str(l)] = BatchedRadialNetwork(
                n_copies=n_copies, num_basis=(2 * l + 1), hidden_dim=hidden_dim,
                radial_net_cls=radial_net, use_norms=use_norms,
            )
        self.direct_sum_w = LRDirectSumW(radial_V_sr, max_degree, k_max)

        self.pair_readout = BatchedNormedPairReadout(2 * feature_dim, hidden_dim, k_max, use_norms=use_norms)

        self.edge_input_norm = nn.RMSNorm(1 + edge_feat_dim) if use_norms else nn.Identity()
        self.psi_e = scalar_ffn(1 + edge_feat_dim, hidden_dim, use_norms=use_norms)
        self.psi_e_out = nn.Sequential(
            nn.RMSNorm(1 + edge_feat_dim) if use_norms else nn.Identity(),
            nn.LeakyReLU(),
            nn.Linear(1 + edge_feat_dim, 1),
        )
        init_sequential_(self.psi_e_out, "leaky_relu")

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
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        cats = [torch.cat([f_in_i_e[k], f0_j_e[k]], dim=-2) for k in range(self.k_max + 1)]
        score = self.pair_readout(cats)

        dist = x_rel_e.norm(dim=-1, keepdim=True)
        edge_feats = self.edge_input_norm(torch.cat([dist, edge_feat_e], dim=-1))
        psi = self.psi_e_out(self.psi_e(edge_feats)).squeeze(-1)

        scores_e = psi * score

        receiver_flat = edge_b * num_atoms + edge_i
        num_receivers = batch_size * num_atoms
        alpha_e = scatter_softmax(scores_e, receiver_flat, num_receivers)

        Wv_j_sr_flat = self.direct_sum_w(f0_j_e, x_rel_e)

        return alpha_e, Wv_j_sr_flat
