from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import cue_backend
from utils.se3_utils import (
    BatchedRadialNetwork, clebsch_gordan_matrix, get_spherical_harmonics, init_linear_,
    init_sequential_, scalar_ffn, scatter_softmax, straight_through_norm,
)
from utils.spherical_harm import get_spherical_harmonics_many


class LRDirectSumW(nn.Module):
    def __init__(
        self,
        radial_nets: nn.ModuleDict,
        max_degree: int,
        k_max: int,
        feature_dim: Optional[int] = None,
        radial_net_cls=None,
        cue_hidden_dim: int = 64,
        use_cuequivariance: bool = False,
    ):
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
        self.register_buffer("Q_GLOBAL", torch.stack(Q_blocks, dim=0)) 
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

        self.use_cuequivariance = use_cuequivariance
        self._cue_ready = False
        want_cue = use_cuequivariance and feature_dim is not None and radial_net_cls is not None
        if want_cue and cue_backend.cuequivariance_available():
            irreps_in1 = cue_backend.so3_irreps({k: feature_dim for k in range(k_max + 1)})
            irreps_in2 = "+".join(f"1x{J}" for J in distinct_Js)
            irreps_out = cue_backend.so3_irreps({l: feature_dim for l in range(max_degree + 1)})
            self.cue_tp = cue_backend.build_channelwise_tp(
                irreps_in1, irreps_in2, irreps_out, device=None, dtype=None,
            )
            self.cue_radial = radial_net_cls(
                num_basis=8, hidden_dim=cue_hidden_dim, num_heads=self.cue_tp.weight_numel,
            )
            self.feature_dim = feature_dim
            self._cue_ready = True

    def forward(self, f: Dict[int, torch.Tensor], x: torch.Tensor) -> Dict[int, torch.Tensor]:
        if self._cue_ready:
            return self._forward_cue(f, x)
        return self._forward_reference(f, x)

    def _forward_reference(self, f: Dict[int, torch.Tensor], x: torch.Tensor) -> Dict[int, torch.Tensor]:
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
        ) 

        Y_by_j = torch.stack(
            [F.pad(get_spherical_harmonics(J, x), (0, self.max_width - (2 * J + 1)))
             for J in self.distinct_Js],
            dim=0,
        )  
        Y_global = Y_by_j[self.copy_to_j_idx]  

        Q_GLOBAL = self.Q_GLOBAL.to(device=device, dtype=dtype)
        QTY_global = torch.einsum("crqi,cni->cnrq", Q_GLOBAL, Y_global)
        W_full = (phi_global.view(-1, N, 1, 1) * QTY_global).sum(dim=0)

        f_concat = torch.cat([f[k] for k in range(self.k_max + 1)], dim=-1)
        out_concat = torch.einsum("nij,ncj->nci", W_full, f_concat)

        for l in range(self.max_degree + 1):
            rs = self.row_starts[l]
            out[l] = out_concat[..., rs:rs + self.row_sizes[l]]
        return out

    def _forward_cue(self, f: Dict[int, torch.Tensor], x: torch.Tensor) -> Dict[int, torch.Tensor]:
        device, dtype = x.device, x.dtype
        N = x.shape[0]
        C = self.feature_dim

        out: Dict[int, torch.Tensor] = {}
        if N == 0:
            for l in range(self.max_degree + 1):
                out[l] = torch.zeros(N, C, 2 * l + 1, device=device, dtype=dtype)
            return out

        r = x.norm(dim=-1, keepdim=True)
        weights = self.cue_radial(r).view(N, -1)

        Y = get_spherical_harmonics_many(self.distinct_Js, x, use_cuequivariance=True)
        Y_concat = torch.cat([Y[J] for J in self.distinct_Js], dim=-1)
        
        f_concat = torch.cat(
            [
                (f[k] @ cue_backend.e3nn_to_cue_basis(k, device, dtype)).reshape(N, -1)
                for k in range(self.k_max + 1)
            ],
            dim=-1,
        )

        out_concat = self.cue_tp(f_concat, Y_concat, weights)

        start = 0
        for l in range(self.max_degree + 1):
            width = C * (2 * l + 1)
            out_cue_basis = out_concat[..., start:start + width].view(N, C, 2 * l + 1)
            out[l] = out_cue_basis @ cue_backend.e3nn_to_cue_basis(l, device, dtype).T
            start += width
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
        norms = []
        for k, x in enumerate(cats):
            v = self.equiv_proj[k](x.transpose(-1, -2)).transpose(-1, -2)
            norms.append(straight_through_norm(v, dim=-1, floor=self.floor))
        h0 = torch.stack(norms, dim=0)
        h1 = torch.einsum("keh,kgh->keg", h0, self.W_h1)
        if self.use_norms:
            h1 = F.rms_norm(h1, (h1.shape[-1],), None, None) * self.g_h1.unsqueeze(1)
        h1 = F.silu(h1)
        out = torch.einsum("keh,koh->keo", h1, self.W_out).squeeze(-1)
        return out.sum(dim=0)


class SRAttention(nn.Module):
    def __init__(
        self, radial_net, max_degree: int, feature_dim: int, hidden_dim: int,
        k_max: int, edge_feat_dim: int, use_norms: bool = True, use_cuequivariance: bool = False,
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
        self.direct_sum_w = LRDirectSumW(
            radial_V_sr, max_degree, k_max, feature_dim=feature_dim,
            radial_net_cls=radial_net, cue_hidden_dim=hidden_dim, use_cuequivariance=use_cuequivariance,
        )

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
