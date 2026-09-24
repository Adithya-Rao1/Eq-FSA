from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.pw_bases import minimum_image
from utils.se3_utils import softmax_over_neighbors, init_linear_

from stage_1.lr_restructure.rbf import LogSpacedRBF


class _PerNodeHead(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, num_rbf: int):
        super().__init__()
        self.A_Q = nn.Linear(feature_dim, hidden_dim)
        self.A_K = nn.Linear(feature_dim, hidden_dim)
        self.A_E = nn.Linear(num_rbf, hidden_dim)
        self.u = nn.Linear(hidden_dim, 1, bias=False)
        self.A_V = nn.Linear(feature_dim, feature_dim)
        self.h_gain = nn.Linear(num_rbf, 1, bias=False)
        for lin in (self.A_Q, self.A_K, self.A_E, self.A_V):
            init_linear_(lin, "leaky_relu")
        init_linear_(self.u, "linear")
        init_linear_(self.h_gain, "linear")

    def score(self, g_a: torch.Tensor, f_n: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        h = F.leaky_relu(self.A_Q(g_a).unsqueeze(1) + self.A_K(f_n) + self.A_E(e))
        return self.u(h).squeeze(-1)

    def value(self, f_n: torch.Tensor) -> torch.Tensor:
        return self.A_V(f_n)

    def radial_gain(self, e: torch.Tensor) -> torch.Tensor:
        return self.h_gain(e).squeeze(-1)


class MasterNodeAggregator(nn.Module):
    def __init__(
        self, feature_dim: int, n_master_nodes: int, hidden_dim: int = 64,
        rbf_num_basis: int = 12, rbf_min: float = 0.3, rbf_cutoff: float = 25.0,
    ):
        super().__init__()
        self.n_master_nodes = n_master_nodes
        self.rbf = LogSpacedRBF(rbf_num_basis, rbf_min, rbf_cutoff)
        self.heads = nn.ModuleList([
            _PerNodeHead(feature_dim, hidden_dim, rbf_num_basis) for _ in range(n_master_nodes)
        ])

    def forward(
        self, g_a: torch.Tensor, f_n: torch.Tensor, x: torch.Tensor, X_a: torch.Tensor,
        node_mask: torch.Tensor, V: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pooled_list, x_new_list = [], []
        for a, head in enumerate(self.heads):
            rel = minimum_image(x - X_a[:, a:a + 1, :], V)      
            d = rel.norm(dim=-1)                                
            e = self.rbf(d)                                      
            s = head.score(g_a[:, a, :], f_n, e)                
            w = softmax_over_neighbors(s, mask=node_mask)       
            pooled_list.append(torch.einsum("bn,bnc->bc", w, head.value(f_n)))
            gain = head.radial_gain(e)                         
            x_new_list.append(X_a[:, a, :] + torch.einsum("bn,bn,bni->bi", w, gain, rel))
        pooled = torch.stack(pooled_list, dim=1)   
        X_new = torch.stack(x_new_list, dim=1)     
        return F.silu(pooled), X_new
