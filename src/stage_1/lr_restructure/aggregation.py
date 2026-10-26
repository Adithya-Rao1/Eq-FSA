from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.pw_bases import minimum_image
from utils.se3_utils import softmax_over_neighbors, init_linear_tensor_

from stage_1.lr_restructure.rbf import LogSpacedRBF


class NodeHeads(nn.Module):
    def __init__(self, feature_dim: int, n_master_nodes: int, hidden_dim: int, num_rbf: int):
        super().__init__()
        self.A_Q_weight = nn.Parameter(torch.empty(n_master_nodes, hidden_dim, feature_dim))
        self.A_Q_bias = nn.Parameter(torch.zeros(n_master_nodes, hidden_dim))
        self.A_K_weight = nn.Parameter(torch.empty(n_master_nodes, hidden_dim, feature_dim))
        self.A_K_bias = nn.Parameter(torch.zeros(n_master_nodes, hidden_dim))
        self.A_E_weight = nn.Parameter(torch.empty(n_master_nodes, hidden_dim, num_rbf))
        self.A_E_bias = nn.Parameter(torch.zeros(n_master_nodes, hidden_dim))
        self.A_V_weight = nn.Parameter(torch.empty(n_master_nodes, feature_dim, feature_dim))
        self.A_V_bias = nn.Parameter(torch.zeros(n_master_nodes, feature_dim))
        self.u_weight = nn.Parameter(torch.empty(n_master_nodes, 1, hidden_dim))
        self.h_gain_weight = nn.Parameter(torch.empty(n_master_nodes, 1, num_rbf))
        for w in (self.A_Q_weight, self.A_K_weight, self.A_E_weight, self.A_V_weight):
            init_linear_tensor_(w.view(-1, w.shape[-1]), "leaky_relu")
        for w in (self.u_weight, self.h_gain_weight):
            init_linear_tensor_(w.view(-1, w.shape[-1]), "linear")

    def forward(
        self, g_a: torch.Tensor, f_n: torch.Tensor, e: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = torch.einsum("ahc,bac->bah", self.A_Q_weight, g_a) + self.A_Q_bias
        k = torch.einsum("ahc,bnc->banh", self.A_K_weight, f_n) + self.A_K_bias.unsqueeze(1)
        v = torch.einsum("ahc,bnc->banh", self.A_V_weight, f_n) + self.A_V_bias.unsqueeze(1)
        ee = torch.einsum("ahk,bank->banh", self.A_E_weight, e) + self.A_E_bias.unsqueeze(1)
        gain = torch.einsum("ahk,bank->banh", self.h_gain_weight, e).squeeze(-1)

        h = F.leaky_relu(q.unsqueeze(2) + k + ee)
        s = torch.einsum("aoh,banh->bano", self.u_weight, h).squeeze(-1)
        return s, v, gain


class MasterNodeAggregator(nn.Module):
    def __init__(
        self, feature_dim: int, n_master_nodes: int, hidden_dim: int = 64,
        rbf_num_basis: int = 12, rbf_min: float = 0.3, rbf_cutoff: float = 25.0,
    ):
        super().__init__()
        self.n_master_nodes = n_master_nodes
        self.rbf = LogSpacedRBF(rbf_num_basis, rbf_min, rbf_cutoff)
        self.heads = NodeHeads(feature_dim, n_master_nodes, hidden_dim, rbf_num_basis)

    def forward(
        self, g_a: torch.Tensor, f_n: torch.Tensor, x: torch.Tensor, X_a: torch.Tensor,
        node_mask: torch.Tensor, V: torch.Tensor, V_inv: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N_a, _ = X_a.shape
        N = x.shape[1]
        rel_raw = x.unsqueeze(1) - X_a.unsqueeze(2)
        rel = minimum_image(rel_raw.reshape(B, N_a * N, 3), V, V_inv).reshape(B, N_a, N, 3)
        d = rel.norm(dim=-1)                                            
        e = self.rbf(d)                                              

        s, v, gain = self.heads(g_a, f_n, e)                            
        mask3 = node_mask.unsqueeze(1).expand(-1, N_a, -1).contiguous()
        w = softmax_over_neighbors(s, mask=mask3)                      

        pooled = torch.einsum("ban,banc->bac", w, v)                   
        X_new = X_a + torch.einsum("ban,ban,bani->bai", w, gain, rel)  
        return F.silu(pooled), X_new
