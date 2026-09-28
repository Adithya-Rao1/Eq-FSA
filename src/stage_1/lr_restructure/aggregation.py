from typing import Optional, Tuple

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


class _BatchedPerNodeHeads(nn.Module):
    def __init__(self, feature_dim: int, n_master_nodes: int, hidden_dim: int, num_rbf: int):
        super().__init__()
        templates = [_PerNodeHead(feature_dim, hidden_dim, num_rbf) for _ in range(n_master_nodes)]
        self.A_Q_w = nn.Parameter(torch.stack([t.A_Q.weight.detach().clone() for t in templates]))
        self.A_Q_b = nn.Parameter(torch.stack([t.A_Q.bias.detach().clone() for t in templates]))
        self.A_K_w = nn.Parameter(torch.stack([t.A_K.weight.detach().clone() for t in templates]))
        self.A_K_b = nn.Parameter(torch.stack([t.A_K.bias.detach().clone() for t in templates]))
        self.A_E_w = nn.Parameter(torch.stack([t.A_E.weight.detach().clone() for t in templates]))
        self.A_E_b = nn.Parameter(torch.stack([t.A_E.bias.detach().clone() for t in templates]))
        self.u_w = nn.Parameter(torch.stack([t.u.weight.detach().clone() for t in templates]))
        self.A_V_w = nn.Parameter(torch.stack([t.A_V.weight.detach().clone() for t in templates]))
        self.A_V_b = nn.Parameter(torch.stack([t.A_V.bias.detach().clone() for t in templates]))
        self.h_gain_w = nn.Parameter(torch.stack([t.h_gain.weight.detach().clone() for t in templates]))

    def forward(
        self, g_a: torch.Tensor, f_n: torch.Tensor, e: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_dim = self.A_K_w.shape[1]
        q = torch.einsum("ahc,bac->bah", self.A_Q_w, g_a) + self.A_Q_b

        KV_w = torch.cat([self.A_K_w, self.A_V_w], dim=1)
        kv = torch.einsum("ahc,bnc->banh", KV_w, f_n)
        k = kv[..., :hidden_dim] + self.A_K_b.unsqueeze(1)
        v = kv[..., hidden_dim:] + self.A_V_b.unsqueeze(1)

        EG_w = torch.cat([self.A_E_w, self.h_gain_w], dim=1)
        eg = torch.einsum("ahk,bank->banh", EG_w, e)
        ee = eg[..., :hidden_dim] + self.A_E_b.unsqueeze(1)
        gain = eg[..., hidden_dim:].squeeze(-1)

        h = F.leaky_relu(q.unsqueeze(2) + k + ee)
        s = torch.einsum("aoh,banh->bano", self.u_w, h).squeeze(-1)
        return s, v, gain


class MasterNodeAggregator(nn.Module):
    def __init__(
        self, feature_dim: int, n_master_nodes: int, hidden_dim: int = 64,
        rbf_num_basis: int = 12, rbf_min: float = 0.3, rbf_cutoff: float = 25.0,
    ):
        super().__init__()
        self.n_master_nodes = n_master_nodes
        self.rbf = LogSpacedRBF(rbf_num_basis, rbf_min, rbf_cutoff)
        self.heads = _BatchedPerNodeHeads(feature_dim, n_master_nodes, hidden_dim, rbf_num_basis)

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
