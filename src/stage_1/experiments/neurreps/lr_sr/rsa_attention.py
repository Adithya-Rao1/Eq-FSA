from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.pw_bases import make_wavevectors
from utils.se3_utils import scalar_ffn

class RSAAttention(nn.Module):
    def __init__(self, feature_dim: int, n_max: int, sigma: float = 5.0, hidden_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_max = n_max
        self.sigma = sigma

        self.W_Q = scalar_ffn(feature_dim, hidden_dim)
        self.W_K = scalar_ffn(feature_dim, hidden_dim)
        self.W_V = nn.Sequential(scalar_ffn(feature_dim, hidden_dim), nn.RMSNorm(feature_dim))

    def forward(
        self,
        f_in0: torch.Tensor,    
        x: torch.Tensor,         
        node_mask: torch.Tensor,
        V: torch.Tensor,         
    ) -> torch.Tensor:
        B, N, C, _ = f_in0.shape
        m = node_mask.to(x.dtype)
        h = f_in0.squeeze(-1) 

        Q = self.W_Q(h)
        K = self.W_K(h)
        V_val = self.W_V(h)
        phi_Q = (F.elu(Q) + 1) * m.unsqueeze(-1)
        phi_K = (F.elu(K) + 1) * m.unsqueeze(-1)

        kvecs, indices, mode_mask = make_wavevectors(V, self.n_max, e_cutoff=None)
        mode_mask_f = mode_mask.to(x.dtype)
        if mode_mask_f.dim() == 1:
            mode_mask_f = mode_mask_f.unsqueeze(0).expand(B, -1)
        nonzero = (indices.abs().sum(dim=-1) > 0).unsqueeze(0).to(x.device)
        active = (nonzero.expand(B, -1) & mode_mask_f.bool()).to(x.dtype)

        k2 = (kvecs ** 2).sum(dim=-1)
        w_k = torch.exp(-k2 * (self.sigma ** 2) / 2.0) / k2.clamp(min=1e-12) * active  

        phase_source = -torch.einsum("bni,bmi->bnm", x, kvecs)  
        cos_n = torch.cos(phase_source) * mode_mask_f.unsqueeze(1)
        sin_n = torch.sin(phase_source) * mode_mask_f.unsqueeze(1)
        S_re = torch.einsum("bnc,bnd,bnm->bmcd", phi_K, V_val, cos_n)  
        S_im = torch.einsum("bnc,bnd,bnm->bmcd", phi_K, V_val, sin_n)

        phase_query = torch.einsum("bni,bmi->bnm", x, kvecs)  
        cos_m = torch.cos(phase_query) * w_k.unsqueeze(1)  
        sin_m = torch.sin(phase_query) * w_k.unsqueeze(1)

        QS_re = torch.einsum("bnc,bmcd->bnmd", phi_Q, S_re)
        QS_im = torch.einsum("bnc,bmcd->bnmd", phi_Q, S_im)

        out = (QS_re * cos_m.unsqueeze(-1) - QS_im * sin_m.unsqueeze(-1)).sum(dim=2)  

        Z = torch.einsum("bnc,bc->bn", phi_Q, phi_K.sum(dim=1)).clamp(min=1e-8) 
        w_k_sum = w_k.sum(dim=-1).clamp(min=1e-8)
        out = out / Z.unsqueeze(-1) / w_k_sum.view(B, 1, 1)

        return out.unsqueeze(-1)
