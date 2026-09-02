from __future__ import annotations

import math

import torch
import torch.nn as nn

from utils.pw_bases import make_wavevectors

class RSAAttention(nn.Module):
    def __init__(self, feature_dim: int, n_max: int, sigma: float = 5.0, hidden_dim: int = 128):
        super().__init__()
        assert feature_dim % 2 == 0, "RSA's rotary encoding pairs channels, so feature_dim must be even"
        self.feature_dim = feature_dim
        self.n_max = n_max
        self.sigma = sigma

        self.qkv = nn.Linear(feature_dim, 3 * feature_dim, bias=False)
        self.act = nn.SiLU()
        self.scale_q = 1.0 / math.sqrt(feature_dim)
        self.out_norm = nn.RMSNorm(feature_dim)

    def _rope(self, t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        a, b = t[..., 0::2].unsqueeze(2), t[..., 1::2].unsqueeze(2)
        cos, sin = cos.unsqueeze(-1), sin.unsqueeze(-1)
        rot_a = a * cos - b * sin
        rot_b = a * sin + b * cos
        return torch.cat([rot_a, rot_b], dim=-1)

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

        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = self.act(q) * m.unsqueeze(-1)
        k = self.act(k) * m.unsqueeze(-1)

        kvecs, indices, mode_mask = make_wavevectors(V, self.n_max, e_cutoff=None)
        mode_mask_f = mode_mask.to(x.dtype)
        if mode_mask_f.dim() == 1:
            mode_mask_f = mode_mask_f.unsqueeze(0).expand(B, -1)
        nonzero = (indices.abs().sum(dim=-1) > 0).unsqueeze(0).to(x.device)
        active = (nonzero.expand(B, -1) & mode_mask_f.bool()).to(x.dtype)

        k2 = (kvecs ** 2).sum(dim=-1)
        w_k = torch.exp(-k2 * (self.sigma ** 2) / 2.0) / k2.clamp(min=1e-12) * active  # (B, M)
        w_k = w_k / w_k.sum(dim=-1, keepdim=True).clamp(min=1e-12)

        phase = torch.einsum("bni,bmi->bnm", x, kvecs)  
        cos, sin = phase.cos(), phase.sin()

        q_rot = self._rope(q, cos, sin) * self.scale_q  
        k_rot = self._rope(k, cos, sin)                 

        kv = torch.einsum("bnmc,bnd,bn->bmcd", k_rot, v, m)
        beta = torch.einsum("bnmc,bmcd->bnmd", q_rot, kv)

        update = torch.einsum("bnmd,bm->bnd", beta, w_k) * m.unsqueeze(-1)
        update = self.out_norm(update) * m.unsqueeze(-1)
        return update.unsqueeze(-1)