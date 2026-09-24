import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MasterNodeUpdate(nn.Module):
    def __init__(self, sigma: Optional[float] = None, alpha_floor: float = 1e-4):
        super().__init__()
        self.alpha_floor = alpha_floor
        self.fixed_sigma = sigma
        if sigma is not None:
            self.register_buffer("alpha", torch.tensor(1.0 / (2.0 * sigma ** 2)))
        else:
            self.alpha_param = nn.Parameter(torch.tensor(1.0))

    def forward(
        self, pooled: torch.Tensor, X_a: torch.Tensor, V: torch.Tensor,
        kvecs: torch.Tensor, mode_mask: torch.Tensor,
    ) -> torch.Tensor:
        alpha = self.alpha if self.fixed_sigma is not None else (
            F.softplus(self.alpha_param) + self.alpha_floor
        )
        k2 = (kvecs ** 2).sum(dim=-1)                       
        mode_mask_f = mode_mask.to(pooled.dtype)
        has_k = (k2 > 1e-12).to(k2.dtype)
        D = has_k / k2.clamp(min=1e-12) * torch.exp(-k2 / (4.0 * alpha)) * mode_mask_f  

        pooled_centered = pooled - pooled.mean(dim=1, keepdim=True)

        phase_b = -torch.einsum("bai,bmi->bam", X_a, kvecs) 
        phase_a = torch.einsum("bai,bmi->bam", X_a, kvecs) 
        cos_b, sin_b = torch.cos(phase_b), torch.sin(phase_b)
        cos_a, sin_a = torch.cos(phase_a), torch.sin(phase_a)

        S_re = torch.einsum("bac,bam->bmc", pooled_centered, cos_b) 
        S_im = torch.einsum("bac,bam->bmc", pooled_centered, sin_b)

        weighted_re = S_re * D.unsqueeze(-1)
        weighted_im = S_im * D.unsqueeze(-1)
        g_new = (
            torch.einsum("bmc,bam->bac", weighted_re, cos_a)
            - torch.einsum("bmc,bam->bac", weighted_im, sin_a)
        )                                                    

        omega = torch.linalg.det(V).abs().view(-1, 1, 1)    
        return g_new * (4.0 * math.pi) / omega
