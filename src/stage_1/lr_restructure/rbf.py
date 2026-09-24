import math

import torch
import torch.nn as nn


class LogSpacedRBF(nn.Module):
    def __init__(self, num_basis: int = 12, rbf_min: float = 0.3, rbf_cutoff: float = 25.0):
        super().__init__()
        self.num_basis = num_basis
        centers = torch.logspace(math.log10(rbf_min), math.log10(rbf_cutoff), num_basis)
        gaps = centers[1:] - centers[:-1]
        widths = torch.empty(num_basis)
        widths[0] = gaps[0]
        widths[-1] = gaps[-1]
        widths[1:-1] = (gaps[:-1] + gaps[1:]) / 2
        self.register_buffer("centers", centers)
        self.register_buffer("inv_width_sq", -1.0 / (2.0 * widths ** 2))

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        diff = dist.unsqueeze(-1) - self.centers.view(*([1] * dist.dim()), -1)
        return torch.exp(self.inv_width_sq * diff * diff)
