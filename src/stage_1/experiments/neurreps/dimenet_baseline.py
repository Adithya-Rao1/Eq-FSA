from __future__ import annotations

import torch
import torch.nn as nn

from torch_geometric.nn.models import DimeNetPlusPlus


class DimeNetPlusPlusClassifier(nn.Module):
    def __init__(
        self,
        num_node_types: int,
        classes: int,
        num_layers: int,
        cutoff: float,
        hidden_channels: int = 32,
        int_emb_size: int = 16,
        basis_emb_size: int = 8,
        out_emb_channels: int = 32,
        num_spherical: int = 5,
        num_radial: int = 6,
        max_num_neighbors: int = 64,
    ):
        super().__init__()
        if num_node_types > 95:
            raise ValueError(
                f"num_node_types={num_node_types} exceeds DimeNet's fixed Embedding(95, ...) "
                f"atom-type table"
            )
        self.dimenet = DimeNetPlusPlus(
            hidden_channels=hidden_channels, out_channels=classes, num_blocks=num_layers,
            int_emb_size=int_emb_size, basis_emb_size=basis_emb_size,
            out_emb_channels=out_emb_channels, num_spherical=num_spherical, num_radial=num_radial,
            cutoff=cutoff, max_num_neighbors=max_num_neighbors,
        )

    @property
    def input_embedding_module(self) -> nn.Embedding:
        return self.dimenet.emb.emb

    def forward(
        self,
        x: torch.Tensor,
        types: torch.Tensor,
        mask: torch.Tensor,
        target_idx: torch.Tensor,
        edge_feats: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mask = mask.bool()
        B, N = types.shape
        device = types.device
        batch_vec = torch.arange(B, device=device).unsqueeze(1).expand(B, N)[mask]
        z = types[mask]
        pos = x[mask]
        return self.dimenet(z, pos, batch_vec)
