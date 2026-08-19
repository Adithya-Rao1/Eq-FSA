from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


def build_edges_from_cutoff(
    x: torch.Tensor, mask: torch.Tensor, cutoff: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, N, _ = x.shape
    dist = torch.cdist(x, x)
    self_mask = ~torch.eye(N, dtype=torch.bool, device=x.device).unsqueeze(0)
    pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1) & (dist <= cutoff) & self_mask
    return pair_mask.nonzero(as_tuple=True)


class EGNNLayer(nn.Module):
    def __init__(
        self, hidden_dim: int, cutoff: float, edge_feat_dim: int = 0, coord_update_scale: float = 1.0,
    ):
        super().__init__()
        self.coord_update_scale = coord_update_scale
        self.dist_sq_scale = 1.0 / (cutoff ** 2)
        self.norm = nn.LayerNorm(hidden_dim)
        self.phi_e = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1 + edge_feat_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.phi_x = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        self.phi_h = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        nn.init.normal_(self.phi_h[-1].weight, std=0.01)
        nn.init.zeros_(self.phi_h[-1].bias)

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        edge_b: torch.Tensor,
        edge_i: torch.Tensor,
        edge_j: torch.Tensor,
        edge_feat_e: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, hidden = h.shape
        h_normed = self.norm(h)

        h_i_e = h_normed[edge_b, edge_i]
        h_j_e = h_normed[edge_b, edge_j]
        rel_e = x[edge_b, edge_i] - x[edge_b, edge_j]
        dist_sq_e = (rel_e ** 2).sum(-1, keepdim=True) * self.dist_sq_scale

        m_inputs = [h_i_e, h_j_e, dist_sq_e]
        if edge_feat_e is not None:
            m_inputs.append(edge_feat_e)
        m_e = self.phi_e(torch.cat(m_inputs, dim=-1))

        coord_w_e = self.coord_update_scale * torch.tanh(self.phi_x(m_e))
        coord_update_e = rel_e * coord_w_e

        receiver_flat = edge_b * N + edge_i
        deg = torch.zeros(B * N, device=x.device, dtype=x.dtype).index_add(
            0, receiver_flat, torch.ones(receiver_flat.shape[0], device=x.device, dtype=x.dtype),
        ).clamp(min=1.0).view(B, N, 1)

        x_agg = torch.zeros(B * N, 3, device=x.device, dtype=x.dtype).index_add(
            0, receiver_flat, coord_update_e,
        ).view(B, N, 3)
        x_out = x + x_agg / deg

        m_agg = torch.zeros(B * N, hidden, device=x.device, dtype=x.dtype).index_add(
            0, receiver_flat, m_e,
        ).view(B, N, hidden) / deg
        h_out = h + self.phi_h(torch.cat([h_normed, m_agg], dim=-1))

        return h_out, x_out


class EGNNModel(nn.Module):
    def __init__(
        self,
        num_node_types: int,
        hidden_dim: int,
        num_layers: int,
        cutoff: float,
        classes: int,
        edge_feat_dim: int = 0,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.embedding = nn.Embedding(num_node_types, hidden_dim)
        self.layers = nn.ModuleList([
            EGNNLayer(hidden_dim, cutoff, edge_feat_dim=edge_feat_dim) for _ in range(num_layers)
        ])
        self.classifier_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, classes),
        )

    @property
    def input_embedding_module(self) -> nn.Embedding:
        return self.embedding

    def forward(
        self,
        x: torch.Tensor,
        types: torch.Tensor,
        mask: torch.Tensor,
        target_idx: torch.Tensor,
        edge_feats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.embedding(types)
        edge_b, edge_i, edge_j = build_edges_from_cutoff(x, mask, self.cutoff)
        edge_feat_e = edge_feats[edge_b, edge_i, edge_j] if edge_feats is not None else None

        for layer in self.layers:
            h, x = layer(h, x, edge_b, edge_i, edge_j, edge_feat_e=edge_feat_e)

        B = h.shape[0]
        h_target = h[torch.arange(B, device=h.device), target_idx]
        return self.classifier_head(h_target)
