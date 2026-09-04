from __future__ import annotations

import torch
import torch.nn as nn

from torch_geometric.nn.models import DimeNetPlusPlus

def _radius_graph(
    pos: torch.Tensor, r: float, batch: torch.Tensor, max_num_neighbors: int = 32,
) -> torch.Tensor:
    same_graph = batch.unsqueeze(0) == batch.unsqueeze(1)
    dist = torch.cdist(pos, pos)
    N = pos.shape[0]
    mask = same_graph & (dist <= r) & ~torch.eye(N, dtype=torch.bool, device=pos.device)
    if max_num_neighbors is not None:
        num_neighbors = mask.sum(dim=1)
        if bool((num_neighbors > max_num_neighbors).any()):
            dist_masked = dist.masked_fill(~mask, float("inf"))
            k = min(max_num_neighbors, N)
            topk_idx = dist_masked.topk(k=k, dim=1, largest=False).indices
            keep = torch.zeros_like(mask)
            keep.scatter_(1, topk_idx, True)
            mask = mask & keep
    row, col = mask.nonzero(as_tuple=True)
    return torch.stack([row, col], dim=0)

def _triplets(
    edge_index: torch.Tensor, num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    row, col = edge_index 
    E = row.shape[0]
    adjacent = col.unsqueeze(0) == row.unsqueeze(1)  
    idx_e, idx_ep = adjacent.nonzero(as_tuple=True)
    idx_i, idx_j, idx_k = col[idx_e], row[idx_e], row[idx_ep]
    keep = idx_i != idx_k  
    idx_i, idx_j, idx_k = idx_i[keep], idx_j[keep], idx_k[keep]
    idx_kj, idx_ji = idx_ep[keep], idx_e[keep]
    return col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji


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
        
        d = self.dimenet
        edge_index = _radius_graph(pos, r=d.cutoff, batch=batch_vec, max_num_neighbors=d.max_num_neighbors)
        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = _triplets(edge_index, num_nodes=z.size(0))

        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()
        pos_jk, pos_ij = pos[idx_j] - pos[idx_k], pos[idx_i] - pos[idx_j]
        a = (pos_ij * pos_jk).sum(dim=-1)
        b = torch.cross(pos_ij, pos_jk, dim=1).norm(dim=-1)
        angle = torch.atan2(b, a)

        rbf = d.rbf(dist)
        sbf = d.sbf(dist, angle, idx_kj)

        h = d.emb(z, rbf, i, j)
        P = d.output_blocks[0](h, rbf, i, num_nodes=pos.size(0))
        for interaction_block, output_block in zip(d.interaction_blocks, d.output_blocks[1:]):
            h = interaction_block(h, rbf, sbf, idx_kj, idx_ji)
            P = P + output_block(h, rbf, i, num_nodes=pos.size(0))

        real_per_graph = mask.long().sum(dim=1)
        graph_start = torch.cat(
            [torch.zeros(1, dtype=torch.long, device=device), real_per_graph.cumsum(0)[:-1]]
        )
        local_rank = mask.long().cumsum(dim=1) - 1
        flat_target = graph_start + local_rank[torch.arange(B, device=device), target_idx]

        return P[flat_target]
