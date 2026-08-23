from __future__ import annotations

import torch
import torch.nn as nn


class TrivialPoolClassifier(nn.Module):
    def __init__(self, num_atom_types: int, classes: int, embed_dim: int, hidden_dim: int, pool: str):
        super().__init__()
        assert pool in ("mean", "attn")
        self.pool_type = pool
        self.embedding = nn.Embedding(num_atom_types, embed_dim)
        if pool == "attn":
            self.query = nn.Parameter(torch.randn(embed_dim) * 0.1)
        self.classifier_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, classes),
        )

    @property
    def input_embedding_module(self) -> nn.Embedding:
        return self.embedding

    def forward(
        self,
        x: torch.Tensor,
        types: torch.Tensor,
        masses: torch.Tensor,
        mask: torch.Tensor,
        V: torch.Tensor,
        mol_props: torch.Tensor,
        edge_feats: torch.Tensor,
        target_idx: torch.Tensor,
    ) -> torch.Tensor:
        B, N = types.shape
        emb = self.embedding(types)  

        pool_mask = mask.clone()
        pool_mask[torch.arange(B, device=types.device), target_idx] = False  

        if self.pool_type == "mean":
            pool_mask_f = pool_mask.float().unsqueeze(-1)
            denom = pool_mask_f.sum(dim=1).clamp(min=1.0)
            pooled = (emb * pool_mask_f).sum(dim=1) / denom
        else:
            scores = torch.einsum("bnd,d->bn", emb, self.query)
            scores = scores.masked_fill(~pool_mask, float("-inf"))
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)
            pooled = (emb * weights).sum(dim=1)

        return self.classifier_head(pooled)