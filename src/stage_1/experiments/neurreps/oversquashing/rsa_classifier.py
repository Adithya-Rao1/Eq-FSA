from __future__ import annotations

import torch
import torch.nn as nn

from stage_1.experiments.neurreps.lr_sr.rsa_attention import RSAAttention


class RSAClassifier(nn.Module):
    def __init__(
        self,
        num_atom_types: int,
        classes: int,
        feature_dim: int,
        hidden_dim: int,
        n_max: int,
        sigma: float,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_atom_types, feature_dim)
        self.rsa_attn = RSAAttention(feature_dim, n_max, sigma=sigma, hidden_dim=hidden_dim)
        self.classifier_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
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
        B = x.shape[0]
        batch_ar = torch.arange(B, device=x.device)

        f_in0 = self.embedding(types).unsqueeze(-1)
        per_atom = self.rsa_attn(f_in0, x, mask, V).squeeze(-1)
        readout = per_atom[batch_ar, target_idx]
        return self.classifier_head(readout)
