from __future__ import annotations

import torch
import torch.nn as nn

from oversquashing.dimenet_baseline import radius_graph

from nequip.data import AtomicDataDict
from nequip.model import NequIPGNNModel
from nequip.utils.global_state import set_global_state


class NequIPAllegroClassifier(nn.Module):
    def __init__(
        self,
        num_node_types: int,
        classes: int,
        num_layers: int,
        cutoff: float,
        num_features: int = 16,
        l_max: int = 1,
        parity: bool = True,
        model_dtype: str = "float32",
        max_num_neighbors: int = 64,
        avg_num_neighbors: float = 1.0,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.type_names = [f"T{i}" for i in range(num_node_types)]

        prev_default_dtype = torch.get_default_dtype()
        set_global_state()
        try:
            self.backbone = NequIPGNNModel(
                seed=0,
                model_dtype=model_dtype,
                type_names=self.type_names,
                r_max=cutoff,
                num_layers=num_layers,
                l_max=l_max,
                parity=parity,
                num_features=num_features,
                radial_mlp_depth=1,
                radial_mlp_width=num_features,
                avg_num_neighbors=avg_num_neighbors,
                per_type_energy_scales=1.0,
                per_type_energy_shifts=0.0,
                do_derivatives=False,
            )
        finally:
            torch.set_default_dtype(prev_default_dtype)
        self.classifier_head = nn.Linear(num_features, classes)

    @property
    def input_embedding_module(self) -> nn.Module:
        return self.backbone.model.func.type_embed.embed_module

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
        pos = x[mask]
        atom_type = types[mask]

        row, col = radius_graph(pos, r=self.cutoff, batch=batch_vec, max_num_neighbors=self.max_num_neighbors)
        edge_index = torch.stack([row, col], dim=0)
        edge_vectors = pos[col] - pos[row]

        data = {
            AtomicDataDict.POSITIONS_KEY: pos,
            AtomicDataDict.EDGE_INDEX_KEY: edge_index,
            AtomicDataDict.EDGE_VECTORS_KEY: edge_vectors,
            AtomicDataDict.ATOM_TYPE_KEY: atom_type,
            AtomicDataDict.BATCH_KEY: batch_vec,
            AtomicDataDict.NUM_NODES_KEY: mask.long().sum(dim=1),
        }

        out = self.backbone(data)
        node_features = out[AtomicDataDict.NODE_FEATURES_KEY]

        real_per_graph = mask.long().sum(dim=1)
        graph_start = torch.cat(
            [torch.zeros(1, dtype=torch.long, device=device), real_per_graph.cumsum(0)[:-1]]
        )
        local_rank = mask.long().cumsum(dim=1) - 1
        flat_target = graph_start + local_rank[torch.arange(B, device=device), target_idx]

        return self.classifier_head(node_features[flat_target])
