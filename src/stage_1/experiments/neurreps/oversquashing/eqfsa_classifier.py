from __future__ import annotations

import torch
import torch.nn as nn

from stage_1.model import EnergyModel
from utils.se3_utils import EquivariantLinear, EquivariantNormReadout, EquivariantRMSNorm


class EqFSAClassifier(nn.Module):
    def __init__(
        self,
        radial_net,
        num_atom_types: int,
        max_degree: int,
        num_layers: int,
        feature_dim: int,
        hidden_dim: int,
        edge_cutoff: float,
        n_max: int,
        classes: int,
        pw_e_cutoff: float | None = None,
        n_master_nodes: int = 4,
        num_fusion_heads: int = 4,
        edge_feat_dim: int = 1,
        ablate_max_degree: int | None = None,
        grad_checkpoint: bool = False,
        multi_layer_readout: bool = True,
    ):
        super().__init__()
        self.multi_layer_readout = multi_layer_readout
        self.energy_model = EnergyModel(
            radial_net=radial_net,
            num_atom_types=num_atom_types,
            max_degree=max_degree,
            num_layers=num_layers,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            edge_cutoff=edge_cutoff,
            n_max=n_max,
            pw_e_cutoff=pw_e_cutoff,
            n_master_nodes=n_master_nodes,
            num_fusion_heads=num_fusion_heads,
            edge_feat_dim=edge_feat_dim,
            ablate_max_degree=ablate_max_degree,
            grad_checkpoint=grad_checkpoint,
        )
        self.norm_readout1 = EquivariantNormReadout(C=feature_dim, hidden=hidden_dim)
        self.norm_readout2 = EquivariantNormReadout(C=feature_dim, hidden=hidden_dim)
        self.classifier_head = nn.Sequential(
            EquivariantRMSNorm(feature_dim, degree=0),
            EquivariantLinear(feature_dim, hidden_dim, nonlinearity="leaky_relu"),
            EquivariantRMSNorm(hidden_dim, degree=0),
            nn.LeakyReLU(),
            EquivariantLinear(hidden_dim, hidden_dim, nonlinearity="leaky_relu"),
            EquivariantRMSNorm(hidden_dim, degree=0),
            nn.LeakyReLU(),
            EquivariantLinear(hidden_dim, hidden_dim, nonlinearity="leaky_relu"),
            EquivariantRMSNorm(hidden_dim, degree=0),
            nn.LeakyReLU(),
            EquivariantLinear(hidden_dim, classes),
        )

    @property
    def input_embedding_module(self) -> nn.Embedding:
        return self.energy_model.input_embedding

    def _readout_logits(self, f: dict, batch_ar: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
        f_target = {l: f[l][batch_ar, target_idx] for l in f}
        u_vec = self.norm_readout1(f_target[1])
        u_ten = self.norm_readout2(f_target[2])
        full_vec = f_target[0].squeeze(-1) + u_vec.unsqueeze(-1) + u_ten.unsqueeze(-1)
        logits = self.classifier_head(full_vec.unsqueeze(-1))
        return logits.squeeze(-1)

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
        seed_masses: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = x.shape[0]
        batch_ar = torch.arange(B, device=x.device)

        if self.multi_layer_readout:
            f, layer_states = self.energy_model(
                x, types, masses, mask, V, mol_props, edge_feats, return_layer_states=True,
                seed_masses=seed_masses,
            )
            logits = self._readout_logits(f, batch_ar, target_idx)
            for layer_f in layer_states:
                logits = logits + self._readout_logits(layer_f, batch_ar, target_idx)
            return logits

        f = self.energy_model(x, types, masses, mask, V, mol_props, edge_feats, seed_masses=seed_masses)
        return self._readout_logits(f, batch_ar, target_idx)
