from typing import Optional

import torch
import torch.nn as nn

from utils.se3_utils import EquivariantLinear, EquivariantNormReadout, EquivariantRMSNorm

from stage_1.lr_restructure.model import LREnergyModel


class LRRestructureClassifier(nn.Module):
    def __init__(
        self, radial_net, num_atom_types: int, max_degree: int, num_layers: int,
        feature_dim: int, hidden_dim: int, edge_cutoff: float, n_max: int, classes: int,
        n_master_nodes: int = 4, num_broadcast_heads: int = 4, edge_feat_dim: int = 1,
        rbf_num_basis: int = 12, rbf_min: float = 0.3, rbf_cutoff: float = 25.0,
        sigma: Optional[float] = None, alpha_floor: float = 1e-4,
        multi_layer_readout: bool = True,
    ):
        super().__init__()
        self.multi_layer_readout = multi_layer_readout
        self.energy_model = LREnergyModel(
            radial_net=radial_net, num_atom_types=num_atom_types, max_degree=max_degree,
            num_layers=num_layers, feature_dim=feature_dim, hidden_dim=hidden_dim,
            edge_cutoff=edge_cutoff, n_max=n_max, n_master_nodes=n_master_nodes,
            edge_feat_dim=edge_feat_dim, num_broadcast_heads=num_broadcast_heads,
            rbf_num_basis=rbf_num_basis, rbf_min=rbf_min, rbf_cutoff=rbf_cutoff,
            sigma=sigma, alpha_floor=alpha_floor,
        )
        self.norm_readout1 = EquivariantNormReadout(C=feature_dim, hidden=hidden_dim, out_dim=feature_dim)
        self.norm_readout2 = EquivariantNormReadout(C=feature_dim, hidden=hidden_dim, out_dim=feature_dim)
        self.classifier_head = nn.Sequential(
            EquivariantRMSNorm(feature_dim, degree=0),
            EquivariantLinear(feature_dim, hidden_dim, nonlinearity="leaky_relu"),
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
        full_vec = f_target[0].squeeze(-1) + u_vec + u_ten
        logits = self.classifier_head(full_vec.unsqueeze(-1))
        return logits.squeeze(-1)

    def forward(
        self, x: torch.Tensor, types: torch.Tensor, masses: torch.Tensor, mask: torch.Tensor,
        V: torch.Tensor, mol_props: torch.Tensor, edge_feats: torch.Tensor,
        target_idx: torch.Tensor, seed_masses: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = x.shape[0]
        batch_ar = torch.arange(B, device=x.device)

        if self.multi_layer_readout:
            f, layer_states = self.energy_model(
                x, types, masses, mask, V, mol_props, edge_feats, seed_masses=seed_masses,
                return_layer_states=True,
            )
            logits = self._readout_logits(f, batch_ar, target_idx)
            for layer_f in layer_states:
                logits = logits + self._readout_logits(layer_f, batch_ar, target_idx)
            return logits

        f = self.energy_model(x, types, masses, mask, V, mol_props, edge_feats, seed_masses=seed_masses)
        return self._readout_logits(f, batch_ar, target_idx)
