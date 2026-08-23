from __future__ import annotations

import torch
import torch.nn as nn

from stage_1.model import EnergyModel, masked_mean
from utils.se3_utils import EquivariantLinear, EquivariantNormReadout, EquivariantRMSNorm


class LRSREnergyForceModel(nn.Module):
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
        edge_feat_dim: int,
        ablate_lr: bool = False,
        n_master_nodes: int = 4,
        num_fusion_heads: int = 4,
    ):
        super().__init__()
        self.energy_model = EnergyModel(
            radial_net=radial_net,
            num_atom_types=num_atom_types,
            max_degree=max_degree,
            num_layers=num_layers,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            edge_cutoff=edge_cutoff,
            n_max=n_max,
            n_master_nodes=n_master_nodes,
            num_fusion_heads=num_fusion_heads,
            edge_feat_dim=edge_feat_dim,
            ablate_lr=ablate_lr,
        )
        self.norm_readout1 = EquivariantNormReadout(C=feature_dim, hidden=hidden_dim)
        self.norm_readout2 = EquivariantNormReadout(C=feature_dim, hidden=hidden_dim)
        self.out = nn.Sequential(
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
            EquivariantLinear(hidden_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        types_: torch.Tensor,
        masses: torch.Tensor,
        mask: torch.Tensor,
        V: torch.Tensor,
        mol_props: torch.Tensor,
        edge_feats: torch.Tensor,
    ) -> torch.Tensor:
        f = self.energy_model(x, types_, masses, mask, V, mol_props, edge_feats)
        f_pooled = {l: masked_mean(f[l], mask) for l in f}

        u_vec = self.norm_readout1(f_pooled[1])
        u_ten = self.norm_readout2(f_pooled[2])
        full_vec = f_pooled[0].squeeze(-1) + u_vec.unsqueeze(-1) + u_ten.unsqueeze(-1)
        out = self.out(full_vec.unsqueeze(-1))
        return out.squeeze(-1).squeeze(-1)


def compute_energy_and_forces(
    model: LRSREnergyForceModel,
    x: torch.Tensor,
    types_: torch.Tensor,
    masses: torch.Tensor,
    mask: torch.Tensor,
    V: torch.Tensor,
    mol_props: torch.Tensor,
    edge_feats: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = x.clone().requires_grad_(True)
    U = model(x, types_, masses, mask, V, mol_props, edge_feats)
    (grad_x,) = torch.autograd.grad(U.sum(), x, create_graph=model.training)
    F = -grad_x
    return U, F
