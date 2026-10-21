from typing import Optional

import torch
import torch.nn as nn

from stage_1.model import masked_mean
from stage_1.experiments.neurreps.lr_sr.invariant_energy_model import _ChargeEmbedding, _Decoder
from utils.calc_props import mol_props_from_atom_counts

from stage_1.lr_restructure.model import LREnergyModel


class LRRestructureEncoder(nn.Module):
    def __init__(
        self, num_master_nodes: int, feature_dim: int, hidden_dim: int, n_max: int,
        num_outputs: int, num_layers: int = 4, edge_cutoff: float = 5.0,
        num_broadcast_heads: int = 4, rbf_num_basis: int = 12, rbf_min: float = 0.3,
        rbf_cutoff: float = 25.0, sigma: Optional[float] = None, alpha_floor: float = 1e-4,
        use_long_range: bool = True,
    ):
        super().__init__()
        from utils.se3_utils import RadialNetwork

        self.charge_embed = _ChargeEmbedding(feature_dim, hidden_dim)
        self.energy_model = LREnergyModel(
            radial_net=RadialNetwork, num_atom_types=1, max_degree=2, num_layers=num_layers,
            feature_dim=feature_dim, hidden_dim=hidden_dim, edge_cutoff=edge_cutoff, n_max=n_max,
            n_master_nodes=num_master_nodes, edge_feat_dim=1,
            num_broadcast_heads=num_broadcast_heads, rbf_num_basis=rbf_num_basis,
            rbf_min=rbf_min, rbf_cutoff=rbf_cutoff, sigma=sigma, alpha_floor=alpha_floor,
            use_long_range=use_long_range,
        )
        self.decoder = _Decoder(feature_dim, hidden_dim, num_outputs)

    def forward(self, charges, positions, mask, V, return_features: bool = False):
        B, N = charges.shape
        input_features = self.charge_embed(charges).squeeze(-1)
        node_types = torch.zeros(B, N, dtype=torch.long, device=charges.device)
        masses = mask.float()
        n_h = ((charges > 0) & mask).sum(dim=1)
        n_o = ((charges < 0) & mask).sum(dim=1)
        mol_props = mol_props_from_atom_counts(n_h, n_o).to(device=charges.device, dtype=positions.dtype)
        edge_feats = torch.zeros(B, N, N, 1, device=charges.device, dtype=positions.dtype)

        f = self.energy_model(
            positions, node_types, masses, mask, V, mol_props, edge_feats,
            input_features=input_features,
        )
        pooled = masked_mean(f[0].squeeze(-1), mask)
        pred = self.decoder(pooled)
        return (pred, pooled) if return_features else pred
