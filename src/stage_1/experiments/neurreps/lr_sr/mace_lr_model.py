from __future__ import annotations

import torch
import torch.nn as nn

from stage_1.attention import MasterNodeAttention
from stage_1.experiments.neurreps.lr_sr.rsa_attention import RSAAttention

MACE_OFF_MODEL_SIZE = "small"
MACE_OFF_DTYPE = "float32"
CHARGE_TO_ELEMENT = {1.0: 1, -1.0: 8}  # +1e -> H (Z=1), -1e -> O (Z=8); see module docstring


def load_frozen_mace_off(model_size: str = MACE_OFF_MODEL_SIZE, device: str = "cpu", default_dtype: str = MACE_OFF_DTYPE):
    from mace.calculators.foundations_models import mace_off

    mace_model = mace_off(model=model_size, device=device, default_dtype=default_dtype, return_raw_model=True)
    mace_model.to(device=device, dtype=getattr(torch, default_dtype))
    for p in mace_model.parameters():
        p.requires_grad_(False)
    return mace_model


def unfreeze_readouts(mace_model) -> None:
    for p in mace_model.readouts.parameters():
        p.requires_grad_(True)


def build_z_table(mace_model):
    from mace.tools.utils import AtomicNumberTable

    return AtomicNumberTable(mace_model.atomic_numbers.tolist())


def configuration_to_atomic_data(positions: torch.Tensor, types: torch.Tensor, box: float, z_table, cutoff: float):
    import numpy as np
    from mace.data import AtomicData, Configuration

    from stage_1.experiments.neurreps.lr_sr.dataset import POSITIVE_TYPE_IDX

    atomic_numbers = np.array(
        [CHARGE_TO_ELEMENT[1.0] if int(t.item()) == POSITIVE_TYPE_IDX else CHARGE_TO_ELEMENT[-1.0] for t in types],
        dtype=int,
    )
    config = Configuration(
        atomic_numbers=atomic_numbers,
        positions=positions.detach().cpu().numpy().astype(np.float64),
        properties={},
        property_weights={},
        cell=np.eye(3) * box,
        pbc=(True, True, True),
    )
    return AtomicData.from_config(config, z_table=z_table, cutoff=cutoff)


def batch_configurations(positions_batch: torch.Tensor, types_batch: torch.Tensor, boxes, z_table, cutoff: float, device):
    from mace.tools.torch_geometric.batch import Batch

    data_list = [
        configuration_to_atomic_data(positions_batch[b], types_batch[b], float(boxes[b]), z_table, cutoff)
        for b in range(positions_batch.shape[0])
    ]
    batch = Batch.from_data_list(data_list)
    return batch.to(device)


class ChargeReadoutEmbedding(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, feature_dim))

    def forward(self, node_energy: torch.Tensor, charges: torch.Tensor) -> torch.Tensor:
        x = torch.stack([node_energy, charges], dim=-1) 
        return self.net(x).unsqueeze(-1)  


class MaceLRModel(nn.Module):
    def __init__(
        self, mace_model, variant: str, feature_dim: int = 32, hidden_dim: int = 64,
        n_max: int = 2, num_master_nodes: int = 4, rsa_sigma: float = 5.0,
    ):
        super().__init__()
        assert variant in ("none", "eqfsa", "rsa")
        self.mace_model = mace_model
        self.variant = variant
        for p in self.mace_model.parameters():
            p.requires_grad_(False)
        if variant == "none":
            return

        unfreeze_readouts(self.mace_model)
        self.embed = ChargeReadoutEmbedding(feature_dim, hidden_dim)
        if variant == "eqfsa":
            self.g_a_init = nn.Parameter(torch.randn(num_master_nodes, feature_dim) * 0.1)
            self.lr_attn = MasterNodeAttention(feature_dim, n_max, hidden_dim=hidden_dim)
            self.decoder = nn.Sequential(
                nn.Linear(num_master_nodes * feature_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1),
            )
        else:
            self.lr_attn = RSAAttention(feature_dim, n_max, sigma=rsa_sigma, hidden_dim=hidden_dim)
            self.decoder = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))

    def _lr_correction(self, node_energy, charges, positions, mask, V):
        from stage_1.model import masked_weighted_com

        f_in0 = self.embed(node_energy, charges)
        if self.variant == "eqfsa":
            B = charges.shape[0]
            g_a = self.g_a_init.unsqueeze(0).expand(B, -1, -1).unsqueeze(-1).contiguous()
            masses = mask.float()
            x_com, _ = masked_weighted_com(positions, masses, mask)
            g_a_out = g_a.squeeze(-1) + self.lr_attn(g_a, f_in0, positions, x_com, mask, V).squeeze(-1)
            return self.decoder(g_a_out.reshape(B, -1)).squeeze(-1)
        per_atom = self.lr_attn(f_in0, positions, mask, V).squeeze(-1)
        pooled = (per_atom * mask.unsqueeze(-1)).sum(dim=1)
        return self.decoder(pooled).squeeze(-1)

    def forward(self, positions_batch: torch.Tensor, types_batch, charges_batch, boxes, z_table, cutoff: float, device):
        mace_batch = batch_configurations(positions_batch.detach(), types_batch, boxes, z_table, cutoff, device)
        N = positions_batch.shape[1]
        mace_batch.positions = positions_batch.reshape(-1, 3)

        data_dict = mace_batch.to_dict()  # VERIFY ON REMOTE: confirm Batch exposes .to_dict()
        out = self.mace_model(data_dict, training=self.training, compute_force=False)
        node_energy = out["node_energy"].reshape(positions_batch.shape[0], N)  # VERIFY ON REMOTE: confirm this key name
        node_energy = node_energy.to(positions_batch.dtype)

        node_e0 = self.mace_model.atomic_energies_fn(data_dict["node_attrs"]).squeeze(-1)
        node_e0 = node_e0.reshape(positions_batch.shape[0], N).to(positions_batch.dtype)
        node_energy = node_energy - node_e0

        total_sr_energy = node_energy.sum(dim=1)

        if self.variant == "none":
            return total_sr_energy

        mask = torch.ones(positions_batch.shape[0], N, dtype=torch.bool, device=device)
        V = torch.stack([torch.eye(3, device=device, dtype=positions_batch.dtype) * float(b) for b in boxes], dim=0)
        e_lr = self._lr_correction(node_energy, charges_batch, positions_batch, mask, V)
        return total_sr_energy + e_lr


def compute_energy_and_forces(model: MaceLRModel, positions_batch, types_batch, charges_batch, boxes, z_table, cutoff, device):
    x = positions_batch.clone().requires_grad_(True)
    U = model(x, types_batch, charges_batch, boxes, z_table, cutoff, device)
    (grad_x,) = torch.autograd.grad(U.sum(), x, create_graph=model.training)
    return U, -grad_x
