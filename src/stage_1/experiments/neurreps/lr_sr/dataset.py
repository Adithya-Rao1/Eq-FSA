from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from utils.calc_props import _MOL_FEAT_DIM
from stage_1.experiments.neurreps.lr_sr.physics import compute_energy_and_forces

N_ATOMS = 128
N_POSITIVE = 64
N_NEGATIVE = 64
MIN_SEPARATION = 2.5
BOX_MIN, BOX_MAX = 15.0, 25.0
MAX_RELAXATION_ITERS = 3000
RELAXATION_LR = 0.05
RELAXATION_LOSS_TOL = 1e-10
MAX_CONFIG_RESTARTS = 50

POSITIVE_TYPE_IDX = 0
NEGATIVE_TYPE_IDX = 1
NUM_ATOM_TYPES = 2
LR_SR_EDGE_FEAT_DIM = 1


def _try_generate_positions(box: float, generator: torch.Generator) -> Optional[torch.Tensor]:
    positions = (torch.rand(N_ATOMS, 3, generator=generator, dtype=torch.double) * box)
    positions.requires_grad_(True)
    optimizer = torch.optim.Adam([positions], lr=RELAXATION_LR)
    self_mask = torch.eye(N_ATOMS, dtype=torch.double) * 1e3 

    for _ in range(MAX_RELAXATION_ITERS):
        optimizer.zero_grad()
        d = positions.unsqueeze(0) - positions.unsqueeze(1)
        d = d - box * torch.round(d.detach() / box)  
        r = d.norm(dim=-1) + self_mask
        violation = (MIN_SEPARATION - r).clamp(min=0.0)
        loss = (violation ** 2).sum() / 2  
        if loss.item() < RELAXATION_LOSS_TOL:
            return positions.detach()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            positions.data = positions.data % box

    return None


def _generate_positions(generator: torch.Generator) -> tuple[torch.Tensor, float]:
    for _ in range(MAX_CONFIG_RESTARTS):
        box = BOX_MIN + (BOX_MAX - BOX_MIN) * torch.rand(1, generator=generator).item()
        positions = _try_generate_positions(box, generator)
        if positions is not None:
            return positions, box
    raise RuntimeError(
        f"Failed to relax a valid configuration after {MAX_CONFIG_RESTARTS} restarts -- "
        f"MIN_SEPARATION may be too large for this box/atom-count density"
    )


def generate_configuration(seed: int) -> Dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    positions, box = _generate_positions(generator)

    charges = torch.cat([
        torch.ones(N_POSITIVE, dtype=torch.double), -torch.ones(N_NEGATIVE, dtype=torch.double),
    ])
    perm = torch.randperm(N_ATOMS, generator=generator)
    charges = charges[perm]
    types = torch.where(charges > 0, POSITIVE_TYPE_IDX, NEGATIVE_TYPE_IDX).long()

    V = torch.eye(3, dtype=torch.double) * box
    pos_b, charges_b, V_b = positions.unsqueeze(0), charges.unsqueeze(0), V.unsqueeze(0)
    energy, forces = compute_energy_and_forces(pos_b, charges_b, V_b)

    mol_props = torch.randn(_MOL_FEAT_DIM, generator=generator, dtype=torch.double)

    return {
        "x": positions.float(),
        "types": types,
        "V": V.float(),
        "energy": energy.squeeze(0).float(),
        "forces": forces.squeeze(0).float(),
        "mol_props": mol_props.float(),
        "box": box,
    }


class LRSRDataset(Dataset):
    def __init__(self, num_configs: int, seed: int = 0):
        self.configs = [generate_configuration(seed=seed + i) for i in range(num_configs)]

    def __len__(self) -> int:
        return len(self.configs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.configs[idx]


def lr_sr_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    B, N = len(batch), N_ATOMS
    return {
        "x": torch.stack([b["x"] for b in batch], dim=0),
        "types": torch.stack([b["types"] for b in batch], dim=0),
        "V": torch.stack([b["V"] for b in batch], dim=0),
        "energy": torch.stack([b["energy"] for b in batch], dim=0),
        "forces": torch.stack([b["forces"] for b in batch], dim=0),
        "mol_props": torch.stack([b["mol_props"] for b in batch], dim=0),
        "edge_feats": torch.zeros(B, N, N, LR_SR_EDGE_FEAT_DIM),
        "mask": torch.ones(B, N, dtype=torch.bool),
        "masses": torch.ones(B, N),
    }
