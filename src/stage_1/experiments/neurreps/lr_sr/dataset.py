from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from utils.calc_props import mol_props_from_atom_counts
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


def _relax_batch(positions: torch.Tensor, boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    B = positions.shape[0]
    device = positions.device
    positions = positions.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([positions], lr=RELAXATION_LR)
    self_mask = torch.eye(N_ATOMS, dtype=torch.double, device=device).unsqueeze(0) * 1e3
    boxes_nnn = boxes.view(B, 1, 1, 1)
    boxes_n11 = boxes.view(B, 1, 1)

    converged = torch.zeros(B, dtype=torch.bool, device=device)
    converged_positions = torch.zeros_like(positions)

    for _ in range(MAX_RELAXATION_ITERS):
        optimizer.zero_grad()
        d = positions.unsqueeze(1) - positions.unsqueeze(2)             
        d = d - boxes_nnn * torch.round(d.detach() / boxes_nnn)         
        r = d.norm(dim=-1) + self_mask                                  
        violation = (MIN_SEPARATION - r).clamp(min=0.0)
        per_sample_loss = (violation ** 2).sum(dim=(1, 2)) / 2         

        newly_converged = (per_sample_loss.detach() < RELAXATION_LOSS_TOL) & (~converged)
        if newly_converged.any():
            converged_positions[newly_converged] = positions.detach()[newly_converged]
            converged = converged | newly_converged
        if converged.all():
            break

        loss = per_sample_loss.sum()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            positions.data = positions.data % boxes_n11

    return converged_positions, converged


def generate_configurations_batch(
    seed: int, batch_size: int, device: "torch.device | str | None" = None,
) -> List[Dict[str, torch.Tensor]]:
    generators = [torch.Generator().manual_seed(seed + i) for i in range(batch_size)]
    boxes = torch.empty(batch_size, dtype=torch.double)
    positions = torch.empty(batch_size, N_ATOMS, 3, dtype=torch.double)
    for i, gen in enumerate(generators):
        box = BOX_MIN + (BOX_MAX - BOX_MIN) * torch.rand(1, generator=gen).item()
        boxes[i] = box
        positions[i] = torch.rand(N_ATOMS, 3, generator=gen, dtype=torch.double) * box

    final_positions = torch.zeros_like(positions)
    final_boxes = boxes.clone()
    remaining = torch.arange(batch_size)

    for _attempt in range(MAX_CONFIG_RESTARTS):
        pos_in, box_in = positions[remaining], boxes[remaining]
        if device is not None:
            pos_in, box_in = pos_in.to(device), box_in.to(device)
        relaxed, converged = _relax_batch(pos_in, box_in)
        relaxed, converged = relaxed.cpu(), converged.cpu()
        done_local = remaining[converged]
        final_positions[done_local] = relaxed[converged]
        final_boxes[done_local] = boxes[remaining][converged]
        remaining = remaining[~converged]
        if remaining.numel() == 0:
            break
        for idx in remaining.tolist():
            gen = generators[idx]
            box = BOX_MIN + (BOX_MAX - BOX_MIN) * torch.rand(1, generator=gen).item()
            boxes[idx] = box
            positions[idx] = torch.rand(N_ATOMS, 3, generator=gen, dtype=torch.double) * box
    else:
        if remaining.numel() > 0:
            raise RuntimeError(
                f"Failed to relax {remaining.numel()} config(s) after {MAX_CONFIG_RESTARTS} "
                f"restarts -- MIN_SEPARATION may be too large for this box/atom-count density"
            )

    configs = []
    for i in range(batch_size):
        box = float(final_boxes[i])
        pos = final_positions[i]
        charges = torch.cat([
            torch.ones(N_POSITIVE, dtype=torch.double), -torch.ones(N_NEGATIVE, dtype=torch.double),
        ])
        perm = torch.randperm(N_ATOMS, generator=generators[i])
        charges = charges[perm]
        types = torch.where(charges > 0, POSITIVE_TYPE_IDX, NEGATIVE_TYPE_IDX).long()

        V = torch.eye(3, dtype=torch.double) * box
        pos_b, charges_b, V_b = pos.unsqueeze(0), charges.unsqueeze(0), V.unsqueeze(0)
        energy, forces = compute_energy_and_forces(pos_b, charges_b, V_b)

        mol_props = mol_props_from_atom_counts(N_POSITIVE, N_NEGATIVE).double()

        configs.append({
            "x": pos.float(),
            "types": types,
            "V": V.float(),
            "energy": energy.squeeze(0).float(),
            "forces": forces.squeeze(0).float(),
            "mol_props": mol_props.float(),
            "box": box,
        })
    return configs


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

    mol_props = mol_props_from_atom_counts(N_POSITIVE, N_NEGATIVE).double()

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
    def __init__(
        self, num_configs: int, seed: int = 0, relax_batch_size: int = 64,
        device: "torch.device | str | None" = None,
    ):
        self.configs: List[Dict[str, torch.Tensor]] = []
        start = 0
        while start < num_configs:
            chunk = min(relax_batch_size, num_configs - start)
            self.configs.extend(
                generate_configurations_batch(seed=seed + start, batch_size=chunk, device=device)
            )
            start += chunk

    def __len__(self) -> int:
        return len(self.configs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = dict(self.configs[idx])
        item["idx"] = idx
        return item


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
        "idx": torch.tensor([b["idx"] for b in batch], dtype=torch.long),
    }
