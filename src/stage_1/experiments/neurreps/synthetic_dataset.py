from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from utils.calc_props import _MOL_FEAT_DIM

from stage_1.experiments.neurreps.synthetic_geometry import (
    DEFAULT_BOND_LENGTH,
    DEFAULT_CONFINEMENT_RADIUS,
    DEFAULT_CUTOFF,
    build_topology_geometry,
)

GRAPH_BOX_HALFWIDTH = 1000.0

GRAPH_EDGE_FEAT_DIM = 1

FILLER_TYPE_IDX = 0


class GraphTransferDataset(Dataset):
    def __init__(
        self,
        topology: str,
        r: int,
        classes: int,
        samples: int,
        seed: int = 0,
        cutoff: float = DEFAULT_CUTOFF,
        bond_length: float = DEFAULT_BOND_LENGTH,
        confinement_radius: float = DEFAULT_CONFINEMENT_RADIUS,
    ):
        self.topology = topology
        self.classes = classes
        self.cutoff = cutoff
        self.num_atom_types = classes + 1

        rng = torch.Generator().manual_seed(seed)
        geo = build_topology_geometry(
            topology, r, cutoff=cutoff, bond_length=bond_length,
            confinement_radius=confinement_radius, rng=rng,
        )
        self.positions = geo["positions"]
        self.source_idx = geo["source_idx"]
        self.target_idx = geo["target_idx"]
        self.realized_r = geo["realized_r"]
        self.realized_euclid = geo["realized_euclid"]
        self.n_atoms = self.positions.shape[0]

        base_types = torch.full((self.n_atoms,), FILLER_TYPE_IDX, dtype=torch.long)

        self._labels: List[int] = [i % classes for i in range(samples)]
        rng_perm = torch.Generator().manual_seed(seed + 1)
        perm = torch.randperm(samples, generator=rng_perm).tolist()
        self._labels = [self._labels[i] for i in perm]
        self._base_types = base_types

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        label = self._labels[idx]
        types = self._base_types.clone()
        types[self.source_idx] = 1 + label
        return {
            "x": self.positions,
            "types": types,
            "mask": torch.ones(self.n_atoms, dtype=torch.bool),
            "target_idx": self.target_idx,
            "label": label,
            "r": self.realized_r,
        }


def _pad1(t: torch.Tensor, n_max: int) -> torch.Tensor:
    n = t.shape[0]
    if n == n_max:
        return t
    pad_shape = (n_max - n,) + tuple(t.shape[1:])
    return torch.cat([t, torch.zeros(pad_shape, dtype=t.dtype)], dim=0)


def graph_transfer_collate_fn(batch: List[Dict[str, object]]) -> Dict[str, object]:
    B = len(batch)
    n_list = [item["x"].shape[0] for item in batch]
    N_max = max(n_list)

    all_x = torch.stack([_pad1(item["x"], N_max) for item in batch], dim=0)
    all_types = torch.stack([_pad1(item["types"], N_max) for item in batch], dim=0)
    all_mask = torch.stack(
        [_pad1(item["mask"].float(), N_max) for item in batch], dim=0
    ).bool()
    all_masses = all_mask.float()

    V = torch.eye(3).unsqueeze(0).expand(B, 3, 3).contiguous() * GRAPH_BOX_HALFWIDTH
    mol_props = torch.zeros(B, _MOL_FEAT_DIM)
    edge_feats = torch.zeros(B, N_max, N_max, GRAPH_EDGE_FEAT_DIM)

    target_idx = torch.tensor([item["target_idx"] for item in batch], dtype=torch.long)
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    r = torch.tensor([item["r"] for item in batch], dtype=torch.long)

    return {
        "x": all_x,
        "types": all_types,
        "masses": all_masses,
        "mask": all_mask,
        "V": V,
        "mol_props": mol_props,
        "edge_feats": edge_feats,
        "target_idx": target_idx,
        "labels": labels,
        "r": r,
    }
