from __future__ import annotations

from typing import Dict, List

import torch
from torch.utils.data import Dataset

from utils.calc_props import _MOL_FEAT_DIM, _MOL_FEAT_NAMES

from stage_1.experiments.neurreps.oversquashing.synthetic_geometry import (
    DEFAULT_CUTOFF,
    FILLER_TYPE_IDX,
    LABEL_TYPE_BASE_IDX,
    MIN_BOND_LENGTH,
    READOUT_TYPE_IDX,
    build_topology_geometry,
    realized_graph_and_r,
)

GEOMETRIC_LABEL_TYPE_IDX = 2 

GRAPH_BOX_HALFWIDTH = 1000.0
GRAPH_EDGE_FEAT_DIM = 1

_COUNT_LIKE_MOL_FEAT_NAMES = {
    "HBA", "HBD", "HeavyAtomCount", "NHOHCount", "NOCount",
    "NumAliphaticCarbocycles", "NumAliphaticHeterocycles", "NumAliphaticRings",
    "NumAromaticCarbocycles", "NumAromaticHeterocycles", "NumAromaticRings",
    "NumHeteroatoms", "NumRadicalElectrons", "NumRotatableBonds",
    "NumSaturatedRings", "NumValenceElectrons",
}
_COUNT_LIKE_MOL_FEAT_MASK = torch.tensor(
    [name in _COUNT_LIKE_MOL_FEAT_NAMES for name in _MOL_FEAT_NAMES]
)


def _random_mol_props(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    continuous = torch.randn(_MOL_FEAT_DIM, generator=g)
    counts = torch.randint(0, 6, (_MOL_FEAT_DIM,), generator=g).float()
    return torch.where(_COUNT_LIKE_MOL_FEAT_MASK, counts, continuous)


class GraphTransferDataset(Dataset):
    def __init__(self, topology: str, r: int, classes: int, samples: int, seed: int = 0, cutoff: float = DEFAULT_CUTOFF):
        self.topology = topology
        self.classes = classes
        self.cutoff = cutoff
        self.num_atom_types = classes + 2  # filler + readout-marker + classes

        rng = torch.Generator().manual_seed(seed)
        geo = build_topology_geometry(topology, r, cutoff=cutoff, rng=rng)
        self.positions = geo["positions"]
        self.readout_idx = geo["readout_idx"]
        self.label_idx = geo["label_idx"]
        self.realized_r = geo["realized_r"]
        self.realized_euclid = geo["realized_euclid"]
        self.n_atoms = self.positions.shape[0]

        base_types = torch.full((self.n_atoms,), FILLER_TYPE_IDX, dtype=torch.long)
        base_types[self.readout_idx] = READOUT_TYPE_IDX
        self._base_types = base_types

        self.mol_props = _random_mol_props(seed)

        samples_per_class = max(1, samples // classes)
        self._labels: List[int] = [min(i // samples_per_class, classes - 1) for i in range(samples)]

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        label = self._labels[idx]
        types = self._base_types.clone()
        types[self.label_idx] = LABEL_TYPE_BASE_IDX + label
        return {
            "x": self.positions,
            "types": types,
            "mask": torch.ones(self.n_atoms, dtype=torch.bool),
            "target_idx": self.readout_idx,
            "label": label,
            "r": self.realized_r,
            "mol_props": self.mol_props,
        }


class GeometricGraphTransferDataset(Dataset):
    def __init__(
        self, topology: str, r: int, classes: int, samples: int, seed: int = 0,
        cutoff: float = DEFAULT_CUTOFF, class_spacing: float | None = None,
    ):
        self.topology = topology
        self.classes = classes
        self.cutoff = cutoff
        self.num_atom_types = 3  # filler + readout-marker + one constant label-marker

        rng = torch.Generator().manual_seed(seed)
        geo = build_topology_geometry(topology, r, cutoff=cutoff, rng=rng)
        self.positions = geo["positions"]
        self.readout_idx = geo["readout_idx"]
        self.label_idx = geo["label_idx"]
        self.realized_r = geo["realized_r"]
        self.realized_euclid = geo["realized_euclid"]
        self.n_atoms = self.positions.shape[0]

        base_types = torch.full((self.n_atoms,), FILLER_TYPE_IDX, dtype=torch.long)
        base_types[self.readout_idx] = READOUT_TYPE_IDX
        base_types[self.label_idx] = GEOMETRIC_LABEL_TYPE_IDX
        self._base_types = base_types

        self.mol_props = _random_mol_props(seed)

        axis = self.positions[self.label_idx] - self.positions[self.readout_idx]
        axis_norm = float(axis.norm())
        self.axis = axis / axis_norm

        dists_from_label = (self.positions - self.positions[self.label_idx]).norm(dim=-1)
        other_mask = torch.arange(self.n_atoms) != self.label_idx
        margins = (dists_from_label[other_mask] - cutoff).abs()
        max_spread = min(2.0 * MIN_BOND_LENGTH, 0.5 * float(margins.min()))
        min_safe_spacing = 0.05
        if max_spread <= 0:
            raise RuntimeError(
                f"GeometricGraphTransferDataset: no safe perturbation margin for topology={topology!r} "
                f"r={r} seed={seed} (closest atom-to-label margin against cutoff={cutoff} is "
                f"{float(margins.min()):.4f}); use a different r/seed for this task."
            )
        spacing = min(class_spacing, max_spread) if class_spacing is not None else max_spread / max(1, classes - 1)
        if spacing < min_safe_spacing:
            raise RuntimeError(
                f"GeometricGraphTransferDataset: safe spacing {spacing:.4f} is below the minimum "
                f"{min_safe_spacing} for topology={topology!r} r={r} seed={seed} -- task would be "
                f"unlearnably tight; use a different r/seed or fewer classes."
            )
        offsets = (torch.arange(classes, dtype=torch.float32) - (classes - 1) / 2.0) * spacing
        self.target_dists = axis_norm + offsets
        self.seed_masses_template = torch.ones(self.n_atoms)
        self.seed_masses_template[self.label_idx] = 0.0

        for c in range(classes):
            perturbed = self.positions.clone()
            perturbed[self.label_idx] = self.positions[self.readout_idx] + self.target_dists[c] * self.axis
            _, r_check, _ = realized_graph_and_r(perturbed, cutoff, self.readout_idx, self.label_idx)
            if r_check != self.realized_r:
                raise RuntimeError(
                    f"GeometricGraphTransferDataset: perturbation for class {c} changed realized_r "
                    f"({self.realized_r} -> {r_check}) for topology={topology!r} r={r}, seed={seed}."
                )

        samples_per_class = max(1, samples // classes)
        self._labels: List[int] = [min(i // samples_per_class, classes - 1) for i in range(samples)]

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        label = self._labels[idx]
        positions = self.positions.clone()
        positions[self.label_idx] = self.positions[self.readout_idx] + self.target_dists[label] * self.axis
        return {
            "x": positions,
            "types": self._base_types.clone(),
            "mask": torch.ones(self.n_atoms, dtype=torch.bool),
            "seed_masses": self.seed_masses_template.clone(),
            "target_idx": self.readout_idx,
            "label": label,
            "r": self.realized_r,
            "mol_props": self.mol_props,
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
    seed_masses = None
    if "seed_masses" in batch[0]:
        seed_masses = torch.stack([_pad1(item["seed_masses"], N_max) for item in batch], dim=0) * all_mask.float()

    V = torch.eye(3).unsqueeze(0).expand(B, 3, 3).contiguous() * GRAPH_BOX_HALFWIDTH
    mol_props = torch.stack([item["mol_props"] for item in batch], dim=0)
    edge_feats = torch.zeros(B, N_max, N_max, GRAPH_EDGE_FEAT_DIM)

    target_idx = torch.tensor([item["target_idx"] for item in batch], dtype=torch.long)
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    r = torch.tensor([item["r"] for item in batch], dtype=torch.long)

    return {
        "x": all_x,
        "types": all_types,
        "masses": all_masses,
        "seed_masses": seed_masses,
        "mask": all_mask,
        "V": V,
        "mol_props": mol_props,
        "edge_feats": edge_feats,
        "target_idx": target_idx,
        "labels": labels,
        "r": r,
    }
