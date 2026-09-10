from __future__ import annotations

import torch

from stage_1.experiments.neurreps.oversquashing.synthetic_dataset import GeometricGraphTransferDataset
from stage_1.model import minimum_image, masked_weighted_com

SEEDS = [0, 7919, 15838, 23757, 31676, 39595, 47514]
R_VALUES = [2, 6, 8]
CUTOFF = 5.0


def build(topology: str, r: int, seed: int):
    for attempt in range(20):
        trial_seed = seed + attempt * 1000003
        try:
            return GeometricGraphTransferDataset(
                topology, r, classes=5, samples=5, seed=trial_seed, cutoff=CUTOFF,
            )
        except RuntimeError:
            continue
    return None


def main() -> None:
    for r in R_VALUES:
        print(f"\n=== r={r} ===")
        for seed in SEEDS:
            ds = build("chain", r, seed)
            if ds is None:
                continue
            x = ds.positions.unsqueeze(0)
            V = ds.V.unsqueeze(0)
            masses = torch.ones(1, ds.n_atoms)
            node_mask = torch.ones(1, ds.n_atoms, dtype=torch.bool)

            x_com, _ = masked_weighted_com(x, masses, node_mask, V=V)
            x_com_linear, _ = masked_weighted_com(x, masses, node_mask, V=None)
            com_diff = (x_com - x_com_linear).norm().item()
            x_rel_raw = x - x_com.unsqueeze(1)
            x_rel_wrapped = minimum_image(x_rel_raw, V)

            raw_dist = x_rel_raw.norm(dim=-1).squeeze(0)
            wrapped_dist = x_rel_wrapped.norm(dim=-1).squeeze(0)
            diff = (raw_dist - wrapped_dist).abs()
            max_diff = float(diff.max())
            max_raw = float(raw_dist.max())

            half_period = V.squeeze(0).norm(dim=-1) / 2 

            print(
                f"  seed={seed:>6} n_atoms={ds.n_atoms:>3} realized_r={ds.realized_r} "
                f"max_dist_to_com(raw)={max_raw:.3f} max|raw-wrapped|={max_diff:.4f} "
                f"periodic_vs_linear_com_diff={com_diff:.6f}"
            )


if __name__ == "__main__":
    main()
