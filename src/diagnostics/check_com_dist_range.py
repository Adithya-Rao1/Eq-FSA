from __future__ import annotations

import torch

from stage_1.experiments.neurreps.oversquashing.synthetic_dataset import GeometricGraphTransferDataset
from stage_1.model import minimum_image, masked_weighted_com

SEEDS = [0, 7919, 15838, 23757, 31676, 39595, 47514]
R_VALUES = [2, 6, 8]
CUTOFF = 5.0
GEOMETRIC_SEED_RETRIES = 20


def build(topology: str, r: int, seed: int):
    for attempt in range(GEOMETRIC_SEED_RETRIES):
        trial_seed = seed + attempt * 1000003
        try:
            return GeometricGraphTransferDataset(
                topology, r, classes=5, samples=5, seed=trial_seed, cutoff=CUTOFF,
            )
        except RuntimeError:
            continue
    return None


def main() -> None:
    all_max_dists = []
    for r in R_VALUES:
        print(f"\n=== r={r} ===")
        for seed in SEEDS:
            ds = build("chain", r, seed)
            if ds is None:
                print(f"  seed={seed}: no safe geometry found after {GEOMETRIC_SEED_RETRIES} retries")
                continue
            x = ds.positions.unsqueeze(0)
            V = ds.V.unsqueeze(0)
            masses = torch.ones(1, ds.n_atoms)
            node_mask = torch.ones(1, ds.n_atoms, dtype=torch.bool)
            x_wrapped = minimum_image(x, V)
            x_com, _ = masked_weighted_com(x_wrapped, masses, node_mask)
            x_rel = x_wrapped - x_com.unsqueeze(1)
            x_rel = minimum_image(x_rel, V)
            dist = x_rel.norm(dim=-1)
            max_dist = float(dist.max())
            mean_dist = float(dist.mean())
            all_max_dists.append(max_dist)
            print(
                f"  seed={seed:>6} n_atoms={ds.n_atoms:>3} realized_r={ds.realized_r} "
                f"realized_euclid={ds.realized_euclid:.2f} max_dist_to_com={max_dist:.3f} "
                f"mean_dist_to_com={mean_dist:.3f}"
            )

    if all_max_dists:
        print(f"\nOverall max distance-to-COM across all (r, seed): {max(all_max_dists):.3f}")
        print(f"Suggested RBF cutoff_radius (with margin): {max(all_max_dists) * 1.15:.2f}")


if __name__ == "__main__":
    main()
