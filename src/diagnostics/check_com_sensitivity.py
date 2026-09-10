from __future__ import annotations

import torch

from stage_1.experiments.neurreps.oversquashing.synthetic_dataset import GeometricGraphTransferDataset
from stage_1.model import masked_weighted_com

SEEDS = [0, 7919, 15838, 23757, 31676, 39595, 47514]
R_VALUES = [2, 6, 8]
CUTOFF = 5.0
CLASSES = 5


def build(topology: str, r: int, seed: int):
    for attempt in range(20):
        trial_seed = seed + attempt * 1000003
        try:
            return GeometricGraphTransferDataset(
                topology, r, classes=CLASSES, samples=5, seed=trial_seed, cutoff=CUTOFF,
            )
        except RuntimeError:
            continue
    return None


def main() -> None:
    for r in R_VALUES:
        print(f"\n=== r={r} ===")
        periodic_sens, linear_sens = [], []
        for seed in SEEDS:
            ds = build("chain", r, seed)
            if ds is None:
                continue
            V = ds.V.unsqueeze(0)
            masses = torch.ones(1, ds.n_atoms)
            node_mask = torch.ones(1, ds.n_atoms, dtype=torch.bool)

            pos_c0 = ds.positions.clone()
            pos_c0[ds.label_idx] = ds.positions[ds.readout_idx] + ds.target_dists[0] * ds.axis
            pos_c1 = ds.positions.clone()
            pos_c1[ds.label_idx] = ds.positions[ds.readout_idx] + ds.target_dists[1] * ds.axis

            com0_p, _ = masked_weighted_com(pos_c0.unsqueeze(0), masses, node_mask, V=V)
            com1_p, _ = masked_weighted_com(pos_c1.unsqueeze(0), masses, node_mask, V=V)
            com0_l, _ = masked_weighted_com(pos_c0.unsqueeze(0), masses, node_mask, V=None)
            com1_l, _ = masked_weighted_com(pos_c1.unsqueeze(0), masses, node_mask, V=None)

            spacing = float(ds.target_dists[1] - ds.target_dists[0])
            sens_p = float((com1_p - com0_p).norm()) / spacing
            sens_l = float((com1_l - com0_l).norm()) / spacing
            periodic_sens.append(sens_p)
            linear_sens.append(sens_l)

            print(
                f"  seed={seed:>6} n_atoms={ds.n_atoms:>3} spacing={spacing:.4f} "
                f"|d(com)/d(spacing)| periodic={sens_p:.5f} linear={sens_l:.5f} "
                f"ratio(periodic/linear)={sens_p / max(sens_l, 1e-12):.4f}"
            )

        if periodic_sens:
            avg_p = sum(periodic_sens) / len(periodic_sens)
            avg_l = sum(linear_sens) / len(linear_sens)
            print(f"  mean sensitivity: periodic={avg_p:.5f} linear={avg_l:.5f} ratio={avg_p / max(avg_l, 1e-12):.4f}")


if __name__ == "__main__":
    main()
