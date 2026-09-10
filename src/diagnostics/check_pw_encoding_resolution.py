from __future__ import annotations

import torch

from stage_1.experiments.neurreps.oversquashing.synthetic_dataset import (
    GeometricGraphTransferDataset,
)
from utils.pw_bases import make_wavevectors

SEEDS = [0, 7919, 15838, 23757, 31676, 39595, 47514]
R_VALUES = [2, 6, 8]
CUTOFF = 5.0
N_MAX = 1 

KNOWN_VAL_ERROR = {
    (2, 0): 0.0, (2, 7919): 0.0, (2, 15838): 0.0, (2, 23757): 0.0,
    (2, 31676): 0.85, (2, 39595): 0.0, (2, 47514): 0.0,
    (6, 0): 0.9167, (6, 7919): 0.8167, (6, 15838): 0.85, (6, 23757): 0.0,
    (6, 31676): 0.0, (6, 39595): 0.2167, (6, 47514): 0.0,
    (8, 0): 0.0, (8, 7919): 0.8333, (8, 15838): 0.0, (8, 23757): 0.0,
    (8, 31676): 0.0, (8, 39595): 0.0, (8, 47514): 0.0,
}


def check_one(topology: str, r: int, seed: int) -> dict | None:
    rng_seed = seed
    try:
        ds = GeometricGraphTransferDataset(
            topology, r, classes=5, samples=5, seed=rng_seed, cutoff=CUTOFF,
        )
    except RuntimeError as e:
        print(f"  r={r} seed={seed}: geometry construction failed: {e}")
        return None

    V = ds.V
    axis = ds.axis 
    spacing = float(ds.target_dists[1] - ds.target_dists[0])  
    full_spread = float(ds.target_dists[-1] - ds.target_dists[0])

    kvecs, _, mode_mask = make_wavevectors(V, N_MAX, e_cutoff=None) 
    k_norms = kvecs.norm(dim=-1)
    max_k = float(k_norms.max())
    nyquist_length = 2 * 3.141592653589793 / max_k if max_k > 0 else float("inf")

    step_vec = axis * spacing
    phase_per_mode_per_step = kvecs @ step_vec 
    max_phase_per_step = float(phase_per_mode_per_step.abs().max())

    full_vec = axis * full_spread
    phase_per_mode_full = kvecs @ full_vec
    max_phase_full = float(phase_per_mode_full.abs().max())

    box_diag = torch.diagonal(V).abs()

    return {
        "r_nominal": r, "seed": seed, "realized_r": ds.realized_r,
        "realized_euclid": ds.realized_euclid, "n_atoms": ds.n_atoms,
        "box_diag": box_diag.tolist(), "max_k": max_k,
        "nyquist_length": nyquist_length, "class_spacing": spacing,
        "max_phase_per_step_rad": max_phase_per_step,
        "max_phase_full_range_rad": max_phase_full,
        "val_error": KNOWN_VAL_ERROR.get((r, seed)),
    }


def main() -> None:
    rows = []
    for r in R_VALUES:
        print(f"\n=== r={r} ===")
        for seed in SEEDS:
            row = check_one("chain", r, seed)
            if row is None:
                continue
            rows.append(row)
            print(
                f"  seed={seed:>6} realized_r={row['realized_r']} euclid={row['realized_euclid']:.2f} "
                f"n_atoms={row['n_atoms']:>3} box_diag={['%.2f' % b for b in row['box_diag']]} "
                f"max|k|={row['max_k']:.4f} nyquist_len={row['nyquist_length']:.2f} "
                f"class_spacing={row['class_spacing']:.4f} "
                f"max_phase/step={row['max_phase_per_step_rad']:.5f} rad "
                f"max_phase/full_range={row['max_phase_full_range_rad']:.5f} rad "
                f"val_error={row['val_error']}"
            )

    print("\n=== summary: does per-step phase separation predict pass/fail? ===")
    solved = [r for r in rows if r["val_error"] == 0.0]
    failed = [r for r in rows if r["val_error"] not in (0.0, None)]
    if solved:
        avg_solved = sum(r["max_phase_per_step_rad"] for r in solved) / len(solved)
        print(f"  solved (n={len(solved)}): mean max_phase/step = {avg_solved:.6f} rad")
    if failed:
        avg_failed = sum(r["max_phase_per_step_rad"] for r in failed) / len(failed)
        print(f"  failed (n={len(failed)}): mean max_phase/step = {avg_failed:.6f} rad")
    print(
        "  (small-angle regime is roughly <0.3-0.5 rad; if both groups sit deep in that "
        "regime with no separation between them, k-space resolution is not what's "
        "distinguishing pass from fail at these r values -- the bottleneck is elsewhere.)"
    )

    print("\n=== does box size grow monotonically with r (as hypothesized)? ===")
    for r in R_VALUES:
        r_rows = [row for row in rows if row["r_nominal"] == r]
        if not r_rows:
            continue
        avg_max_k = sum(row["max_k"] for row in r_rows) / len(r_rows)
        avg_euclid = sum(row["realized_euclid"] for row in r_rows) / len(r_rows)
        print(f"  r={r}: mean euclid={avg_euclid:.2f}, mean max|k|={avg_max_k:.4f}")


if __name__ == "__main__":
    main()
