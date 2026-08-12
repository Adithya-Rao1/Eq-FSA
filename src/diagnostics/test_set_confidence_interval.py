from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy import stats
from torch.utils.data import DataLoader

from diagnostics.checkpoint_utils import (
    build_model_from_config, build_test_dataset_from_split, load_checkpoint, load_test_split,
)
from stage_1.spice_dataset import spice_collate_fn
from stage_1.train_spice import compute_energy_and_forces


def compute_per_item_errors(model, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    energy_errors: list[float] = []
    energy_errors_per_atom: list[float] = []
    force_errors: list[float] = []

    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        U, F = compute_energy_and_forces(
            model, batch["all_solute_x"], batch["all_solute_types"], batch["all_solute_masses"],
            batch["all_solute_mask"], batch["V"], batch["mol_props"], batch["all_solute_edge_feats"],
        )
        mask = batch["all_solute_mask"].bool()

        e_err = (U.detach() - batch["energy"]).abs()
        energy_errors.extend(e_err.tolist())

        n_atoms_per_mol = mask.sum(dim=1).clamp(min=1).to(e_err.dtype)
        energy_errors_per_atom.extend((e_err / n_atoms_per_mol).tolist())

        f_err = (F.detach() - batch["solute_forces"]).abs()
        f_err_masked = f_err[mask.unsqueeze(-1).expand_as(f_err)]
        force_errors.extend(f_err_masked.tolist())

    return {
        "energy_errors": energy_errors,
        "energy_errors_per_atom": energy_errors_per_atom,
        "force_errors": force_errors,
    }


def bootstrap_mae_trials(errors: list[float], n_trials: int, sample_size: int | None, seed: int) -> list[float]:
    if not errors:
        return []
    arr = np.asarray(errors, dtype=np.float64)
    n = len(arr)
    size = sample_size or n
    rng = np.random.default_rng(seed)
    resample_idx = rng.integers(0, n, size=(n_trials, size))
    trial_means = arr[resample_idx].mean(axis=1)
    return trial_means.tolist()


def confidence_interval(trial_values: list[float], confidence: float = 0.95) -> dict:
    n = len(trial_values)
    if n == 0:
        return {"mean": None, "std": None, "ci_low": None, "ci_high": None, "margin": None, "n_trials": 0, "confidence": confidence}
    arr = np.asarray(trial_values, dtype=np.float64)
    mean = float(arr.mean())
    if n < 2:
        return {"mean": mean, "std": None, "ci_low": None, "ci_high": None, "margin": None, "n_trials": n, "confidence": confidence}
    std = float(arr.std(ddof=1))
    sem = float(stats.sem(arr))
    ci_low, ci_high = stats.t.interval(confidence, df=n - 1, loc=mean, scale=sem)
    return {
        "mean": mean, "std": std, "ci_low": float(ci_low), "ci_high": float(ci_high),
        "margin": float(mean - ci_low), "n_trials": n, "confidence": confidence,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=str, required=True, help="Checkpoint .pt file (e.g. <ckpt_dir>/checkpoint_epoch_0007.pt)")
    p.add_argument("--split-path", type=str, default=None, help="dataset_split.pt written by train_spice.py; defaults to <dirname(--ckpt)>/dataset_split.pt")
    p.add_argument("--hdf5-paths", type=str, nargs="+", default=None, help="Override the split's stored hdf5 paths (e.g. if the files moved to a different box since training)")
    p.add_argument("--batch-size", type=int, default=None, help="Defaults to the checkpoint's own config value")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--n-trials", type=int, default=5, help="Number of bootstrap resamples (trials)")
    p.add_argument("--bootstrap-sample-size", type=int, default=None, help="Items drawn (with replacement) per trial; defaults to the full test-set size")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--json-out", type=str, default=None)
    args = p.parse_args()

    device = torch.device(args.device) if args.device else (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))

    ckpt = load_checkpoint(args.ckpt, device)
    config = dict(ckpt["config"])

    split_path = args.split_path or os.path.join(os.path.dirname(args.ckpt) or ".", "dataset_split.pt")
    if not os.path.exists(split_path):
        raise FileNotFoundError(
            f"No dataset_split.pt at {split_path}. Checkpoints saved before the deterministic "
            f"train/val/test split existed (--test-fraction, added this session) don't have one -- "
            f"retrain (or resume) with a current train_spice.py, or pass --split-path explicitly "
            f"if one exists elsewhere."
        )
    split = load_test_split(split_path)
    print(f"Loaded test split from {split_path}: {len(split['test_identities'])} molecules "
          f"(test_fraction={split['test_fraction']}, val_fraction={split['val_fraction']})")

    test_dataset = build_test_dataset_from_split(split, hdf5_paths_override=args.hdf5_paths)

    batch_size = args.batch_size if args.batch_size is not None else config["batch_size"]
    loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=spice_collate_fn,
        num_workers=args.num_workers,
    )
    print(f"Test set: {len(test_dataset)} molecules, {len(loader)} batches")

    model = build_model_from_config(config, device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    print("Running inference over the full test set (one pass -- the model is deterministic, so "
          "trials are bootstrap resamples of the resulting per-item errors, not redundant passes)...")
    errors = compute_per_item_errors(model, loader, device)
    print(f"Collected {len(errors['energy_errors'])} per-molecule energy errors, "
          f"{len(errors['force_errors'])} per-component force errors")

    energy_trials = bootstrap_mae_trials(errors["energy_errors"], args.n_trials, args.bootstrap_sample_size, args.seed)
    energy_per_atom_trials = bootstrap_mae_trials(errors["energy_errors_per_atom"], args.n_trials, args.bootstrap_sample_size, args.seed + 2)
    force_trials = bootstrap_mae_trials(errors["force_errors"], args.n_trials, args.bootstrap_sample_size, args.seed + 1)

    energy_ci = confidence_interval(energy_trials)
    energy_per_atom_ci = confidence_interval(energy_per_atom_trials)
    force_ci = confidence_interval(force_trials)

    def _fmt(ci):
        if ci["ci_low"] is None:
            return f"{ci['mean']:.6g} (need >=2 trials for a CI)"
        return f"{ci['mean']:.6g}  95% CI [{ci['ci_low']:.6g}, {ci['ci_high']:.6g}]"

    print(f"\nEnergy MAE (per molecule): {_fmt(energy_ci)} eV  (n_trials={args.n_trials})")
    print(f"Energy MAE (per atom):     {_fmt(energy_per_atom_ci)} eV/atom  (n_trials={args.n_trials})")
    print(f"Force MAE:                 {_fmt(force_ci)} eV/A  (n_trials={args.n_trials})")

    os.makedirs(args.outdir, exist_ok=True)
    result = {
        "ckpt": args.ckpt, "split_path": split_path, "hdf5_paths_override": args.hdf5_paths,
        "n_test_molecules": len(test_dataset),
        "n_trials": args.n_trials, "bootstrap_sample_size": args.bootstrap_sample_size or len(test_dataset),
        "energy_mae_trials": energy_trials, "energy_mae_ci": energy_ci,
        "energy_mae_per_atom_trials": energy_per_atom_trials, "energy_mae_per_atom_ci": energy_per_atom_ci,
        "force_mae_trials": force_trials, "force_mae_ci": force_ci,
    }
    json_out = args.json_out or os.path.join(args.outdir, "test_set_confidence_interval.json")
    with open(json_out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nWrote {json_out}")


if __name__ == "__main__":
    sys.exit(main())
