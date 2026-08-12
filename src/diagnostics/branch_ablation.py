from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys

import torch

from stage_1.attention import MasterNodeAttention
from diagnostics.checkpoint_utils import (
    evaluate_validation,
    load_model_and_val_loader,
)


def _zero_master_attn_forward(self, g_a, f_in0, x, x_com, node_mask, V):
    return torch.zeros_like(g_a)


@contextlib.contextmanager
def master_attn_zeroed():
    orig_forward = MasterNodeAttention.forward
    MasterNodeAttention.forward = _zero_master_attn_forward
    try:
        yield
    finally:
        MasterNodeAttention.forward = orig_forward


def _fmt_row(label: str, m: dict) -> str:
    return (
        f"{label:22s} normalized_loss={m['normalized_loss']:.6g}  energy_term={m['energy_term']:.6g}  "
        f"force_term={m['force_term']:.6g}  energy_MAE={m['energy_mae_eV']:.6g} eV  "
        f"force_MAE={m['force_mae_eV_per_A']:.6g} eV/A  (n_molecules={m['n_molecules']}, n_atoms={m['n_atoms']})"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=str, required=True, help="Path to a train_spice.py checkpoint (e.g. checkpoint_epoch_0007.pt)")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--hdf5-paths", type=str, nargs="+", default=None,
                    help="Override the checkpoint's stored hdf5 paths (needed if the dataset moved since training)")
    p.add_argument("--batch-size", type=int, default=None, help="Override the checkpoint's stored batch size")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--json-out", type=str, default=None, help="Session report generator JSON for phase 0, 1, 2, 3 experiments")
    args = p.parse_args()

    device = torch.device(args.device) if args.device else (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))

    model, val_loader, config, energy_mean, energy_std, force_rms, ckpt = load_model_and_val_loader(
        args.ckpt, device,
        hdf5_paths_override=args.hdf5_paths,
        batch_size_override=args.batch_size,
        num_workers_override=args.num_workers,
    )
    lambda1, lambda2 = config.get("lambda1", 1.0), config.get("lambda2", 1.0)
    print(
        f"Loaded {args.ckpt}: epoch={ckpt.get('epoch')} global_step={ckpt.get('global_step')} "
        f"val_loss(at save time)={ckpt.get('val_loss')}"
    )
    print(f"Re-derived validation split: {len(val_loader.dataset)} molecules, {len(val_loader)} batches\n")

    print("Running BASELINE (master-node branch intact) over the full validation set...")
    baseline = evaluate_validation(model, val_loader, device, energy_mean, energy_std, force_rms, lambda1, lambda2)
    print(_fmt_row("baseline", baseline))

    print("\nRunning ABLATED (MasterNodeAttention.forward -> zeros) over the full validation set...")
    with master_attn_zeroed():
        ablated = evaluate_validation(model, val_loader, device, energy_mean, energy_std, force_rms, lambda1, lambda2)
    print(_fmt_row("ablated (zeroed)", ablated))

    print("\n=== Comparison ===")
    for key in ("normalized_loss", "energy_term", "force_term", "energy_mae_eV", "force_mae_eV_per_A"):
        base_v, abl_v = baseline[key], ablated[key]
        rel_change = (abl_v - base_v) / base_v if base_v != 0 else float("nan")
        print(f"  {key:20s} baseline={base_v:.6g}  ablated={abl_v:.6g}  rel_change={rel_change:+.2%}")

    rel_energy_change = (ablated["energy_mae_eV"] - baseline["energy_mae_eV"]) / baseline["energy_mae_eV"] if baseline["energy_mae_eV"] else float("nan")
    rel_force_change = (ablated["force_mae_eV_per_A"] - baseline["force_mae_eV_per_A"]) / baseline["force_mae_eV_per_A"] if baseline["force_mae_eV_per_A"] else float("nan")
    print("\nInterpretation guide (not an automated verdict -- read the numbers above):")
    print("  - If both MAEs are ~unchanged (say, within a few percent): the collapsing master/fusion")
    print("    grad-norm buckets late in training reflect the branch correctly learning to contribute")
    print("    ~nothing for gas-phase molecules -- expected, not a bug.")
    print("  - If either MAE jumps materially: the branch is load-bearing and the collapsing grad norms")
    print("    are worth investigating further (e.g. an LR/warmup issue specific to that branch, or a")
    print("    downstream saturation silently zeroing its gradient before its contribution vanishes).")
    print(f"  Observed: energy MAE rel_change={rel_energy_change:+.2%}, force MAE rel_change={rel_force_change:+.2%}")

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump({
                "ckpt": args.ckpt, "epoch": ckpt.get("epoch"), "global_step": ckpt.get("global_step"),
                "baseline": baseline, "ablated": ablated,
                "energy_mae_rel_change": rel_energy_change, "force_mae_rel_change": rel_force_change,
            }, f, indent=2, default=str)
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    sys.exit(main())