from __future__ import annotations

import argparse
import json
import os
import sys

import torch

from diagnostics.checkpoint_utils import load_model_and_val_loader
from utils.se3_utils import EquivariantRMSNorm


class SiteAccumulator:
    def __init__(self):
        self.would_trip_calls = 0
        self.total_calls = 0
        self.elementwise_hits = 0
        self.elementwise_total = 0
        self.atom_hits = 0
        self.atom_total = 0

    def update(self, x: torch.Tensor, node_mask: torch.Tensor, floor: float):
        near_degenerate = (x.detach() ** 2).sum(dim=-1) < floor ** 2  # [B, N, C]
        atom_mask = node_mask.bool()

        self.total_calls += 1
        self.would_trip_calls += int(near_degenerate.any().item())

        masked = near_degenerate[atom_mask]  # [n_real_atoms_in_batch, C]
        self.elementwise_hits += int(masked.sum().item())
        self.elementwise_total += masked.numel()

        any_channel_per_atom = near_degenerate.any(dim=-1)  # [B, N]
        self.atom_hits += int(any_channel_per_atom[atom_mask].sum().item())
        self.atom_total += int(atom_mask.sum().item())

    def summary(self) -> dict:
        return {
            "n_calls": self.total_calls,
            "would_trip_fraction_of_calls": self.would_trip_calls / max(self.total_calls, 1),
            "elementwise_fraction_of_real_atom_channels": self.elementwise_hits / max(self.elementwise_total, 1),
            "fraction_of_real_atoms_with_ge1_near_degenerate_channel": self.atom_hits / max(self.atom_total, 1),
            "n_real_atom_channel_pairs_seen": self.elementwise_total,
        }


def attach_degeneracy_hooks(model, floor: float) -> tuple[dict[str, SiteAccumulator], list, dict]:
    accumulators: dict[str, SiteAccumulator] = {}
    handles = []
    current_mask_box: dict = {"mask": None}

    def make_hook(name):
        acc = accumulators.setdefault(name, SiteAccumulator())

        def hook(module, inputs, output):
            x = inputs[0]
            mask = current_mask_box["mask"]
            if mask is None or x.dim() < 2 or x.shape[:2] != mask.shape:
                return  # not an atom-indexed tensor (shouldn't happen for the sites we target below)
            acc.update(x, mask, floor)
        return hook

    for name, module in model.named_modules():
        if isinstance(module, EquivariantRMSNorm) and module.degree > 0:
            handles.append(module.register_forward_hook(make_hook(name)))

    return accumulators, handles, current_mask_box


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=str, required=True, help="Path to a train_spice.py checkpoint (e.g. checkpoint_epoch_0007.pt)")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--hdf5-paths", type=str, nargs="+", default=None,
                    help="Override the checkpoint's stored hdf5 paths (needed if the dataset moved since training)")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--num-batches", type=int, default=-1, help="Cap on validation batches to run (-1 = all)")
    p.add_argument("--floor", type=float, default=0.5, help="Must match EquivariantRMSNorm's own floor (se3_utils.py default: 0.5)")
    p.add_argument("--json-out", type=str, default=None, help="Session report generator JSON for phase 0, 1, 2, 3 experiments")
    args = p.parse_args()

    device = torch.device(args.device) if args.device else (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))

    model, val_loader, config, energy_mean, energy_std, force_rms, ckpt = load_model_and_val_loader(
        args.ckpt, device,
        hdf5_paths_override=args.hdf5_paths,
        batch_size_override=args.batch_size,
        num_workers_override=args.num_workers,
    )
    print(
        f"Loaded {args.ckpt}: epoch={ckpt.get('epoch')} global_step={ckpt.get('global_step')}\n"
        f"Re-derived validation split: {len(val_loader.dataset)} molecules, {len(val_loader)} batches"
    )

    accumulators, handles, current_mask_box = attach_degeneracy_hooks(model, args.floor)
    print(f"Hooked {len(handles)} degree>0 EquivariantRMSNorm instances.\n")

    model.eval()
    n_run = 0
    try:
        with torch.no_grad():
            for batch in val_loader:
                if args.num_batches >= 0 and n_run >= args.num_batches:
                    break
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                current_mask_box["mask"] = batch["all_solute_mask"].bool()
                model.energy_model(
                    batch["all_solute_x"], batch["all_solute_types"], batch["all_solute_masses"],
                    batch["all_solute_mask"], batch["V"], batch["mol_props"], batch["all_solute_edge_feats"],
                )
                n_run += 1
    finally:
        for h in handles:
            h.remove()

    print(f"Ran {n_run} real validation batch(es) through the model with hooks attached.\n")

    if not accumulators:
        print("No degree>0 EquivariantRMSNorm instances were hit -- check that the model has "
              "max_degree >= 1 and that the hooked forward actually ran.")
        return

    rows = sorted(accumulators.items(), key=lambda kv: -kv[1].summary()["elementwise_fraction_of_real_atom_channels"])

    print(f"{'site':45s} {'calls':>6s} {'would_trip%':>12s} {'real_elemwise%':>15s} {'real_atom%':>11s}")
    print("-" * 92)
    agg_would_trip, agg_elem_hits, agg_elem_total, agg_atom_hits, agg_atom_total = 0, 0, 0, 0, 0
    for name, acc in rows:
        s = acc.summary()
        print(
            f"{name:45s} {s['n_calls']:>6d} {s['would_trip_fraction_of_calls']:>11.1%} "
            f"{s['elementwise_fraction_of_real_atom_channels']:>14.4%} "
            f"{s['fraction_of_real_atoms_with_ge1_near_degenerate_channel']:>10.1%}"
        )
        agg_would_trip += acc.would_trip_calls
        agg_elem_hits += acc.elementwise_hits
        agg_elem_total += acc.elementwise_total
        agg_atom_hits += acc.atom_hits
        agg_atom_total += acc.atom_total

    total_calls = sum(acc.total_calls for _, acc in rows)
    print("-" * 92)
    print(
        f"{'OVERALL (all sites pooled)':45s} {total_calls:>6d} "
        f"{agg_would_trip / max(total_calls, 1):>11.1%} "
        f"{agg_elem_hits / max(agg_elem_total, 1):>14.4%} "
        f"{agg_atom_hits / max(agg_atom_total, 1):>10.1%}"
    )
    print(
        "\nCompare 'would_trip%' above against the real CSV's near-universal per-step firing rate for "
        "eqln_deg_gt0_near_degenerate. If 'real_elemwise%'/'real_atom%' (computed with the correct "
        "node_mask, real atoms only) are near 0 while 'would_trip%' stays high, that confirms the "
        "counter is structurally driven by padded slots, not real chemistry -- the same missing-mask "
        "pattern as the confirmed min_abs_radial bug. If the real-atom fractions are themselves "
        "non-trivial, this is a genuine training-dynamics signal worth acting on, not just padding."
    )

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump({
                "ckpt": args.ckpt, "epoch": ckpt.get("epoch"), "global_step": ckpt.get("global_step"),
                "n_batches_run": n_run, "floor": args.floor,
                "sites": {name: acc.summary() for name, acc in rows},
                "overall": {
                    "n_calls": total_calls,
                    "would_trip_fraction_of_calls": agg_would_trip / max(total_calls, 1),
                    "elementwise_fraction_of_real_atom_channels": agg_elem_hits / max(agg_elem_total, 1),
                    "fraction_of_real_atoms_with_ge1_near_degenerate_channel": agg_atom_hits / max(agg_atom_total, 1),
                },
            }, f, indent=2, default=str)
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    sys.exit(main())