from __future__ import annotations

import argparse
import json
import os
import sys

import torch

from diagnostics.phase2_experiments import PHASE2_ITEMS
from diagnostics.phase2_runner import (
    estimate_batches_per_epoch,
    run_diagnostics_script,
    run_train_spice,
)

DATALOADER_SWEEP_MODULE = "diagnostics.spice_dataloader_sweep"


def run_phase2_battery(
    base_config: dict,
    results_root: str,
    dry_run: bool = False,
    device: str | None = None,
) -> dict:
    os.makedirs(results_root, exist_ok=True)
    summary: dict = {"items": {}}

    try:
        batches_per_epoch = estimate_batches_per_epoch(base_config)
    except Exception as e:
        print(f"[phase2] WARNING: couldn't compute batches_per_epoch ({e!r}); "
              f"epoch_cap variants will run their full nominal max_steps uncapped instead.")
        batches_per_epoch = None
    summary["batches_per_epoch"] = batches_per_epoch

    items_by_id = {item_id: (builder, desc) for item_id, builder, desc in PHASE2_ITEMS}

    def _run_item(item_id: str, builder, description: str):
        variants = builder(base_config)
        item_dir = os.path.join(results_root, item_id)
        item_result: dict = {"description": description, "variants": {}}

        for variant in variants:
            max_steps = variant["max_steps"]
            epoch_multiplier = variant.get("epoch_multiplier")
            if epoch_multiplier and batches_per_epoch:
                max_steps = epoch_multiplier * batches_per_epoch
            elif variant.get("epoch_cap") and batches_per_epoch:
                max_steps = min(max_steps, batches_per_epoch)

            overrides = dict(variant["cli_overrides"])
            overrides["max_steps"] = max_steps
            if device:
                overrides["device"] = device

            variant_dir = os.path.join(item_dir, variant["id"])
            label = f"{item_id}/{variant['id']}"

            if dry_run:
                print(f"[DRY RUN] {label}: max_steps={max_steps}  batch_size={overrides.get('batch_size')}  lr={overrides.get('lr')}")
                item_result["variants"][variant["id"]] = {"status": "DRY_RUN", "max_steps": max_steps, "cli_overrides": overrides}
                continue

            meta = run_train_spice(overrides, variant_dir, label, timeout_s=None)
            item_result["variants"][variant["id"]] = meta

        summary["items"][item_id] = item_result

    def _run_dataloader_sweep():
        item_id = "2.2_dataloader_tuning"
        description = "DataLoader tuning (num_workers x pin_memory x persistent_workers), pure throughput, no model compute"
        item_dir = os.path.join(results_root, item_id)

        if dry_run:
            print(f"[DRY RUN] {item_id}")
            summary["items"][item_id] = {"description": description, "status": "DRY_RUN"}
            return

        os.makedirs(item_dir, exist_ok=True)
        csv_path = os.path.join(item_dir, "dataloader_sweep.csv")
        args = [
            "--hdf5-paths", *base_config["hdf5_paths"],
            "--subsets", *(base_config.get("subsets") or ["pubchem"]),
            "--box-padding", str(base_config["box_padding"]),
            "--edge-cutoff", str(base_config["edge_cutoff"]),
            "--batch-size", str(base_config["batch_size"]),
            "--csv-path", csv_path,
        ]
        meta = run_diagnostics_script(DATALOADER_SWEEP_MODULE, args, item_dir, item_id, timeout_s=None)
        meta["description"] = description
        summary["items"][item_id] = meta

    _run_item("2.1_diagnostics_overhead", *items_by_id["2.1_diagnostics_overhead"])
    _run_dataloader_sweep()
    for item_id in ("2.3_batch_size_lr", "2.5_lambda_rebalance", "2.6_warmup_schedule", "2.7_progressive_instability_watch", "2.8_equivariant_degree_ablation"):
        _run_item(item_id, *items_by_id[item_id])

    summary_path = os.path.join(results_root, "phase2_battery_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n[phase2] Battery complete. Summary written to {summary_path}")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-ckpt", type=str, required=True, help="Checkpoint whose config seeds every Phase 2 variant (e.g. checkpoint_epoch_0007.pt)")
    p.add_argument("--results-root", type=str, required=True)
    p.add_argument("--hdf5-paths", type=str, nargs="+", default=None, help="Override the checkpoint's stored hdf5 paths")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--dry-run", action="store_true", help="Print the planned schedule (variants, max_steps) without running anything")
    args = p.parse_args()

    ckpt = torch.load(args.base_ckpt, map_location="cpu", weights_only=False)
    base_config = dict(ckpt["config"])
    if args.hdf5_paths:
        base_config["hdf5_paths"] = args.hdf5_paths

    run_phase2_battery(base_config, args.results_root, dry_run=args.dry_run, device=args.device)


if __name__ == "__main__":
    sys.exit(main())