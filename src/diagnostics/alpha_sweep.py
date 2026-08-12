from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys

import torch
import torch.nn.functional as nnf

from diagnostics.checkpoint_utils import (
    evaluate_validation,
    load_model_and_val_loader,
)

_ALPHA_FLOOR = 1e-4 


def inverse_softplus(target_post_softplus: float) -> float:
    y = target_post_softplus - _ALPHA_FLOOR
    if y <= 0:
        raise ValueError(
            f"target post-softplus alpha must be > {_ALPHA_FLOOR} (the floor added in forward), got {target_post_softplus}"
        )
    return math.log(math.expm1(y))


def learned_alphas(model) -> list[float]:
    return [
        float(nnf.softplus(layer.master_attn.alpha).item() + _ALPHA_FLOOR)
        for layer in model.energy_model.layers
    ]


@contextlib.contextmanager
def alpha_overridden(model, target_alpha: float):
    layers = [layer.master_attn for layer in model.energy_model.layers]
    originals = [ma.alpha.data.clone() for ma in layers]
    raw = inverse_softplus(target_alpha)
    try:
        with torch.no_grad():
            for ma in layers:
                ma.alpha.data.fill_(raw)
        yield
    finally:
        with torch.no_grad():
            for ma, orig in zip(layers, originals):
                ma.alpha.data.copy_(orig)


def _fmt_row(label: str, m: dict) -> str:
    return (
        f"{label:24s} normalized_loss={m['normalized_loss']:.6g}  energy_MAE={m['energy_mae_eV']:.6g} eV  "
        f"force_MAE={m['force_mae_eV_per_A']:.6g} eV/A"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=str, required=True, help="Path to a train_spice.py checkpoint (e.g. checkpoint_epoch_0007.pt)")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--hdf5-paths", type=str, nargs="+", default=None,
                    help="Override the checkpoint's stored hdf5 paths (needed if the dataset moved since training)")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--alphas", type=float, nargs="+", default=[0.01, 1.0, 10.0],
                    help="Post-softplus alpha values to sweep, in addition to the checkpoint's own learned baseline")
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
    print(f"Re-derived validation split: {len(val_loader.dataset)} molecules, {len(val_loader)} batches")

    baseline_alphas = learned_alphas(model)
    print(f"Learned per-layer alpha (post-softplus): {[f'{a:.6g}' for a in baseline_alphas]}\n")

    results = {}

    print("Running BASELINE (checkpoint's own learned alpha per layer)...")
    results["baseline"] = evaluate_validation(model, val_loader, device, energy_mean, energy_std, force_rms, lambda1, lambda2)
    print(_fmt_row("baseline", results["baseline"]))

    for target in args.alphas:
        label = f"alpha={target:g}"
        print(f"\nRunning {label} (all layers overridden)...")
        with alpha_overridden(model, target):
            results[label] = evaluate_validation(model, val_loader, device, energy_mean, energy_std, force_rms, lambda1, lambda2)
        print(_fmt_row(label, results[label]))

    print("\n=== Summary (alpha sensitivity sweep) ===")
    base = results["baseline"]
    for label, m in results.items():
        rel_energy = (m["energy_mae_eV"] - base["energy_mae_eV"]) / base["energy_mae_eV"] if base["energy_mae_eV"] else float("nan")
        rel_force = (m["force_mae_eV_per_A"] - base["force_mae_eV_per_A"]) / base["force_mae_eV_per_A"] if base["force_mae_eV_per_A"] else float("nan")
        print(
            f"  {label:24s} energy_MAE={m['energy_mae_eV']:.6g} eV ({rel_energy:+.2%} vs baseline)  "
            f"force_MAE={m['force_mae_eV_per_A']:.6g} eV/A ({rel_force:+.2%} vs baseline)"
        )
    print(
        "\nA flat curve across the grid means validation error is insensitive to k-space filter "
        "width in this regime (consistent with the long-range branch mattering little for "
        "gas-phase molecules); a clear minimum near the learned baseline value would instead "
        "suggest the optimizer already found a good width and the branch is doing real work."
    )

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump({
                "ckpt": args.ckpt, "epoch": ckpt.get("epoch"), "global_step": ckpt.get("global_step"),
                "learned_alphas": baseline_alphas, "alphas_swept": args.alphas, "results": results,
            }, f, indent=2, default=str)
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    sys.exit(main())