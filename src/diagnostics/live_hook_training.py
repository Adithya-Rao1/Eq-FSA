import argparse
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

from stage_1.model import EnergyModel, masked_mean
from stage_1.spice_dataset import SPICEDataset, spice_collate_fn, _EDGE_FEAT_DIM
from stage_1.train_spice import SPICEEnergyForceModel, compute_energy_and_forces, spice_loss
from utils.se3_utils import RadialNetwork
from diagnostics.hooks import (
    MagnitudeTracker,
    GradientTracker,
    format_hook_report,
    attach_hooks,
    attach_grad_hooks,
)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hdf5-paths", type=str, nargs="+", required=True)
    p.add_argument("--subsets", type=str, nargs="+", default=["pubchem"])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--feature-dim", type=int, default=32)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--max-degree", type=int, default=2)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--edge-cutoff", type=float, default=5.0)
    p.add_argument("--n-max", type=int, default=2)
    p.add_argument("--num-atom-types", type=int, default=23)
    p.add_argument("--box-padding", type=float, default=12.0)
    p.add_argument("--n-master-nodes", type=int, default=4)
    p.add_argument("--num-fusion-heads", type=int, default=4)
    p.add_argument("--explode-threshold", type=float, default=50.0,
                    help="max|activation| in any hooked submodule that counts as 'exploded'")
    p.add_argument("--save-dir", type=str, default="/home/ubuntu/metis-v1-storage/sr_explosion_batches")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--force-fp64", action="store_true",
                    help="cast model + batch to float64 -- run this as a second pass to check if the "
                         "explosion is precision-dependent (float32/TF32) vs. a genuine logic bug")
    p.add_argument("--no-stop-on-explosion", dest="stop_on_explosion", action="store_false")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device) if args.device else (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
    dtype = torch.float64 if args.force_fp64 else torch.float32
    print(f"Device: {device}  dtype: {dtype}")

    os.makedirs(args.save_dir, exist_ok=True)

    dataset = SPICEDataset(
        hdf5_paths=args.hdf5_paths, subsets=args.subsets, exclude_solvated=True,
        box_padding=args.box_padding, edge_cutoff=args.edge_cutoff,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=spice_collate_fn)
    energy_mean, energy_std, force_rms = dataset.energy_force_stats()
    print(f"Dataset: {len(dataset)} conformers. energy_mean={energy_mean:.3f} energy_std={energy_std:.3f} force_rms={force_rms:.3f}")

    model = SPICEEnergyForceModel(
        radial_net=RadialNetwork, num_atom_types=args.num_atom_types, max_degree=args.max_degree,
        num_layers=args.num_layers, feature_dim=args.feature_dim, hidden_dim=args.hidden_dim,
        edge_cutoff=args.edge_cutoff, n_max=args.n_max,
        n_master_nodes=args.n_master_nodes, num_fusion_heads=args.num_fusion_heads,
    ).to(device=device, dtype=dtype)

    tracker = attach_hooks(model)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    step = 0
    t0 = time.time()
    for batch in loader:
        if step >= args.max_steps:
            break
        batch = {k: (v.to(device=device, dtype=dtype) if torch.is_tensor(v) and torch.is_floating_point(v)
                     else v.to(device) if torch.is_tensor(v) else v)
                  for k, v in batch.items()}

        tracker.reset()
        opt.zero_grad(set_to_none=True)

        U, F = compute_energy_and_forces(
            model, batch["all_solute_x"], batch["all_solute_types"], batch["all_solute_masses"],
            batch["all_solute_mask"], batch["V"], batch["mol_props"], batch["all_solute_edge_feats"],
        )
        U_n = (U - energy_mean) / energy_std
        U_target_n = (batch["energy"] - energy_mean) / energy_std
        F_n = F / force_rms
        F_target_n = batch["solute_forces"] / force_rms
        loss, _, _ = spice_loss(U_n, U_target_n, F_n, F_target_n, batch["all_solute_mask"], 1.0, 1.0)

        max_activation = max((v for v in tracker.stats.values() if v == v), default=0.0)  # skip NaN in this max
        has_nonfinite_activation = any(v != v for v in tracker.stats.values())

        loss.backward()

        grads = [pm.grad for pm in model.parameters() if pm.grad is not None]
        total_norm = torch.stack([g.norm() for g in grads]).norm().item() if grads else 0.0
        is_finite_grad = all(torch.isfinite(g).all() for g in grads)

        exploded = (max_activation > args.explode_threshold) or has_nonfinite_activation or (not is_finite_grad)

        if step % 10 == 0 or exploded:
            print(f"step {step:5d} | loss={loss.item():.4f} | grad_norm={total_norm:.4f} | "
                  f"max_hooked_activation={max_activation:.4g} | non_finite_activation={has_nonfinite_activation} | "
                  f"finite_grad={is_finite_grad}")

        if exploded:
            print("\n" + "=" * 90)
            print(f"EXPLOSION DETECTED at step {step}")
            print("=" * 90)
            print("Top offending submodules this step (NaN first, then largest |activation|):")
            print(tracker.report())

            save_path = os.path.join(args.save_dir, f"explosion_step{step}.pt")
            torch.save({
                "step": step,
                "batch": {k: v for k, v in batch.items() if torch.is_tensor(v) or isinstance(v, list)},
                "model_state_dict": model.state_dict(),
                "hook_stats": dict(tracker.stats),
                "args": vars(args),
            }, save_path)
            print(f"\nSaved offending batch + model state to: {save_path}")
            print(f"Replay it with: python -m diagnostics.replay_batch --path {save_path}")

            if args.stop_on_explosion:
                tracker.remove()
                return

        if is_finite_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            opt.step()
        else:
            opt.zero_grad(set_to_none=True)

        step += 1

    tracker.remove()
    print(f"\nCompleted {step} steps in {time.time()-t0:.1f}s with no explosion "
          f"(threshold={args.explode_threshold}). Consider lowering --explode-threshold "
          f"or running --max-steps higher if the training log's spikes are sparser than this run's coverage.")


if __name__ == "__main__":
    sys.exit(main())
