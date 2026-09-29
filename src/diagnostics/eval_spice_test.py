from __future__ import annotations

import argparse
import csv
import os

import torch
from torch.utils.data import DataLoader, Subset

from stage_1.spice_dataset import SPICEDataset, spice_collate_fn
from stage_1.train_spice import SPICEEnergyForceModel, compute_energy_and_forces, detect_device
from utils.se3_utils import RadialNetwork


def build_model_from_config(config: dict) -> SPICEEnergyForceModel:
    return SPICEEnergyForceModel(
        radial_net=RadialNetwork,
        num_atom_types=config["num_atom_types"],
        max_degree=config["max_degree"],
        num_layers=config["num_layers"],
        feature_dim=config["feature_dim"],
        hidden_dim=config["hidden_dim"],
        edge_cutoff=config["edge_cutoff"],
        n_max=config["n_max"],
        pw_e_cutoff=config.get("pw_e_cutoff"),
        n_master_nodes=config.get("n_master_nodes", 4),
        backbone=config.get("backbone", "production"),
        num_fusion_heads=config.get("num_fusion_heads", 4),
        ablate_max_degree=config.get("ablate_max_degree"),
        num_broadcast_heads=config.get("num_broadcast_heads", 4),
        rbf_num_basis=config.get("rbf_num_basis", 12),
        rbf_min=config.get("rbf_min", 0.3),
        rbf_cutoff=config.get("rbf_cutoff", 25.0),
        sigma=config.get("sigma"),
        alpha_floor=config.get("alpha_floor", 1e-4),
        use_norms=config.get("use_norms", True),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", type=str, required=True,
                         help="Directory containing dataset_split.pt and checkpoint_epoch_*.pt / best_checkpoint.pt")
    parser.add_argument("--checkpoint", type=str, default=None,
                         help="Explicit checkpoint path (default: <ckpt-dir>/best_checkpoint.pt)")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val", "train"],
                         help="Which saved split to evaluate")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out-csv", type=str, default=None,
                         help="Per-molecule CSV output path (default: <ckpt-dir>/eval_<split>.csv)")
    args = parser.parse_args()

    device = detect_device(args.device)

    split_path = os.path.join(args.ckpt_dir, "dataset_split.pt")
    split = torch.load(split_path, map_location="cpu")
    identities = split[f"{args.split}_identities"]
    print(
        f"Loaded split from {split_path}: {len(identities)} {args.split} examples "
        f"(n_total={split['n_total']}, seed={split['seed']})"
    )

    dataset = SPICEDataset(
        hdf5_paths=split["hdf5_paths"],
        subsets=split.get("subsets"),
        exclude_solvated=True,
        box_padding=split["box_padding"],
        edge_cutoff=split["edge_cutoff"],
    )
    indices = dataset.indices_for_identities(identities)
    if len(indices) != len(identities):
        raise RuntimeError

    loader = DataLoader(
        Subset(dataset, indices), batch_size=args.batch_size, shuffle=False,
        collate_fn=spice_collate_fn, num_workers=args.num_workers,
    )

    ckpt_path = args.checkpoint or os.path.join(args.ckpt_dir, "best_checkpoint.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt["config"]
    model = build_model_from_config(config).to(device)
    load_result = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if load_result.unexpected_keys:
        raise RuntimeError
    unexplained = [
        k for k in load_result.missing_keys
        if not any(f".{p}." in k for p in ("spatial_radial", "spatial_query_proj", "node_k_gate"))
    ]
    if unexplained:
        raise RuntimeError
    if load_result.missing_keys:
        print(f"Checkpoint predates {len(load_result.missing_keys)} MasterNodeAttention key(s) added ")
    model.eval()
    print(f"Loaded {ckpt_path} (epoch={ckpt.get('epoch')}, val_loss={ckpt.get('val_loss')})")
    print(f"energy_mean={config.get('energy_mean')}, energy_std={config.get('energy_std')}, "f"force_rms={config.get('force_rms')}")

    out_csv = args.out_csv or os.path.join(args.ckpt_dir, f"eval_{args.split}.csv")
    rows = []
    total_energy_ae, total_force_ae, n_force_components, n_configs = 0.0, 0.0, 0, 0

    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        U, F = compute_energy_and_forces(
            model,
            batch["all_solute_x"], batch["all_solute_types"],
            batch["all_solute_masses"], batch["all_solute_mask"], batch["V"],
            batch["mol_props"], batch["all_solute_edge_feats"],
        )
        U = U.detach()
        F = F.detach()
        mask = batch["all_solute_mask"].bool()
        n_atoms = mask.sum(dim=1).clamp(min=1)

        energy_ae = (U - batch["energy"]).abs()
        force_ae = (F - batch["solute_forces"]).abs() * mask.unsqueeze(-1)

        for i in range(U.shape[0]):
            rows.append({
                "idx": batch["idx"][i],
                "smiles": batch["smiles"][i],
                "subset": batch["subset"][i],
                "n_atoms": int(n_atoms[i].item()),
                "energy_true_eV": float(batch["energy"][i].item()),
                "energy_pred_eV": float(U[i].item()),
                "energy_ae_meV_per_atom": float(energy_ae[i].item() / n_atoms[i].item() * 1000.0),
                "force_ae_mean_meV_per_A": float(force_ae[i].sum().item() / (n_atoms[i].item() * 3) * 1000.0),
            })

        total_energy_ae += float((energy_ae / n_atoms).sum().item())
        total_force_ae += float(force_ae.sum().item())
        n_force_components += int(mask.unsqueeze(-1).expand_as(F).sum().item())
        n_configs += U.shape[0]

    energy_mae_mev_per_atom = total_energy_ae / max(n_configs, 1) * 1000.0
    force_mae_mev_per_A = total_force_ae / max(n_force_components, 1) * 1000.0

    if rows:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"\n=== {args.split} split ({n_configs} configurations) ===")
    print(f"Energy MAE: {energy_mae_mev_per_atom:.4f} meV/atom")
    print(f"Force MAE:  {force_mae_mev_per_A:.4f} meV/A")
    print(f"Per-molecule results written to {out_csv}")


if __name__ == "__main__":
    main()