from __future__ import annotations

import argparse
import csv
import os
import time

import torch
from torch.utils.data import DataLoader

from stage_1.md17_dataset import MD17Dataset
from stage_1.spice_dataset import spice_collate_fn
from stage_1.train_spice import SPICEEnergyForceModel, SPICETrainer, detect_device
from utils.se3_utils import RadialNetwork


def build_dataloaders(config: dict, device: torch.device | None = None):
    train_ds = MD17Dataset(
        molecule=config["molecule"], root=config["data_root"], split_idx=config["split_idx"],
        train=True, box_padding=config["box_padding"], edge_cutoff=config["edge_cutoff"],
    )
    val_ds = MD17Dataset(
        molecule=config["molecule"], root=config["data_root"], split_idx=config["split_idx"],
        train=False, box_padding=config["box_padding"], edge_cutoff=config["edge_cutoff"],
        energy_shift=train_ds.energy_shift,
    )
    energy_mean, energy_std, force_rms = train_ds.energy_force_stats()

    num_workers = config.get("num_workers", 0)
    pin_memory = device.type == "cuda" if device is not None else torch.cuda.is_available()
    loader_kwargs: dict = dict(
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=num_workers > 0,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = config.get("prefetch_factor", 2)

    train_loader = DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True, drop_last=True,
        collate_fn=spice_collate_fn, **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["batch_size"], shuffle=False,
        collate_fn=spice_collate_fn, **loader_kwargs,
    )
    return train_loader, val_loader, energy_mean, energy_std, force_rms


def run_training(config: dict, device: torch.device | None = None):
    device = device or detect_device(config.get("device"))
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    train_loader, val_loader, energy_mean, energy_std, force_rms = build_dataloaders(config, device=device)
    config["energy_mean"] = energy_mean
    config["energy_std"] = energy_std
    config["force_rms"] = force_rms
    print(
        f"[{config['molecule']} split={config['split_idx']}] Dataset stats (train split): "
        f"energy_mean={energy_mean:.4f} eV, energy_std={energy_std:.4f} eV, "
        f"force_rms={force_rms:.4f} eV/A", flush=True,
    )

    model = SPICEEnergyForceModel(
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
        backbone=config.get("backbone", "lr_restructure"),
        num_fusion_heads=config.get("num_fusion_heads", 4),
        ablate_max_degree=config.get("ablate_max_degree"),
        grad_checkpoint=config.get("grad_checkpoint", False),
        num_broadcast_heads=config.get("num_broadcast_heads", 4),
        rbf_num_basis=config.get("rbf_num_basis", 12),
        rbf_min=config.get("rbf_min", 0.3),
        rbf_cutoff=config.get("rbf_cutoff", 25.0),
        sigma=config.get("sigma"),
        alpha_floor=config.get("alpha_floor", 1e-4),
        use_norms=config.get("use_norms", True),
    )
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}", flush=True)

    trainer = SPICETrainer(model, train_loader, val_loader, config, device)
    start_epoch = 0
    if config.get("resume"):
        start_epoch = trainer.load_checkpoint(config["resume"]) + 1

    try:
        best_val_loss = float("inf")
        for epoch in range(start_epoch, config["n_epochs"]):
            trainer.train_epoch(epoch)
            if (epoch + 1) % config.get("val_every", 1) == 0:
                val_loss = trainer.validate(epoch)
                is_best = val_loss < best_val_loss
                best_val_loss = min(best_val_loss, val_loss)
                if (epoch + 1) % config.get("save_every", 1) == 0:
                    trainer.save_checkpoint(epoch, val_loss, is_best)
    finally:
        trainer.close()

    return trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molecule", type=str, required=True,choices=["aspirin", "ethanol", "malonaldehyde", "benzene"])
    parser.add_argument("--split-idx", type=int, default=0, choices=[0, 1, 2, 3, 4],)
    parser.add_argument("--data-root", type=str, default="data/rmd17",)
    parser.add_argument("--ckpt-dir", type=str, default="stage_1/results_md17")
    parser.add_argument("--csv-path", type=str, default=None,)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-epochs", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda1", type=float, default=1.0, help="Energy loss weight")
    parser.add_argument("--lambda2", type=float, default=100.0, help="Force loss weight (rMD17's 1000-config regime is force-starved relative to SPICE's 1.1M-conformation scale; weighted toward forces like the field's own rMD17 protocols)")
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--max-degree", type=int, default=2)
    parser.add_argument("--ablate-max-degree", type=int, default=None, choices=[0, 1])
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--edge-cutoff", type=float, default=5.0)
    parser.add_argument("--box-padding", type=float, default=12.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--grad-checkpoint", action="store_true")
    parser.add_argument("--n-max", type=int, default=1)
    parser.add_argument("--pw-e-cutoff", type=float, default=None)
    parser.add_argument("--num-atom-types", type=int, default=23)
    parser.add_argument("--n-master-nodes", type=int, default=8)
    parser.add_argument("--backbone", type=str, default="lr_restructure", choices=["lr_restructure", "production"])
    parser.add_argument("--num-fusion-heads", type=int, default=4)
    parser.add_argument("--num-broadcast-heads", type=int, default=4)
    parser.add_argument("--rbf-num-basis", type=int, default=12)
    parser.add_argument("--rbf-min", type=float, default=0.3)
    parser.add_argument("--rbf-cutoff", type=float, default=25.0)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--alpha-floor", type=float, default=1e-4)
    parser.add_argument("--use-norms", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    config = vars(args)

    if config["smoke_test"]:
        config["n_epochs"] = 1
        config["batch_size"] = 4
        config["feature_dim"] = 8
        config["hidden_dim"] = 16
        config["num_layers"] = 2
        config["n_master_nodes"] = 2
        config["num_workers"] = 0
        config["val_every"] = 1
        config["save_every"] = 1

    if config["csv_path"] is None:
        config["csv_path"] = os.path.join(config["ckpt_dir"], "loss_log.csv")
    os.makedirs(config["ckpt_dir"], exist_ok=True)

    run_training(config)


if __name__ == "__main__":
    main()
