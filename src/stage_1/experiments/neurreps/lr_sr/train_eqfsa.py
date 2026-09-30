from __future__ import annotations

import argparse
import csv
import os
import time

import torch
from torch.utils.data import DataLoader, random_split

from stage_1.experiments.neurreps.lr_sr.dataset import (
    LRSRDataset, LR_SR_EDGE_FEAT_DIM, N_ATOMS, NUM_ATOM_TYPES, lr_sr_collate_fn,
)
from stage_1.train_spice import SPICEEnergyForceModel, compute_energy_and_forces, spice_loss
from utils.se3_utils import RadialNetwork


def detect_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def energy_force_stats(dataset: LRSRDataset, indices) -> tuple[float, float, float]:
    energies = torch.tensor([dataset.configs[i]["energy"].item() for i in indices])
    force_sumsq = torch.tensor([dataset.configs[i]["forces"].pow(2).sum().item() for i in indices])
    energy_mean = energies.mean().item()
    energy_std = energies.std().item()
    force_rms = (force_sumsq / (N_ATOMS * 3)).mean().sqrt().item()
    return energy_mean, energy_std, force_rms


def build_model(config: dict, device: torch.device) -> SPICEEnergyForceModel:
    return SPICEEnergyForceModel(
        radial_net=RadialNetwork,
        num_atom_types=NUM_ATOM_TYPES,
        max_degree=config["max_degree"],
        num_layers=config["num_layers"],
        feature_dim=config["feature_dim"],
        hidden_dim=config["hidden_dim"],
        edge_cutoff=config["edge_cutoff"],
        n_max=config["n_max"],
        n_master_nodes=config["num_master_nodes"],
        backbone=config["backbone"],
        edge_feat_dim=LR_SR_EDGE_FEAT_DIM,
        num_broadcast_heads=config["num_broadcast_heads"],
        rbf_num_basis=config["rbf_num_basis"],
        rbf_min=config["rbf_min"],
        rbf_cutoff=config["rbf_cutoff"],
        sigma=config.get("sigma"),
        alpha_floor=config["alpha_floor"],
    ).to(device)


def train_one(seed: int, config: dict, dataset_cache: "dict | None" = None) -> dict:
    device = config["_device"]
    torch.manual_seed(seed)
    cache_key = (config["num_configs"], seed)
    if dataset_cache is not None and cache_key in dataset_cache:
        dataset = dataset_cache[cache_key]
    else:
        _t0 = time.time()
        dataset = LRSRDataset(config["num_configs"], seed=seed)
        if config.get("log_every", 0):
            print(f"  [eqfsa seed={seed}] LRSRDataset generation (position relaxation): "
                  f"{time.time() - _t0:.1f}s for {config['num_configs']} configs", flush=True)
        if dataset_cache is not None:
            dataset_cache[cache_key] = dataset

    n_val = max(1, int(round(len(dataset) * config["val_fraction"])))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed + 100000))
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, collate_fn=lr_sr_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, collate_fn=lr_sr_collate_fn)

    energy_mean, energy_std, force_rms = energy_force_stats(dataset, train_ds.indices)

    torch.manual_seed(seed)
    model = build_model(config, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    scheduler = None
    if config.get("lr_schedule", True):
        steps_per_epoch = max(len(train_loader), 1)
        total_steps = steps_per_epoch * config["epochs"]
        warmup_steps = max(config.get("warmup_epochs", 5) * steps_per_epoch, 1)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps,
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(total_steps - warmup_steps, 1), eta_min=config.get("min_lr", 1e-6),
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps],
        )

    model.train()
    two_phase = config.get("two_phase_loss", False)
    phase1_epochs = config.get("phase1_epochs", 0)
    prev_phase = None
    for epoch in range(config["epochs"]):
        if two_phase and epoch < phase1_epochs:
            lambda_energy, lambda_force, phase = config["phase1_lambda_energy"], config["phase1_lambda_force"], 1
        elif two_phase:
            lambda_energy, lambda_force, phase = config["phase2_lambda_energy"], config["phase2_lambda_force"], 2
        else:
            lambda_energy, lambda_force, phase = config["lambda_energy"], config["lambda_force"], None
        if phase is not None and phase != prev_phase:
            print(f"  [eqfsa seed={seed}] epoch={epoch}: entering phase {phase} "
                  f"(lambda_energy={lambda_energy:g}, lambda_force={lambda_force:g})", flush=True)
            prev_phase = phase
        log_every = config.get("log_every", 0)
        is_log_epoch = log_every and (epoch % log_every == 0 or epoch == config["epochs"] - 1)
        epoch_loss, epoch_e_term, epoch_f_term, n_batches = 0.0, 0.0, 0.0, 0
        for batch in train_loader:
            positions = batch["x"].to(device, dtype=torch.get_default_dtype())
            types = batch["types"].to(device)
            masses = batch["masses"].to(device)
            mask = batch["mask"].to(device)
            V = batch["V"].to(device)
            mol_props = batch["mol_props"].to(device)
            edge_feats = batch["edge_feats"].to(device)
            optimizer.zero_grad()
            U, F_pred = compute_energy_and_forces(
                model, positions, types, masses, mask, V, mol_props, edge_feats,
            )
            U_n = (U - energy_mean) / energy_std
            U_target_n = (batch["energy"].to(device) - energy_mean) / energy_std
            F_n = F_pred / force_rms
            F_target_n = batch["forces"].to(device) / force_rms
            loss, e_term, f_term = spice_loss(
                U_n, U_target_n, F_n, F_target_n, mask, lambda_energy, lambda_force,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            if is_log_epoch:
                epoch_loss += loss.item()
                epoch_e_term += e_term.item()
                epoch_f_term += f_term.item()
                n_batches += 1
        if is_log_epoch:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  [eqfsa seed={seed}] epoch={epoch} loss={epoch_loss / max(1, n_batches):.4f} "
                  f"(energy_term={epoch_e_term / max(1, n_batches):.4f} "
                  f"force_term={epoch_f_term / max(1, n_batches):.4f}) lr={lr_now:.2e}", flush=True)

    model.eval()
    total_energy_ae, total_force_ae, total_n_configs, total_n_force_components = 0.0, 0.0, 0, 0
    for batch in val_loader:
        positions = batch["x"].to(device, dtype=torch.get_default_dtype())
        types = batch["types"].to(device)
        masses = batch["masses"].to(device)
        mask = batch["mask"].to(device)
        V = batch["V"].to(device)
        mol_props = batch["mol_props"].to(device)
        edge_feats = batch["edge_feats"].to(device)
        U, F_pred = compute_energy_and_forces(
            model, positions, types, masses, mask, V, mol_props, edge_feats,
        )
        U, F_pred = U.detach(), F_pred.detach()
        energy_target, force_target = batch["energy"].to(device), batch["forces"].to(device)
        total_energy_ae += (U - energy_target).abs().sum().item()
        total_force_ae += (F_pred - force_target).abs().sum().item()
        total_n_configs += U.shape[0]
        total_n_force_components += F_pred.numel()

    energy_mae_mev_per_atom = (total_energy_ae / max(1, total_n_configs) / N_ATOMS) * 1000.0
    force_mae_mev = (total_force_ae / max(1, total_n_force_components)) * 1000.0
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "energy_mae_mev_per_atom": energy_mae_mev_per_atom,
        "force_mae_mev": force_mae_mev, "n_trainable_params": n_trainable,
    }


def _mean_std(values: list[float]) -> tuple[float, float]:
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / max(1, n - 1)
    return mean, var ** 0.5


def run(config: dict, raw_writer: "csv.DictWriter | None" = None, raw_file=None) -> dict:
    base_seed = config["seed"]
    keys = ["energy_mae_mev_per_atom", "force_mae_mev", "n_trainable_params"]
    per_realization = {k: [] for k in keys}
    dataset_cache: dict = {}
    for i in range(config["num_realizations"]):
        seed = base_seed + i * 7919
        t0 = time.time()
        result = train_one(seed, config, dataset_cache=dataset_cache)
        realization_time_s = time.time() - t0
        for k in keys:
            per_realization[k].append(result[k])
        if raw_writer is not None:
            raw_row = {
                "variant": "eqfsa", "realization": i, "seed": seed, "train_time_s": realization_time_s,
                **{k: result[k] for k in keys},
            }
            raw_writer.writerow(raw_row)
            raw_file.flush()
            os.fsync(raw_file.fileno())
            print(f"  [eqfsa] realization {i+1}/{config['num_realizations']} (seed={seed}) done: "
                  f"energy_mae={result['energy_mae_mev_per_atom']:.4f} meV/atom  "
                  f"force_mae={result['force_mae_mev']:.4f} meV/A  ({realization_time_s:.1f}s)", flush=True)
    row = {"variant": "eqfsa", "num_realizations": config["num_realizations"]}
    for k in keys:
        mean, std = _mean_std(per_realization[k])
        row[k] = mean
        row[f"{k}_std"] = std
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", type=str, default="lr_restructure", choices=["lr_restructure", "production"],
                         help="'lr_restructure' is \"purely EqFSA\" (this script's reason for existing); "
                              "'production' is available for a quick comparison against the old backbone.")
    parser.add_argument("--num-configs", type=int, default=2000)
    parser.add_argument("--max-degree", type=int, default=2)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--feature-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--edge-cutoff", type=float, default=5.5,
                         help="Local/SR attention cutoff (Angstrom); matches physics.py's own "
                              "ewald_real_space real_cutoff=5.5, the range the target's own "
                              "short-range physics acts over.")
    parser.add_argument("--n-max", type=int, default=2)
    parser.add_argument("--num-master-nodes", type=int, default=4)
    parser.add_argument("--num-broadcast-heads", type=int, default=4,
                         help="[--backbone lr_restructure only] must divide --feature-dim")
    parser.add_argument("--rbf-num-basis", type=int, default=12, help="[--backbone lr_restructure only]")
    parser.add_argument("--rbf-min", type=float, default=0.3, help="[--backbone lr_restructure only]")
    parser.add_argument("--rbf-cutoff", type=float, default=25.0,
                         help="[--backbone lr_restructure only]")
    parser.add_argument("--sigma", type=float, default=None,
                         help="[--backbone lr_restructure only] Fixed Ewald damping width; default "
                              "None leaves alpha learned")
    parser.add_argument("--alpha-floor", type=float, default=1e-4, help="[--backbone lr_restructure only]")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no-lr-schedule", dest="lr_schedule", action="store_false")
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--lambda-energy", type=float, default=1.0)
    parser.add_argument("--lambda-force", type=float, default=0.5)
    parser.add_argument("--two-phase-loss", action="store_true",)
    parser.add_argument("--phase1-epochs", type=int, default=100)
    parser.add_argument("--phase1-lambda-energy", type=float, default=1.0)
    parser.add_argument("--phase1-lambda-force", type=float, default=1000.0)
    parser.add_argument("--phase2-lambda-energy", type=float, default=100.0)
    parser.add_argument("--phase2-lambda-force", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-realizations", type=int, default=3)
    parser.add_argument("--outdir", type=str, default="stage_1/experiments/neurreps/lr_sr/results_eqfsa")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    config = vars(args)
    config["_device"] = detect_device(config["device"])
    print(f"Using device: {config['_device']}", flush=True)

    if config["smoke_test"]:
        config["num_configs"] = 8
        config["epochs"] = 2
        config["num_realizations"] = 1

    os.makedirs(config["outdir"], exist_ok=True)
    csv_path = os.path.join(config["outdir"], "results.csv")
    raw_csv_path = os.path.join(config["outdir"], "results_raw.csv")
    raw_fieldnames = [
        "variant", "realization", "seed", "train_time_s",
        "energy_mae_mev_per_atom", "force_mae_mev", "n_trainable_params",
    ]

    with open(csv_path, "w", newline="") as csv_file, open(raw_csv_path, "w", newline="") as raw_file:
        raw_writer = csv.DictWriter(raw_file, fieldnames=raw_fieldnames)
        raw_writer.writeheader()
        print("=== variant=eqfsa ===", flush=True)
        t0 = time.time()
        result = run(config, raw_writer=raw_writer, raw_file=raw_file)
        result["train_time_s"] = time.time() - t0
        print(
            f"  energy_mae={result['energy_mae_mev_per_atom']:.4f}+/-{result['energy_mae_mev_per_atom_std']:.4f} meV/atom  "
            f"force_mae={result['force_mae_mev']:.4f}+/-{result['force_mae_mev_std']:.4f} meV/A  "
            f"({result['train_time_s']:.1f}s)",
            flush=True,
        )
        writer = csv.DictWriter(csv_file, fieldnames=list(result.keys()))
        writer.writeheader()
        writer.writerow(result)
        csv_file.flush()
        os.fsync(csv_file.fileno())
    print(f"Wrote 1 row to {csv_path}")
    print(f"Wrote per-realization rows to {raw_csv_path}")


if __name__ == "__main__":
    main()
