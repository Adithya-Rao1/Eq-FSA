from __future__ import annotations

import argparse
import copy
import csv
import os
import time

import torch
from torch.utils.data import DataLoader, random_split

from stage_1.experiments.neurreps.lr_sr.dataset import (
    LRSRDataset, N_ATOMS, POSITIVE_TYPE_IDX, lr_sr_collate_fn,
)
from stage_1.experiments.neurreps.lr_sr.mace_lr_model import (
    MACE_OFF_DTYPE, MaceLRModel, build_z_table, compute_energy_and_forces, load_frozen_mace_off,
)


def detect_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _types_to_charges(types: torch.Tensor) -> torch.Tensor:
    return torch.where(types == POSITIVE_TYPE_IDX, 1.0, -1.0).to(torch.get_default_dtype())


def energy_force_stats(dataset: LRSRDataset, indices) -> tuple[float, float, float]:
    energies = torch.tensor([dataset.configs[i]["energy"].item() for i in indices])
    force_sumsq = torch.tensor([dataset.configs[i]["forces"].pow(2).sum().item() for i in indices])
    energy_mean = energies.mean().item()
    energy_std = energies.std().item()
    force_rms = (force_sumsq / (N_ATOMS * 3)).mean().sqrt().item()
    return energy_mean, energy_std, force_rms


def build_model(variant: str, mace_model, mace_init_state: dict, config: dict, device) -> MaceLRModel:
    mace_model.load_state_dict(mace_init_state)
    for p in mace_model.parameters():
        p.requires_grad_(False)
    return MaceLRModel(
        mace_model, variant, feature_dim=config["feature_dim"], hidden_dim=config["hidden_dim"],
        n_max=config["n_max"], num_master_nodes=config["num_master_nodes"], rsa_sigma=config["rsa_sigma"],
    ).to(device)


def train_one(variant: str, mace_model, mace_init_state: dict, z_table, config: dict, seed: int) -> dict:
    device = config["_device"]
    torch.manual_seed(seed)
    dataset = LRSRDataset(config["num_configs"], seed=seed)

    n_val = max(1, int(round(len(dataset) * config["val_fraction"])))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed + 100000))
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, collate_fn=lr_sr_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, collate_fn=lr_sr_collate_fn)

    energy_mean, energy_std, force_rms = energy_force_stats(dataset, train_ds.indices)

    torch.manual_seed(seed)
    model = build_model(variant, mace_model, mace_init_state, config, device)
    cutoff = float(model.mace_model.r_max)

    trainable = [p for p in model.parameters() if p.requires_grad]
    if trainable:
        optimizer = torch.optim.AdamW(trainable, lr=config["lr"], weight_decay=1e-4)
        model.train()
        for _ in range(config["epochs"]):
            for batch in train_loader:
                positions = batch["x"].to(device, dtype=torch.get_default_dtype())
                types = batch["types"]
                charges = _types_to_charges(types).to(device)
                boxes = batch["V"][:, 0, 0]
                optimizer.zero_grad()
                U, F_pred = compute_energy_and_forces(model, positions, types, charges, boxes, z_table, cutoff, device)
                U_n = (U - energy_mean) / energy_std
                U_target_n = (batch["energy"].to(device) - energy_mean) / energy_std
                F_n = F_pred / force_rms
                F_target_n = batch["forces"].to(device) / force_rms
                loss = (config["lambda_energy"] * (U_n - U_target_n) ** 2).mean() + (
                    config["lambda_force"] * ((F_n - F_target_n) ** 2).mean(dim=(-2, -1))
                ).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()

    model.eval()
    total_energy_ae, total_force_ae, total_n_configs, total_n_force_components = 0.0, 0.0, 0, 0
    for batch in val_loader:
        positions = batch["x"].to(device, dtype=torch.get_default_dtype())
        types = batch["types"]
        charges = _types_to_charges(types).to(device)
        boxes = batch["V"][:, 0, 0]
        U, F_pred = compute_energy_and_forces(model, positions, types, charges, boxes, z_table, cutoff, device)
        U, F_pred = U.detach(), F_pred.detach()
        energy_target, force_target = batch["energy"].to(device), batch["forces"].to(device)
        total_energy_ae += (U - energy_target).abs().sum().item()
        total_force_ae += (F_pred - force_target).abs().sum().item()
        total_n_configs += U.shape[0]
        total_n_force_components += F_pred.numel()

    energy_mae_mev_per_atom = (total_energy_ae / max(1, total_n_configs) / N_ATOMS) * 1000.0
    force_mae_mev = (total_force_ae / max(1, total_n_force_components)) * 1000.0
    n_trainable = sum(p.numel() for p in trainable) if trainable else 0

    return {
        "variant": variant, "energy_mae_mev_per_atom": energy_mae_mev_per_atom,
        "force_mae_mev": force_mae_mev, "n_trainable_params": n_trainable,
    }


def _mean_std(values: list[float]) -> tuple[float, float]:
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / max(1, n - 1)
    return mean, var ** 0.5


def run_variant(variant: str, mace_model, mace_init_state, z_table, config: dict) -> dict:
    base_seed = config["seed"]
    num_realizations = 1 if variant == "none" else config["num_realizations"] 
    keys = ["energy_mae_mev_per_atom", "force_mae_mev", "n_trainable_params"]
    per_realization = {k: [] for k in keys}
    for i in range(num_realizations):
        result = train_one(variant, mace_model, mace_init_state, z_table, config, seed=base_seed + i * 7919)
        for k in keys:
            per_realization[k].append(result[k])
    row = {"variant": variant, "num_realizations": num_realizations}
    for k in keys:
        mean, std = _mean_std(per_realization[k])
        row[k] = mean
        row[f"{k}_std"] = std
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", type=str, nargs="+", default=["none", "eqfsa", "rsa"])
    parser.add_argument("--mace-model-size", type=str, default="small")
    parser.add_argument("--num-configs", type=int, default=2000)
    parser.add_argument("--n-max", type=int, default=2)
    parser.add_argument("--num-master-nodes", type=int, default=4)
    parser.add_argument("--rsa-sigma", type=float, default=5.0)
    parser.add_argument("--feature-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-energy", type=float, default=1.0)
    parser.add_argument("--lambda-force", type=float, default=0.5)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-realizations", type=int, default=3)
    parser.add_argument("--outdir", type=str, default="stage_1/experiments/neurreps/lr_sr/results_mace_lr")
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

    torch.set_default_dtype(torch.float32 if MACE_OFF_DTYPE == "float32" else torch.float64)

    print(f"Loading frozen MACE-OFF ({config['mace_model_size']})...", flush=True)
    mace_model = load_frozen_mace_off(config["mace_model_size"], device=str(config["_device"]), default_dtype=MACE_OFF_DTYPE)
    mace_init_state = copy.deepcopy(mace_model.state_dict())
    z_table = build_z_table(mace_model)

    os.makedirs(config["outdir"], exist_ok=True)
    csv_path = os.path.join(config["outdir"], "results.csv")

    rows = []
    writer = None
    with open(csv_path, "w", newline="") as csv_file:
        for variant in config["variants"]:
            print(f"=== variant={variant} ===", flush=True)
            t0 = time.time()
            result = run_variant(variant, mace_model, mace_init_state, z_table, config)
            result["train_time_s"] = time.time() - t0
            print(
                f"  energy_mae={result['energy_mae_mev_per_atom']:.4f}+/-{result['energy_mae_mev_per_atom_std']:.4f} meV/atom  "
                f"force_mae={result['force_mae_mev']:.4f}+/-{result['force_mae_mev_std']:.4f} meV/A  "
                f"({result['train_time_s']:.1f}s)",
                flush=True,
            )
            rows.append(result)
            if writer is None:
                writer = csv.DictWriter(csv_file, fieldnames=list(result.keys()))
                writer.writeheader()
            writer.writerow(result)
            csv_file.flush()
            os.fsync(csv_file.fileno())
    print(f"Wrote {len(rows)} rows to {csv_path}")


if __name__ == "__main__":
    main()
