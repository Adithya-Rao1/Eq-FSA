from __future__ import annotations

import argparse
import csv
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from utils.se3_utils import RadialNetwork

from stage_1.experiments.neurreps.lr_sr.dataset import LRSRDataset, NUM_ATOM_TYPES, LR_SR_EDGE_FEAT_DIM, lr_sr_collate_fn
from stage_1.experiments.neurreps.lr_sr.lr_sr_model import LRSREnergyForceModel, compute_energy_and_forces


def detect_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _to_device(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def energy_force_loss(U, U_hat, F_, F_hat, lambda1: float, lambda2: float):
    energy_term = lambda1 * (U - U_hat) ** 2
    force_term = lambda2 * ((F_ - F_hat) ** 2).mean(dim=(-2, -1))
    return (energy_term + force_term).mean(), energy_term.mean(), force_term.mean()


def build_model(variant: str, config: dict) -> LRSREnergyForceModel:
    return LRSREnergyForceModel(
        radial_net=RadialNetwork, num_atom_types=NUM_ATOM_TYPES, max_degree=config["max_degree"],
        num_layers=config["num_layers"], feature_dim=config["feature_dim"],
        hidden_dim=config["hidden_dim"], edge_cutoff=config["cutoff"], n_max=config["n_max"],
        edge_feat_dim=LR_SR_EDGE_FEAT_DIM, ablate_lr=(variant == "sr_only"),
    )


def train_one(variant: str, config: dict, seed: int) -> dict:
    torch.manual_seed(seed)
    dataset = LRSRDataset(config["num_configs"], seed=seed)

    n_val = max(1, int(round(len(dataset) * config["val_fraction"])))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed + 100000),
    )
    train_loader = DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True, collate_fn=lr_sr_collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["batch_size"], shuffle=False, collate_fn=lr_sr_collate_fn,
    )

    device = config["_device"]
    torch.manual_seed(seed)
    model = build_model(variant, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)

    for _ in range(config["epochs"]):
        model.train()
        for batch in train_loader:
            batch = _to_device(batch, device)
            optimizer.zero_grad()
            U, F_pred = compute_energy_and_forces(
                model, batch["x"], batch["types"], batch["masses"], batch["mask"], batch["V"],
                batch["mol_props"], batch["edge_feats"],
            )
            loss, _, _ = energy_force_loss(
                U, batch["energy"], F_pred, batch["forces"],
                config["lambda_energy"], config["lambda_force"],
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    model.eval()
    total_energy_ae, total_force_ae, total_n_configs, total_n_force_components = 0.0, 0.0, 0, 0
    for batch in val_loader:
        batch = _to_device(batch, device)
        U, F_pred = compute_energy_and_forces(
            model, batch["x"], batch["types"], batch["masses"], batch["mask"], batch["V"],
            batch["mol_props"], batch["edge_feats"],
        )
        U, F_pred = U.detach(), F_pred.detach()
        total_energy_ae += (U - batch["energy"]).abs().sum().item()
        total_force_ae += (F_pred - batch["forces"]).abs().sum().item()
        total_n_configs += U.shape[0]
        total_n_force_components += F_pred.numel()

    energy_mae = total_energy_ae / max(1, total_n_configs)
    force_mae = total_force_ae / max(1, total_n_force_components)
    n_params = sum(p.numel() for p in model.parameters())

    return {"variant": variant, "energy_mae": energy_mae, "force_mae": force_mae, "n_params": n_params}


def _mean_std(values: list[float]) -> tuple[float, float]:
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / max(1, n - 1)
    return mean, var ** 0.5


def run_variant(variant: str, config: dict) -> dict:
    base_seed = config["seed"]
    keys = ["energy_mae", "force_mae", "n_params"]
    per_realization = {k: [] for k in keys}
    for i in range(config["num_realizations"]):
        result = train_one(variant, config, seed=base_seed + i * 7919)
        for k in keys:
            per_realization[k].append(result[k])
    row = {"variant": variant, "num_realizations": config["num_realizations"]}
    for k in keys:
        mean, std = _mean_std(per_realization[k])
        row[k] = mean
        row[f"{k}_std"] = std
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", type=str, nargs="+", default=["lr_full", "sr_only"])
    parser.add_argument("--num-configs", type=int, default=2000)
    parser.add_argument("--cutoff", type=float, default=5.0, help="SR receptive field (and LR's, per the reference protocol)")
    parser.add_argument("--num-layers", type=int, default=2, help="SR message-passing depth, per the reference protocol")
    parser.add_argument("--max-degree", type=int, default=2)
    parser.add_argument("--n-max", type=int, default=1)
    parser.add_argument("--feature-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-energy", type=float, default=1.0)
    parser.add_argument("--lambda-force", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-realizations", type=int, default=3)
    parser.add_argument("--outdir", type=str, default="stage_1/experiments/neurreps/lr_sr/results")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    config = vars(args)
    config["_device"] = detect_device(config["device"])
    print(f"Using device: {config['_device']}", flush=True)

    if config["smoke_test"]:
        config["num_configs"] = 20
        config["epochs"] = 2
        config["num_realizations"] = 1

    os.makedirs(config["outdir"], exist_ok=True)
    csv_path = os.path.join(config["outdir"], "results.csv")

    rows = []
    writer = None
    with open(csv_path, "w", newline="") as csv_file:
        for variant in config["variants"]:
            print(f"=== variant={variant} ({config['num_realizations']} realizations) ===", flush=True)
            t0 = time.time()
            result = run_variant(variant, config)
            result["train_time_s"] = time.time() - t0
            print(
                f"  energy_mae={result['energy_mae']:.4f}+/-{result['energy_mae_std']:.4f} eV  "
                f"force_mae={result['force_mae']:.4f}+/-{result['force_mae_std']:.4f} eV/A  "
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
