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


def train_one(
    variant: str, mace_model, mace_init_state: dict, z_table, config: dict, seed: int,
    dataset_cache: "dict | None" = None,
) -> dict:
    device = config["_device"]
    torch.manual_seed(seed)
    cache_key = (config["num_configs"], seed)
    if dataset_cache is not None and cache_key in dataset_cache:
        dataset = dataset_cache[cache_key]
    else:
        _t_dataset0 = time.time()
        dataset = LRSRDataset(config["num_configs"], seed=seed)
        if config.get("log_every", 0):
            print(f"  [{variant} seed={seed}] LRSRDataset generation (position relaxation): "
                  f"{time.time() - _t_dataset0:.1f}s for {config['num_configs']} configs", flush=True)
        if dataset_cache is not None:
            dataset_cache[cache_key] = dataset

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
                print(f"  [{variant} seed={seed}] epoch={epoch}: entering phase {phase} "
                      f"(lambda_energy={lambda_energy:g}, lambda_force={lambda_force:g})", flush=True)
                prev_phase = phase
            log_every = config.get("log_every", 0)
            is_log_epoch = log_every and (epoch % log_every == 0 or epoch == config["epochs"] - 1)
            epoch_loss, epoch_e_term, epoch_f_term, n_batches = 0.0, 0.0, 0.0, 0
            for batch in train_loader:
                positions = batch["x"].to(device, dtype=torch.get_default_dtype())
                types = batch["types"]
                charges = _types_to_charges(types).to(device)
                boxes = batch["V"][:, 0, 0]
                optimizer.zero_grad()
                U, F_pred = compute_energy_and_forces(
                    model, positions, types, charges, boxes, z_table, cutoff, device, idx=batch["idx"],
                )
                U_n = (U - energy_mean) / energy_std
                U_target_n = (batch["energy"].to(device) - energy_mean) / energy_std
                F_n = F_pred / force_rms
                F_target_n = batch["forces"].to(device) / force_rms
                e_term = (lambda_energy * (U_n - U_target_n) ** 2).mean()
                f_term = (lambda_force * ((F_n - F_target_n) ** 2).mean(dim=(-2, -1))).mean()
                loss = e_term + f_term
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
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
                print(f"  [{variant} seed={seed}] epoch={epoch} loss={epoch_loss / max(1, n_batches):.4f} "
                      f"(energy_term={epoch_e_term / max(1, n_batches):.4f} "
                      f"force_term={epoch_f_term / max(1, n_batches):.4f}) lr={lr_now:.2e}", flush=True)

    model.eval()
    total_energy_ae, total_force_ae, total_n_configs, total_n_force_components = 0.0, 0.0, 0, 0
    for batch in val_loader:
        positions = batch["x"].to(device, dtype=torch.get_default_dtype())
        types = batch["types"]
        charges = _types_to_charges(types).to(device)
        boxes = batch["V"][:, 0, 0]
        U, F_pred = compute_energy_and_forces(
            model, positions, types, charges, boxes, z_table, cutoff, device, idx=batch["idx"],
        )
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


def run_variant(
    variant: str, mace_model, mace_init_state, z_table, config: dict,
    raw_writer: "csv.DictWriter | None" = None, raw_file=None,
    dataset_cache: "dict | None" = None,
) -> dict:
    base_seed = config["seed"]
    num_realizations = 1 if variant == "none" else config["num_realizations"]
    keys = ["energy_mae_mev_per_atom", "force_mae_mev", "n_trainable_params"]
    per_realization = {k: [] for k in keys}
    for i in range(num_realizations):
        seed = base_seed + i * 7919
        t0 = time.time()
        result = train_one(variant, mace_model, mace_init_state, z_table, config, seed=seed,
                            dataset_cache=dataset_cache)
        realization_time_s = time.time() - t0
        for k in keys:
            per_realization[k].append(result[k])
        if raw_writer is not None:
            raw_row = {
                "variant": variant, "realization": i, "seed": seed, "train_time_s": realization_time_s,
                **{k: result[k] for k in keys},
            }
            raw_writer.writerow(raw_row)
            raw_file.flush()
            os.fsync(raw_file.fileno())
            print(f"  [{variant}] realization {i+1}/{num_realizations} (seed={seed}) done: "
                  f"energy_mae={result['energy_mae_mev_per_atom']:.4f} meV/atom  "
                  f"force_mae={result['force_mae_mev']:.4f} meV/A  ({realization_time_s:.1f}s)", flush=True)
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
    parser.add_argument("--log-every", type=int, default=0,
                         help="Print mean training loss (energy/force terms + current lr) every N "
                              "epochs (0=silent, the previous default -- no per-epoch visibility at "
                              "all otherwise).")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no-lr-schedule", dest="lr_schedule", action="store_false",
                         help="Disable the LR schedule (default: on) -- linear warmup then cosine "
                              "decay to --min-lr, exactly matching train_spice.py's scheduler "
                              "(LinearLR -> CosineAnnealingLR via SequentialLR, stepped per-batch).")
    parser.add_argument("--warmup-epochs", type=int, default=5,
                         help="Epochs of linear LR warmup before cosine annealing begins (matches "
                              "train_spice.py's own default).")
    parser.add_argument("--min-lr", type=float, default=1e-6,
                         help="Cosine annealing floor (matches train_spice.py's own default).")
    parser.add_argument("--lambda-energy", type=float, default=1.0)
    parser.add_argument("--lambda-force", type=float, default=0.5)
    parser.add_argument("--two-phase-loss", action="store_true",
                         help="Curriculum loss: force-dominated for the first --phase1-epochs "
                              "(denser per-config supervision, N*3 components vs. 1 energy scalar), "
                              "then energy-dominated for the rest. Overrides --lambda-energy/"
                              "--lambda-force entirely when set.")
    parser.add_argument("--phase1-epochs", type=int, default=100)
    parser.add_argument("--phase1-lambda-energy", type=float, default=1.0)
    parser.add_argument("--phase1-lambda-force", type=float, default=1000.0)
    parser.add_argument("--phase2-lambda-energy", type=float, default=100.0)
    parser.add_argument("--phase2-lambda-force", type=float, default=1.0)
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
    raw_csv_path = os.path.join(config["outdir"], "results_raw.csv")
    raw_fieldnames = [
        "variant", "realization", "seed", "train_time_s",
        "energy_mae_mev_per_atom", "force_mae_mev", "n_trainable_params",
    ]

    rows = []
    writer = None
    dataset_cache: dict = {}
    with open(csv_path, "w", newline="") as csv_file, open(raw_csv_path, "w", newline="") as raw_file:
        raw_writer = csv.DictWriter(raw_file, fieldnames=raw_fieldnames)
        raw_writer.writeheader()
        for variant in config["variants"]:
            print(f"=== variant={variant} ===", flush=True)
            t0 = time.time()
            result = run_variant(
                variant, mace_model, mace_init_state, z_table, config,
                raw_writer=raw_writer, raw_file=raw_file, dataset_cache=dataset_cache,
            )
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
    print(f"Wrote per-realization rows to {raw_csv_path}")


if __name__ == "__main__":
    main()
