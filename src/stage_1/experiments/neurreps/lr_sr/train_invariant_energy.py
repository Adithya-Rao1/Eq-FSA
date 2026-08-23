from __future__ import annotations

import argparse
import csv
import os
import time

import torch
import torch.nn.functional as F

from stage_1.experiments.neurreps.lr_sr.invariant_energy_data import generate_batch, make_wavevector_grid
from stage_1.experiments.neurreps.lr_sr.invariant_energy_model import EqFSAEncoder, RSAEncoder


def detect_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def build_model(encoder: str, num_outputs: int, config: dict) -> torch.nn.Module:
    if encoder == "eqfsa":
        return EqFSAEncoder(
            num_master_nodes=config["num_master_nodes"], feature_dim=config["feature_dim"],
            hidden_dim=config["hidden_dim"], n_max=config["n_max"], num_outputs=num_outputs,
        )
    if encoder == "rsa":
        return RSAEncoder(
            feature_dim=config["feature_dim"], hidden_dim=config["hidden_dim"],
            n_max=config["n_max"], sigma=config["rsa_sigma"], num_outputs=num_outputs,
        )
    raise ValueError(f"Unknown encoder {encoder!r}")


def train_one(encoder: str, n_atoms: int, config: dict, seed: int) -> dict:
    device = config["_device"]
    generator = torch.Generator().manual_seed(seed)
    kvecs = make_wavevector_grid(config["n_max"], box=config["box"]).to(device)
    sigmas = torch.tensor(config["sigmas"], device=device)
    num_outputs = sigmas.shape[0]

    torch.manual_seed(seed)
    model = build_model(encoder, num_outputs, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.0)

    with torch.no_grad():
        _, _, _, _, E_probe = generate_batch(
            config["eval_batches"] * config["batch_size"], n_atoms, kvecs.cpu(), sigmas.cpu(),
            box=config["box"], generator=torch.Generator().manual_seed(seed + 999),
        )
        target_var = E_probe.var(dim=0).to(device)
        target_std = target_var.sqrt().clamp(min=1e-12)
        target_mean = E_probe.mean(dim=0).to(device)
        model.decoder.net[-1].weight.zero_()
        model.decoder.net[-1].bias.copy_(target_mean)

    model.train()
    for step in range(config["train_steps"]):
        charges, positions, V, mask, E_LR = generate_batch(
            config["batch_size"], n_atoms, kvecs.cpu(), sigmas.cpu(), box=config["box"], generator=generator,
        )
        charges, positions, mask, V, E_LR = (
            charges.to(device), positions.to(device), mask.to(device), V.to(device), E_LR.to(device),
        )
        optimizer.zero_grad()
        pred = model(charges, positions, mask, V)
        loss = F.mse_loss((pred - target_mean) / target_std, (E_LR - target_mean) / target_std)
        loss.backward()
        optimizer.step()
        if config.get("log_every", 0) and (step % config["log_every"] == 0 or step == config["train_steps"] - 1):
            print(f"    [{encoder} N={n_atoms} seed={seed}] step={step} loss={loss.item():.4f}", flush=True)

    model.eval()
    total_sq_err = torch.zeros(num_outputs, device=device)
    total_n = 0
    ipr_sum, ipr_n = 0.0, 0
    master_attn = getattr(model, "master_attn", None)
    if master_attn is not None:
        master_attn.diag_enabled = True
    eval_generator = torch.Generator().manual_seed(seed + 12345)
    with torch.no_grad():
        for _ in range(config["eval_batches"]):
            charges, positions, V, mask, E_LR = generate_batch(
                config["batch_size"], n_atoms, kvecs.cpu(), sigmas.cpu(), box=config["box"], generator=eval_generator,
            )
            charges, positions, mask, V, E_LR = (
                charges.to(device), positions.to(device), mask.to(device), V.to(device), E_LR.to(device),
            )
            pred = model(charges, positions, mask, V)
            total_sq_err += ((pred - E_LR) ** 2).sum(dim=0)
            total_n += E_LR.shape[0]
            if master_attn is not None and master_attn._diag_ipr is not None:
                ipr_sum += float(master_attn._diag_ipr.sum())
                ipr_n += master_attn._diag_ipr.numel()
    if master_attn is not None:
        master_attn.diag_enabled = False
    mean_ipr = (ipr_sum / ipr_n) if ipr_n else None

    nmse_per_sigma = (total_sq_err / max(1, total_n)) / target_var.clamp(min=1e-12)
    nmse = nmse_per_sigma.mean().item()
    n_params = sum(p.numel() for p in model.parameters())
    return {
        "encoder": encoder, "n_atoms": n_atoms, "nmse": nmse,
        "nmse_per_sigma": nmse_per_sigma.detach().cpu().tolist(), "n_params": n_params,
        "mean_ipr": mean_ipr,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-values", type=int, nargs="+", default=[32, 64, 128, 256, 512])
    parser.add_argument("--encoders", type=str, nargs="+", default=["eqfsa", "rsa"])
    parser.add_argument("--num-master-nodes", type=int, default=4)
    parser.add_argument("--rsa-sigma", type=float, default=5.0)
    parser.add_argument("--n-max", type=int, default=2)
    parser.add_argument("--sigmas", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0, 16.0])
    parser.add_argument("--box", type=float, default=20.0)
    parser.add_argument("--feature-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-steps", type=int, default=2000)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-realizations", type=int, default=3)
    parser.add_argument("--log-every", type=int, default=0, help="Print training loss every N steps (0=silent)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="stage_1/experiments/neurreps/lr_sr/results_invariant_energy")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    config = vars(args)
    config["_device"] = detect_device(config["device"])
    print(f"Using device: {config['_device']}", flush=True)

    if config["smoke_test"]:
        config["n_values"] = [32, 64]
        config["train_steps"] = 50
        config["eval_batches"] = 2
        config["num_realizations"] = 1

    os.makedirs(config["outdir"], exist_ok=True)
    csv_path = os.path.join(config["outdir"], "results.csv")

    rows = []
    writer = None
    with open(csv_path, "w", newline="") as csv_file:
        sigmas = config["sigmas"]
        for n_atoms in config["n_values"]:
            for encoder in config["encoders"]:
                print(f"=== encoder={encoder} N={n_atoms} ===", flush=True)
                t0 = time.time()
                nmses, n_params, nmse_per_sigma_runs, iprs = [], None, [], []
                for i in range(config["num_realizations"]):
                    result = train_one(encoder, n_atoms, config, seed=config["seed"] + i * 7919)
                    nmses.append(result["nmse"])
                    nmse_per_sigma_runs.append(result["nmse_per_sigma"])
                    n_params = result["n_params"]
                    if result["mean_ipr"] is not None:
                        iprs.append(result["mean_ipr"])
                mean_nmse = sum(nmses) / len(nmses)
                std_nmse = (sum((v - mean_nmse) ** 2 for v in nmses) / max(1, len(nmses) - 1)) ** 0.5
                row = {
                    "encoder": encoder, "n_atoms": n_atoms, "nmse": mean_nmse, "nmse_std": std_nmse,
                    "n_params": n_params, "train_time_s": time.time() - t0,
                }
                if iprs:
                    mean_ipr = sum(iprs) / len(iprs)
                    std_ipr = (sum((v - mean_ipr) ** 2 for v in iprs) / max(1, len(iprs) - 1)) ** 0.5
                    row["master_node_ipr"] = mean_ipr
                    row["master_node_ipr_std"] = std_ipr
                    print(f"    master-node IPR={mean_ipr:.2f}+/-{std_ipr:.2f} (of N={n_atoms} real atoms)", flush=True)
                else:
                    row["master_node_ipr"] = ""
                    row["master_node_ipr_std"] = ""
                sigma_summary = []
                for sigma_idx, sigma_val in enumerate(sigmas):
                    vals = [run[sigma_idx] for run in nmse_per_sigma_runs]
                    m = sum(vals) / len(vals)
                    s = (sum((v - m) ** 2 for v in vals) / max(1, len(vals) - 1)) ** 0.5
                    row[f"nmse_sigma_{sigma_val:g}"] = m
                    row[f"nmse_sigma_{sigma_val:g}_std"] = s
                    sigma_summary.append(f"sigma={sigma_val:g}:{m:.4f}+/-{s:.4f}")
                print(f"  nmse={mean_nmse:.4f}+/-{std_nmse:.4f}  n_params={n_params}  ({row['train_time_s']:.1f}s)", flush=True)
                print(f"    per-sigma: {'  '.join(sigma_summary)}", flush=True)
                rows.append(row)
                if writer is None:
                    writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
                    writer.writeheader()
                writer.writerow(row)
                csv_file.flush()
                os.fsync(csv_file.fileno())
    print(f"Wrote {len(rows)} rows to {csv_path}")


if __name__ == "__main__":
    main()
