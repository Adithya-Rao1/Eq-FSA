from __future__ import annotations

import argparse
import csv
import math
import os
import time

import torch
import torch.nn.functional as F

from stage_1.experiments.neurreps.lr_sr.invariant_energy_data import generate_batch, make_wavevector_grid
from stage_1.experiments.neurreps.lr_sr.invariant_energy_model import EqFSAEncoder, RSAEncoder

def linear_probe_r2(features: torch.Tensor, targets: torch.Tensor, train_frac: float = 0.8) -> torch.Tensor:
    n = features.shape[0]
    n_train = int(n * train_frac)
    perm = torch.randperm(n, device=features.device)
    train_idx, test_idx = perm[:n_train], perm[n_train:]
    X_train = torch.cat([features[train_idx], torch.ones(n_train, 1, device=features.device)], dim=-1)
    X_test = torch.cat([features[test_idx], torch.ones(n - n_train, 1, device=features.device)], dim=-1)
    y_train, y_test = targets[train_idx], targets[test_idx]
    coeffs = torch.linalg.lstsq(X_train, y_train).solution
    pred_test = X_test @ coeffs
    ss_res = ((y_test - pred_test) ** 2).sum(dim=0)
    ss_tot = ((y_test - y_test.mean(dim=0, keepdim=True)) ** 2).sum(dim=0).clamp(min=1e-12)
    return 1.0 - ss_res / ss_tot


def final_linear(decoder: torch.nn.Module) -> torch.nn.Linear:
    return decoder.net[-1] if isinstance(decoder.net, torch.nn.Sequential) else decoder.net


def build_lr_scheduler(optimizer: torch.optim.Optimizer, train_steps: int, warmup_frac: float):
    warmup_steps = max(1, int(train_steps * warmup_frac))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, train_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def detect_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def build_optimizer(model: torch.nn.Module, lr: float, qk_gamma_wd: float) -> torch.optim.Optimizer:
    decay_params, other_params = [], []
    for name, p in model.named_parameters():
        if (name.endswith("w_vq_norm.gamma") or name.endswith("w_vk_norm.gamma")
                or name.startswith("master_attn.W_QK.")
                or name.endswith("s_ri_norm.weight")):
            decay_params.append(p)
        else:
            other_params.append(p)
    return torch.optim.AdamW([
        {"params": other_params, "weight_decay": 0.0},
        {"params": decay_params, "weight_decay": qk_gamma_wd},
    ], lr=lr)


def build_model(encoder: str, num_outputs: int, config: dict) -> torch.nn.Module:
    if encoder == "eqfsa":
        return EqFSAEncoder(
            num_master_nodes=config["num_master_nodes"], feature_dim=config["feature_dim"],
            hidden_dim=config["hidden_dim"], n_max=config["n_max"], num_outputs=num_outputs,
            use_scale_feat=config.get("eqfsa_scale_feat", True),
            use_per_node_com=config.get("eqfsa_per_node_com", True),
            rescale_init_scale=config.get("eqfsa_rescale_init", 500.0),
            alpha_floor=config.get("eqfsa_alpha_floor", 1e-4),
            use_input_dependent_g_a=config.get("eqfsa_input_dependent_g_a", True),
            use_node_conditioned_k=config.get("eqfsa_node_conditioned_k", True),
            use_linear_decoder=config.get("eqfsa_linear_decoder", False),
            use_normalize_z_by_n_real=config.get("eqfsa_normalize_z_by_n_real", True),
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
    optimizer = build_optimizer(model, config["lr"], config.get("qk_gamma_wd", 0.0))
    scheduler = None
    if config.get("lr_schedule", "constant") == "warmup_cosine":
        scheduler = build_lr_scheduler(optimizer, config["train_steps"], config.get("lr_warmup_frac", 0.05))

    with torch.no_grad():
        _, _, _, _, E_probe = generate_batch(
            config["eval_batches"] * config["batch_size"], n_atoms, kvecs.cpu(), sigmas.cpu(),
            box=config["box"], generator=torch.Generator().manual_seed(seed + 999),
        )
        target_var = E_probe.var(dim=0).to(device)
        target_std = target_var.sqrt().clamp(min=1e-12)
        target_mean = E_probe.mean(dim=0).to(device)
        final_linear(model.decoder).weight.zero_()
        final_linear(model.decoder).bias.copy_(target_mean)
        if config.get("log_every", 0):
            skew = (((E_probe.to(device) - target_mean) / target_std) ** 3).mean(dim=0)
            for i, s in enumerate(config["sigmas"]):
                print(f"    [{encoder} N={n_atoms} seed={seed}] sigma={s:g} target_mean="
                      f"{target_mean[i].item():.6g} target_std={target_std[i].item():.6g} "
                      f"skew={skew[i].item():.4g}", flush=True)

    model.train()
    master_attn = getattr(model, "master_attn", None)
    freeze_step = int(config["train_steps"] * config.get("freeze_encoder_frac", 0.0))
    encoder_frozen = False
    for step in range(config["train_steps"]):
        if freeze_step and step == freeze_step and not encoder_frozen:
            for name, p in model.named_parameters():
                if not name.startswith("decoder."):
                    p.requires_grad_(False)
            encoder_frozen = True
            if config.get("log_every", 0):
                print(f"    [{encoder} N={n_atoms} seed={seed}] step={step}: encoder frozen, "
                      f"decoder-only training for the remaining "
                      f"{config['train_steps'] - freeze_step} steps", flush=True)
        charges, positions, V, mask, E_LR = generate_batch(
            config["batch_size"], n_atoms, kvecs.cpu(), sigmas.cpu(), box=config["box"], generator=generator,
        )
        charges, positions, mask, V, E_LR = (
            charges.to(device), positions.to(device), mask.to(device), V.to(device), E_LR.to(device),
        )
        is_log_step = config.get("log_every", 0) and (
            step % config["log_every"] == 0 or step == config["train_steps"] - 1
        )
        if is_log_step and master_attn is not None:
            master_attn.diag_enabled = True
        optimizer.zero_grad()
        pred = model(charges, positions, mask, V)
        loss = F.mse_loss((pred - target_mean) / target_std, (E_LR - target_mean) / target_std)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip_norm"])
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if is_log_step:
            extra = f" lr={optimizer.param_groups[0]['lr']:.2e}" if scheduler is not None else ""
            if master_attn is not None and master_attn._diag_Z_min is not None:
                alpha_val = float(F.softplus(master_attn.alpha) + master_attn.alpha_floor)
                extra += (f" Z_min={master_attn._diag_Z_min:.4g} "
                          f"damping_sum_min={master_attn._diag_damping_sum_min:.4g} alpha={alpha_val:.4g}")
                master_attn.diag_enabled = False
            with torch.no_grad():
                per_sigma_nmse = (((pred - E_LR) ** 2).mean(dim=0) / target_var.clamp(min=1e-12))
                per_sigma_str = "  ".join(
                    f"s{s:g}={per_sigma_nmse[i].item():.4f}" for i, s in enumerate(config["sigmas"])
                )
            print(f"    [{encoder} N={n_atoms} seed={seed}] step={step} loss={loss.item():.4f} "
                  f"grad_norm={float(grad_norm):.4f}{extra}  [{per_sigma_str}]", flush=True)

    if config.get("log_every", 0):
        final_w = final_linear(model.decoder).weight.detach()  # started at exactly 0
        per_col_norm = final_w.norm(dim=1)
        col_str = "  ".join(f"s{s:g}={per_col_norm[i].item():.4f}" for i, s in enumerate(config["sigmas"]))
        print(f"    [{encoder} N={n_atoms} seed={seed}] decoder.net[-1].weight per-output-column norm "
              f"(started at exactly 0.0 for all): [{col_str}]", flush=True)

    model.eval()

    if config.get("log_every", 0):
        probe_charges, probe_positions, probe_V, probe_mask, probe_E_LR = generate_batch(
            config["eval_batches"] * config["batch_size"], n_atoms, kvecs.cpu(), sigmas.cpu(),
            box=config["box"], generator=torch.Generator().manual_seed(seed + 54321),
        )
        probe_charges, probe_positions, probe_mask, probe_V, probe_E_LR = (
            probe_charges.to(device), probe_positions.to(device), probe_mask.to(device),
            probe_V.to(device), probe_E_LR.to(device),
        )
        with torch.no_grad():
            _, probe_features = model(probe_charges, probe_positions, probe_mask, probe_V, return_features=True)
            r2 = linear_probe_r2(probe_features, probe_E_LR)
        r2_str = "  ".join(f"s{s:g}={r2[i].item():.4f}" for i, s in enumerate(config["sigmas"]))
        print(f"    [{encoder} N={n_atoms} seed={seed}] linear-probe R^2 of decoder_in vs target "
              f"(held-out split, n={probe_features.shape[0]}): [{r2_str}]", flush=True)

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
    parser.add_argument("--lr-schedule", type=str, default="constant", choices=["constant", "warmup_cosine"])
    parser.add_argument("--lr-warmup-frac", type=float, default=0.05,
                         help="Fraction of train_steps used for linear LR warmup (only used when "
                              "--lr-schedule warmup_cosine).")
    parser.add_argument("--freeze-encoder-frac", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--qk-gamma-wd", type=float, default=0.0)
    parser.add_argument("--no-eqfsa-scale-feat", dest="eqfsa_scale_feat", action="store_false")
    parser.add_argument("--no-eqfsa-per-node-com", dest="eqfsa_per_node_com", action="store_false")
    parser.add_argument("--no-eqfsa-input-dependent-g-a", dest="eqfsa_input_dependent_g_a", action="store_false")
    parser.add_argument("--no-eqfsa-node-conditioned-k", dest="eqfsa_node_conditioned_k", action="store_false")
    parser.add_argument("--no-eqfsa-normalize-z-by-n-real", dest="eqfsa_normalize_z_by_n_real", action="store_false",)
    parser.add_argument("--eqfsa-linear-decoder", action="store_true")
    parser.add_argument("--eqfsa-alpha-floor", type=float, default=1e-4)
    parser.add_argument("--eqfsa-rescale-init", type=float, default=500.0)
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
