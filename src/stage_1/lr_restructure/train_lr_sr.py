import argparse
import csv
import os

import torch
import torch.nn.functional as F

from stage_1.experiments.neurreps.lr_sr.invariant_energy_data import generate_batch, make_wavevector_grid
from stage_1.experiments.neurreps.lr_sr.train_invariant_energy import build_lr_scheduler
from stage_1.lr_restructure.lr_sr_encoder import LRRestructureEncoder


def train_one(n_atoms: int, sigma: float, config: dict, device: torch.device) -> float:
    torch.manual_seed(config["seed"])
    model = LRRestructureEncoder(
        num_master_nodes=config["num_master_nodes"], feature_dim=config["feature_dim"],
        hidden_dim=config["hidden_dim"], n_max=config["n_max"], num_outputs=1,
        num_layers=config["num_layers"], edge_cutoff=config["edge_cutoff"], sigma=sigma,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config["lr"], fused=(device.type == "cuda"))
    scheduler = (
        build_lr_scheduler(opt, config["train_steps"], config["lr_warmup_frac"])
        if config["lr_schedule"] == "warmup_cosine" else None
    )
    train_model = model
    if config.get("torch_compile"):
        torch._dynamo.config.capture_dynamic_output_shape_ops = True
        compile_mode = config.get("compile_mode")
        if compile_mode == "reduce-overhead":
            train_model = torch.compile(model, mode=compile_mode)
        else:
            train_model = torch.compile(model, dynamic=True, **({"mode": compile_mode} if compile_mode else {}))
    kvecs = make_wavevector_grid(config["n_max"], box=config["box"]).to(device)
    sigmas = torch.tensor([sigma], device=device)
    gen = torch.Generator().manual_seed(config["seed"])

    with torch.no_grad():
        _, _, _, _, E_probe = generate_batch(
            320, n_atoms, kvecs.cpu(), sigmas.cpu(), box=config["box"],
            generator=torch.Generator().manual_seed(config["seed"] + 999),
        )
        target_mean = E_probe.mean(dim=0).to(device)
        target_std = E_probe.std(dim=0).clamp(min=1e-12).to(device)
        model.decoder.net[-1].weight.zero_()
        model.decoder.net[-1].bias.copy_(target_mean)

    for step in range(config["train_steps"]):
        charges, positions, V, mask, E_LR = generate_batch(
            config["batch_size"], n_atoms, kvecs.cpu(), sigmas.cpu(), box=config["box"], generator=gen,
        )
        charges, positions, V, mask, E_LR = (
            charges.to(device), positions.to(device), V.to(device), mask.to(device), E_LR.to(device),
        )
        opt.zero_grad()
        pred = train_model(charges, positions, mask, V)
        loss = F.mse_loss((pred - target_mean) / target_std, (E_LR - target_mean) / target_std)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip_norm"])
        opt.step()
        if scheduler is not None:
            scheduler.step()
        if config["log_every"] and step % config["log_every"] == 0:
            print(f"[n={n_atoms} sigma={sigma}] step={step} loss={loss.item():.4f}", flush=True)

    model.eval()
    eval_nmse = []
    with torch.no_grad():
        for _ in range(config["eval_batches"]):
            charges, positions, V, mask, E_LR = generate_batch(
                config["batch_size"], n_atoms, kvecs.cpu(), sigmas.cpu(), box=config["box"], generator=gen,
            )
            charges, positions, V, mask, E_LR = (
                charges.to(device), positions.to(device), V.to(device), mask.to(device), E_LR.to(device),
            )
            pred = train_model(charges, positions, mask, V)
            nmse = F.mse_loss(pred, E_LR) / E_LR.var().clamp(min=1e-12)
            eval_nmse.append(nmse.item())
    return sum(eval_nmse) / len(eval_nmse)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-values", type=int, nargs="+", default=[32, 64, 128, 256, 512])
    parser.add_argument("--sigma", type=float, required=True)
    parser.add_argument("--n-max", type=int, default=2)
    parser.add_argument("--box", type=float, default=20.0)
    parser.add_argument("--feature-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-master-nodes", type=int, default=4)
    parser.add_argument("--edge-cutoff", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-steps", type=int, default=2000)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-schedule", type=str, default="constant", choices=["constant", "warmup_cosine"])
    parser.add_argument("--lr-warmup-frac", type=float, default=0.05)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--num-realizations", type=int, default=3)
    parser.add_argument("--log-every", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="stage_1/lr_restructure/results_lr_sr")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--compile-mode", type=str, default=None,choices=[None, "default", "reduce-overhead", "max-autotune"],)
    args = parser.parse_args()

    if args.smoke_test:
        args.n_values = [8, 16]
        args.train_steps = 5
        args.eval_batches = 1
        args.num_realizations = 1
        args.batch_size = 4
        args.feature_dim = 8
        args.hidden_dim = 16
        args.num_layers = 2
        args.num_master_nodes = 2
        args.n_max = 1

    device = torch.device(args.device) if args.device else (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    print(f"Using device: {device}", flush=True)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, "results.csv")

    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["n_atoms", "sigma", "realization", "seed", "nmse"])
        for n_atoms in args.n_values:
            for r in range(args.num_realizations):
                seed = args.seed + r
                config = dict(vars(args))
                config["seed"] = seed
                nmse = train_one(n_atoms, args.sigma, config, device)
                writer.writerow([n_atoms, args.sigma, r, seed, nmse])
                fcsv.flush()
                print(
                    f"n={n_atoms} sigma={args.sigma} realization={r} seed={seed} nmse={nmse:.4f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
