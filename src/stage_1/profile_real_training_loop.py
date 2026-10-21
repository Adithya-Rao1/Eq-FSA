import argparse

import torch
from torch.profiler import profile, ProfilerActivity

from stage_1.train_md17 import build_dataloaders
from stage_1.train_spice import SPICEEnergyForceModel, SPICETrainer, detect_device
from stage_1.spice_dataset import _EDGE_FEAT_DIM
from utils.se3_utils import RadialNetwork


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molecule", type=str, default="aspirin")
    parser.add_argument("--split-idx", type=int, default=0)
    parser.add_argument("--data-root", type=str, default="data/rmd17")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--box-padding", type=float, default=12.0)
    parser.add_argument("--edge-cutoff", type=float, default=5.0)
    parser.add_argument("--num-atom-types", type=int, default=23)
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--n-master-nodes", type=int, default=8)
    parser.add_argument("--n-max", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--lambda1", type=float, default=1.0)
    parser.add_argument("--lambda2", type=float, default=1000.0)
    parser.add_argument("--use-long-range", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--n-warmup-epochs", type=int, default=1, help="Real epochs run before profiling starts (lets any one-time setup cost happen outside the trace)")
    parser.add_argument("--n-epochs-to-profile", type=int, default=4, help="Real epochs captured inside the profiler -- use >=2 so both a 'fast' and a 'slow' epoch are likely included")
    parser.add_argument("--row-limit", type=int, default=30)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = detect_device(args.device)
    print(f"Using device: {device}  use_long_range={args.use_long_range}", flush=True)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    config = {
        "molecule": args.molecule, "split_idx": args.split_idx, "data_root": args.data_root,
        "batch_size": args.batch_size, "num_workers": args.num_workers,
        "box_padding": args.box_padding, "edge_cutoff": args.edge_cutoff,
        "prefetch_factor": 2,
    }
    train_loader, val_loader, energy_mean, energy_std, force_rms = build_dataloaders(config, device=device)
    print(f"Real dataset: {len(train_loader.dataset)} train configs, {len(train_loader)} batches/epoch", flush=True)

    torch.manual_seed(0)

    full_config = {
        "ckpt_dir": "/tmp/profile_real_training_loop_ckpt", "csv_path": "/tmp/profile_real_training_loop_loss.csv",
        "lr": args.lr, "warmup_epochs": 5, "min_lr": 1e-6, "plateau_factor": 0.9, "plateau_patience": 50,
        "weight_decay": 1e-4, "lambda1": args.lambda1, "lambda2": args.lambda2,
        "n_epochs": args.n_warmup_epochs + args.n_epochs_to_profile, "grad_clip_norm": 1.0,
        "force_mode": "conservative", "torch_compile": False, "log_interval": 100000,
        "diagnostics": False,
        "energy_mean": energy_mean, "energy_std": energy_std, "force_rms": force_rms,
    }

    model = SPICEEnergyForceModel(
        radial_net=RadialNetwork, num_atom_types=args.num_atom_types, max_degree=2,
        num_layers=args.num_layers, feature_dim=args.feature_dim, hidden_dim=args.hidden_dim,
        edge_cutoff=args.edge_cutoff, n_max=args.n_max, n_master_nodes=args.n_master_nodes,
        backbone="lr_restructure", num_broadcast_heads=4, edge_feat_dim=_EDGE_FEAT_DIM,
        use_long_range=args.use_long_range,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{n_params:,} params", flush=True)

    trainer = SPICETrainer(model, train_loader, val_loader, full_config, device)

    print(f"Running {args.n_warmup_epochs} warmup epoch(s) (not profiled, not timed)...", flush=True)
    for epoch in range(args.n_warmup_epochs):
        trainer.train_epoch(epoch)

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    print(f"Profiling {args.n_epochs_to_profile} real epoch(s)...", flush=True)
    with profile(activities=activities, record_shapes=True) as prof:
        for epoch in range(args.n_warmup_epochs, args.n_warmup_epochs + args.n_epochs_to_profile):
            trainer.train_epoch(epoch)
        if device.type == "cuda":
            torch.cuda.synchronize()

    print("\n" + "=" * 100)
    print("TOP OPS BY SELF_CUDA_TIME_TOTAL -- real epochs, real DataLoader, real CSV logging")
    print("=" * 100)
    sort_key = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_key, row_limit=args.row_limit))

    trainer.close()


if __name__ == "__main__":
    main()
