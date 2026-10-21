import argparse

import torch
from torch.profiler import profile, ProfilerActivity

from stage_1.train_spice import SPICEEnergyForceModel, compute_energy_and_forces, spice_loss
from stage_1.spice_dataset import _EDGE_FEAT_DIM
from utils.calc_props import _MOL_FEAT_DIM
from utils.se3_utils import RadialNetwork


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-atoms", type=int, default=21)
    parser.add_argument("--num-atom-types", type=int, default=23)
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--n-master-nodes", type=int, default=8)
    parser.add_argument("--edge-cutoff", type=float, default=5.0)
    parser.add_argument("--n-max", type=int, default=1)
    parser.add_argument("--lambda1", type=float, default=1.0)
    parser.add_argument("--lambda2", type=float, default=1000.0)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--profile-steps", type=int, default=10)
    parser.add_argument("--row-limit", type=int, default=30)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--use-long-range", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    print(f"Using device: {device}  use_long_range={args.use_long_range}", flush=True)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    torch.manual_seed(0)
    model = SPICEEnergyForceModel(
        radial_net=RadialNetwork, num_atom_types=args.num_atom_types, max_degree=2,
        num_layers=args.num_layers, feature_dim=args.feature_dim, hidden_dim=args.hidden_dim,
        edge_cutoff=args.edge_cutoff, n_max=args.n_max, n_master_nodes=args.n_master_nodes,
        backbone="lr_restructure", num_broadcast_heads=4, edge_feat_dim=_EDGE_FEAT_DIM,
        use_long_range=args.use_long_range,
    ).to(device)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{n_params:,} params", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2, fused=(device.type == "cuda"))

    g = torch.Generator().manual_seed(1)
    B, N = args.batch_size, args.n_atoms
    x = (torch.rand(B, N, 3, generator=g) * 10.0 - 5.0).to(device)
    types_ = torch.randint(0, args.num_atom_types, (B, N), generator=g).to(device)
    masses = torch.empty(B, N).uniform_(1.0, 16.0, generator=g).to(device)
    mask = torch.ones(B, N, dtype=torch.bool, device=device)
    V = (torch.eye(3).unsqueeze(0).repeat(B, 1, 1) * 30.0).to(device)
    mol_props = torch.randn(B, _MOL_FEAT_DIM, generator=g).to(device)
    edge_feats = torch.rand(B, N, N, _EDGE_FEAT_DIM, generator=g).to(device)
    U_target = torch.randn(B, generator=g).to(device)
    F_target = torch.randn(B, N, 3, generator=g).to(device)

    def step():
        opt.zero_grad(set_to_none=True)
        U, F = compute_energy_and_forces(
            model, x, types_, masses, mask, V, mol_props, edge_feats, force_mode="conservative",
        )
        loss, _, _ = spice_loss(U, U_target, F, F_target, mask, args.lambda1, args.lambda2)
        loss.backward()
        opt.step()
        return loss

    print(f"Warming up ({args.warmup_steps} steps)...", flush=True)
    for _ in range(args.warmup_steps):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    print(f"Profiling ({args.profile_steps} steps)...", flush=True)
    with profile(activities=activities, record_shapes=True) as prof:
        for _ in range(args.profile_steps):
            step()
        if device.type == "cuda":
            torch.cuda.synchronize()

    print("\n" + "=" * 100)
    print("TOP OPS BY SELF_CUDA_TIME_TOTAL (time-bound hotspots)")
    print("=" * 100)
    sort_key = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_key, row_limit=args.row_limit))

    total_events = sum(1 for _ in prof.key_averages())
    print(f"\n-> wall clock: run `time` around this command for total elapsed seconds over {args.profile_steps} steps")


if __name__ == "__main__":
    main()
