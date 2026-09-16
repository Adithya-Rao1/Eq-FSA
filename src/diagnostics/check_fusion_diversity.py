import argparse

import torch
import torch.nn.functional as F

from stage_1.experiments.neurreps.lr_sr.invariant_energy_model import EqFSAEncoder
from stage_1.experiments.neurreps.lr_sr.invariant_energy_data import generate_batch, make_wavevector_grid
from stage_1.experiments.neurreps.lr_sr.train_invariant_energy import build_lr_scheduler
from utils.calc_props import mol_props_from_atom_counts


def cross_atom_diversity(t: torch.Tensor, mask: torch.Tensor) -> float:
    m = mask.float().unsqueeze(-1)
    n_real = mask.sum(dim=1, keepdim=True).clamp(min=1).float()
    mean = (t * m).sum(dim=1, keepdim=True) / n_real.unsqueeze(-1)
    dev = ((t - mean) * m).norm(dim=-1)
    mag = (t * m).norm(dim=-1)
    return (dev.sum(dim=1) / n_real.squeeze(-1) / mag.sum(dim=1).clamp(min=1e-8) * n_real.squeeze(-1)).mean().item()


def run(sigma_val, label, use_pool_fix, device, n_atoms, num_master, feature_dim, hidden_dim,
        n_max, box, steps, batch, lr, clip_norm, warmup_frac, seed):
    torch.manual_seed(seed)
    model = EqFSAEncoder(
        num_master_nodes=num_master, feature_dim=feature_dim, hidden_dim=hidden_dim, n_max=n_max,
        num_outputs=1, num_layers=4, edge_cutoff=5.0,
        use_direction_aware_pool_agg=use_pool_fix, pool_expand_learnable_gain=use_pool_fix,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = build_lr_scheduler(opt, steps, warmup_frac)
    kvecs = make_wavevector_grid(n_max, box=box).to(device)
    sigmas = torch.tensor([sigma_val], device=device)
    gen = torch.Generator().manual_seed(seed)

    with torch.no_grad():
        _, _, _, _, E_probe = generate_batch(
            320, n_atoms, kvecs.cpu(), sigmas.cpu(), box=box,
            generator=torch.Generator().manual_seed(seed + 999),
        )
        target_mean = E_probe.mean(dim=0).to(device)
        target_std = E_probe.std(dim=0).clamp(min=1e-12).to(device)
        model.decoder.net[-1].weight.zero_()
        model.decoder.net[-1].bias.copy_(target_mean)

    loss_hist = []
    for step in range(steps):
        charges, positions, V, mask, E_LR = generate_batch(
            batch, n_atoms, kvecs.cpu(), sigmas.cpu(), box=box, generator=gen,
        )
        charges, positions, V, mask, E_LR = (
            charges.to(device), positions.to(device), V.to(device), mask.to(device), E_LR.to(device),
        )
        opt.zero_grad()
        pred = model(charges, positions, mask, V)
        loss = F.mse_loss((pred - target_mean) / target_std, (E_LR - target_mean) / target_std)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        opt.step()
        scheduler.step()
        loss_hist.append(loss.item())
    avg_last10 = sum(loss_hist[-10:]) / 10
    
    model.eval()
    with torch.no_grad():
        charges, positions, V, mask, E_LR = generate_batch(
            batch, n_atoms, kvecs.cpu(), sigmas.cpu(), box=box, generator=gen,
        )
        charges, positions, V, mask = charges.to(device), positions.to(device), V.to(device), mask.to(device)
        B, N = charges.shape
        input_features = model.charge_embed(charges).squeeze(-1)
        node_types = torch.zeros(B, N, dtype=torch.long, device=device)
        masses = mask.float()
        n_h = ((charges > 0) & mask).sum(dim=1)
        n_o = ((charges < 0) & mask).sum(dim=1)
        mol_props = mol_props_from_atom_counts(n_h, n_o).to(device=device, dtype=positions.dtype)
        edge_feats = torch.zeros(B, N, N, 1, device=device, dtype=positions.dtype)
        out = model.energy_model(
            positions, node_types, masses, mask, V, mol_props, edge_feats,
            input_features=input_features, collect_scale_feats=True,
        )
        f, scale_feats = out
        u_vec = model.norm_readout1(f[1])
        u_ten = model.norm_readout2(f[2])
        full_vec = f[0].squeeze(-1) + u_vec + u_ten
        div = cross_atom_diversity(full_vec, mask)
    print(f"[{label} seed={seed}] avg_loss_last10={avg_last10:.4f} cross-atom relative diversity "
          f"of full_vec (fusion output, pre-pool): {div * 100:.4f}%", flush=True)
    return avg_last10, div


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n-atoms", type=int, default=64)
    parser.add_argument("--num-master-nodes", type=int, default=4)
    parser.add_argument("--feature-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--n-max", type=int, default=1)
    parser.add_argument("--box", type=float, default=20.0)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--clip-norm", type=float, default=10.0)
    parser.add_argument("--warmup-frac", type=float, default=0.05)
    parser.add_argument("--sigma", type=float, default=16.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    device = torch.device(args.device) if args.device else (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    print(f"Using device: {device}", flush=True)

    for use_pool_fix, label in [(False, f"sigma={args.sigma:g}, WITHOUT pooling fix"),
                                 (True, f"sigma={args.sigma:g}, WITH pooling fix")]:
        results = [
            run(args.sigma, label, use_pool_fix, device, args.n_atoms, args.num_master_nodes,
                args.feature_dim, args.hidden_dim, args.n_max, args.box, args.steps, args.batch,
                args.lr, args.clip_norm, args.warmup_frac, seed)
            for seed in args.seeds
        ]
        avg_loss = sum(r[0] for r in results) / len(results)
        avg_div = sum(r[1] for r in results) / len(results)
        print(f"  == {label}: mean over {len(args.seeds)} seeds -- "
              f"avg_loss_last10={avg_loss:.4f} avg_diversity={avg_div * 100:.4f}% ==\n", flush=True)


if __name__ == "__main__":
    main()
