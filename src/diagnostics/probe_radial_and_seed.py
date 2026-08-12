import argparse
import sys

import torch

from utils.se3_utils import RadialNetwork
from stage_1.model import seed_equivariant_features, masked_weighted_com
from utils.pw_bases import minimum_image

SEED_NUM_BASIS = 1

PRODUCTION_NUM_BASIS_VALUES = (1, 3, 5, 8)

def probe_radial_basis(device: torch.device, cutoff_radius: float = 5.0, hidden_dim: int = 64):
    print(f"\n=== Gaussian-RBF RadialNetwork basis sweep on {device} ===")
    for num_basis in PRODUCTION_NUM_BASIS_VALUES:
        torch.manual_seed(0)
        net = RadialNetwork(num_basis=num_basis, hidden_dim=hidden_dim, cutoff_radius=cutoff_radius).to(device)
        net.eval()
        width = cutoff_radius / max(num_basis - 1, 1)
        print(f"\n  --- num_basis={num_basis} (Gaussian centres on [0,{cutoff_radius}], width={width:.4f}) ---")

        r_core = torch.linspace(0.0, cutoff_radius, 200_000, device=device)
        r_extra = torch.tensor([0.0, 1e-6, cutoff_radius - 1e-6, cutoff_radius,
                                 cutoff_radius + 1e-3, cutoff_radius * 2], device=device)
        r = torch.cat([r_core, r_extra]).view(-1, 1)

        with torch.no_grad():
            basis = net._basis(r)
            out = net(r)

        for name, t in [("basis (Gaussian x envelope, pre-Linear)", basis), ("net(r) (final radial scalar)", out)]:
            finite = torch.isfinite(t)
            n_bad = (~finite).sum().item()
            if n_bad:
                bad_r = r[(~finite).any(dim=-1)]
                print(f"    [{name}] {n_bad} NON-FINITE entries! r range of bad entries: "
                      f"[{bad_r.min().item():.6f}, {bad_r.max().item():.6f}]")
            for col in range(t.shape[-1]):
                col_abs = t[:, col].abs()
                mi = col_abs.argmax().item()
                print(f"    [{name}] col={col}: max|.|={col_abs.max().item():.6g} at r={r[mi,0].item():.6f}")

        r_scaled = (r_extra / cutoff_radius).clamp(max=1.0)
        envelope = 0.5 * (torch.cos(torch.pi * r_scaled) + 1.0)
        print(f"    envelope at r={r_extra.tolist()}: {envelope.tolist()}")

    return net, r


def probe_seed_pathway(device: torch.device, trials: int = 500, n_max_atoms: int = 40, cutoff_radius: float = 5.0,
                        hidden_dim: int = 64, feature_dim: int = 32, seed_num_basis: int = 1,
                        dtype=torch.float32):
    print(f"\n=== seed_equivariant_features stress test on {device} ({dtype}, seed_num_basis={seed_num_basis}) ===")
    torch.manual_seed(1)
    seed_radial_net = RadialNetwork(
        num_basis=seed_num_basis, hidden_dim=hidden_dim, cutoff_radius=cutoff_radius,
    ).to(device).to(dtype)

    worst = {"max_abs": -1.0, "trial": None}
    n_nonfinite_trials = 0

    for trial in range(trials):
        g = torch.Generator().manual_seed(trial)
        N = torch.randint(3, n_max_atoms + 1, (1,), generator=g).item()
        B = 1

        steps = torch.randn(B, N, 3, generator=g) * 0.15 + torch.randn(B, 1, 3, generator=g).sign() * 1.4
        x = torch.cumsum(steps, dim=1).to(device=device, dtype=dtype)
        masses = torch.empty(B, N).uniform_(1.0, 127.0, generator=g).to(device=device, dtype=dtype)  
        mask = torch.ones(B, N, dtype=torch.bool, device=device)
        V = (torch.eye(3, dtype=dtype) * 20.0).unsqueeze(0).to(device)
        combined_scalar = torch.randn(B, N, feature_dim, generator=g).to(device=device, dtype=dtype)

        seeded = seed_equivariant_features(combined_scalar, seed_radial_net, x, masses, mask, V)
        f1, f2 = seeded[1], seeded[2]

        finite = torch.isfinite(f1).all() and torch.isfinite(f2).all()
        if not finite:
            n_nonfinite_trials += 1
            print(f"  trial {trial}: NON-FINITE f1/f2! N={N}")
            continue

        max_abs = max(f1.abs().max().item(), f2.abs().max().item())
        if max_abs > worst["max_abs"]:
            x_wrapped = minimum_image(x, V)
            com, _ = masked_weighted_com(x_wrapped, masses, mask)
            r_true = (x_wrapped - com.unsqueeze(1)).norm(dim=-1)
            worst = {
                "max_abs": max_abs, "trial": trial, "N": N,
                "r_min": r_true.min().item(), "r_max": r_true.max().item(),
            }

    print(f"  {n_nonfinite_trials}/{trials} trials produced non-finite f1/f2")
    print(f"  worst finite trial: max|f1 or f2|={worst['max_abs']:.6g}  "
          f"(trial={worst['trial']}, N={worst.get('N')}, "
          f"r_true range=[{worst.get('r_min'):.4f}, {worst.get('r_max'):.4f}])")
    return worst


def probe_min_abs_radial_padding_bug(
    device: torch.device, trials: int = 200, min_atoms: int = 3, max_atoms: int = 15,
    cutoff_radius: float = 5.0, hidden_dim: int = 64, feature_dim: int = 8,
    com_shift_scale: float = 10.0, dtype=torch.float32,
):
    print(f"\n=== min_abs_radial padding-bug repro on {device} ({dtype}) ===")
    torch.manual_seed(2)
    seed_radial_net = RadialNetwork(
        num_basis=SEED_NUM_BASIS, hidden_dim=hidden_dim, cutoff_radius=cutoff_radius,
    ).to(device).to(dtype)
    seed_radial_net.eval()

    n_batches_zero_unmasked = 0
    n_batches_padded_exceeds_cutoff = 0
    n_batches_with_padding = 0
    worst_gap = {"gap": -1.0, "trial": None}

    with torch.no_grad():
        for trial in range(trials):
            g = torch.Generator().manual_seed(trial)
            B = 4
            n_atoms = torch.randint(min_atoms, max_atoms + 1, (B,), generator=g).tolist()
            N_max = max(n_atoms)

            x = torch.zeros(B, N_max, 3, dtype=dtype)
            mask = torch.zeros(B, N_max, dtype=torch.bool)
            masses = torch.zeros(B, N_max, dtype=dtype)
            for b, n in enumerate(n_atoms):
                shift = torch.randn(3, generator=g)
                shift_mag = torch.empty(1).uniform_(0.0, com_shift_scale, generator=g)
                shift = shift / shift.norm().clamp(min=1e-8) * shift_mag
                local = torch.randn(n, 3, generator=g) * 0.6
                x[b, :n] = (local + shift).to(dtype)
                mask[b, :n] = True
                masses[b, :n] = torch.empty(n).uniform_(1.0, 127.0, generator=g).to(dtype)
            # padded slots (x=0, mask=False) left as zero-initialized, matching the real collate pipeline.

            V = (torch.eye(3, dtype=dtype) * (2 * com_shift_scale + 20.0)).unsqueeze(0).expand(B, 3, 3).to(device)
            x, mask, masses = x.to(device), mask.to(device), masses.to(device)
            combined_scalar = torch.randn(B, N_max, feature_dim, generator=g).to(device=device, dtype=dtype)

            diag_sink: dict = {}
            seed_equivariant_features(combined_scalar, seed_radial_net, x, masses, mask, V, diag_sink=diag_sink)

            x_wrapped = minimum_image(x, V)
            center, _ = masked_weighted_com(x_wrapped, masses, mask)
            r_true = (x_wrapped - center.unsqueeze(1)).norm(dim=-1)
            radial = seed_radial_net(r_true.reshape(-1, 1)).view(B, N_max)

            pad = ~mask
            has_padding = bool(pad.any())
            if has_padding:
                n_batches_with_padding += 1
                com_norm = center.norm(dim=-1)
                for b in range(B):
                    if pad[b].any():
                        pad_r = r_true[b, pad[b]]
                        assert torch.allclose(pad_r, com_norm[b].expand_as(pad_r), atol=1e-4), (
                            f"trial {trial} batch {b}: padded r_true != ||center|| -- hypothesis falsified"
                        )
                if bool((com_norm[pad.any(dim=1)] >= cutoff_radius).any()):
                    n_batches_padded_exceeds_cutoff += 1

            unmasked_min = radial.abs().min().item()
            masked_min = radial.abs().masked_fill(pad, float("inf")).min().item()
            if unmasked_min == 0.0 or unmasked_min < 1e-6:
                n_batches_zero_unmasked += 1
            gap = masked_min - unmasked_min
            if gap > worst_gap["gap"]:
                worst_gap = {"gap": gap, "trial": trial, "unmasked_min": unmasked_min, "masked_min": masked_min}

            reported = diag_sink.get("min_abs_radial")
            assert reported is not None and abs(reported - unmasked_min) < 1e-6, (
                f"trial {trial}: diag_sink min_abs_radial={reported} != recomputed {unmasked_min}"
            )

    print(f"  trials={trials}  batches_with_padding={n_batches_with_padding}/{trials}")
    print(f"  batches where a padded atom's COM-distance >= cutoff_radius={cutoff_radius}: "
          f"{n_batches_padded_exceeds_cutoff}/{n_batches_with_padding}")
    print(f"  trials where the (buggy, unmasked) min_abs_radial reads ~0.0: "
          f"{n_batches_zero_unmasked}/{trials}")
    print(f"  worst masked-vs-unmasked gap: trial={worst_gap['trial']}  "
          f"unmasked_min={worst_gap.get('unmasked_min'):.6g}  masked_min={worst_gap.get('masked_min'):.6g}")
    print("  Padded-slot r_true == ||center|| confirmed exactly (to float tolerance) in every "
          "batch with padding -- see assertions above (would have raised otherwise).")
    print("  Fix candidate: reduce over `radial.masked_fill(~node_mask, float('inf')).abs().min()` "
          "(or equivalent) before writing diag_sink['min_abs_radial'], so padding never contributes.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trials", type=int, default=500)
    p.add_argument("--n-max-atoms", type=int, default=40)
    p.add_argument("--seed-num-basis", type=int, default=1,
                    help="matches model.py's self.seed_radial_net = radial_net(num_basis=1, ...)")
    p.add_argument("--skip-min-abs-radial-probe", action="store_true",
                    help="Skip the min_abs_radial padding-bug repro (probe_min_abs_radial_padding_bug)")
    p.add_argument("--min-abs-radial-trials", type=int, default=200)
    args = p.parse_args()

    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda:0"))
    else:
        print("CUDA not available in this environment -- only running on CPU. "
              "Re-run this script on the GPU server to get the comparison that matters.")

    results = {}
    for device in devices:
        probe_radial_basis(device)
        results[device] = probe_seed_pathway(
            device, trials=args.trials, n_max_atoms=args.n_max_atoms, seed_num_basis=args.seed_num_basis,
        )
        if not args.skip_min_abs_radial_probe:
            probe_min_abs_radial_padding_bug(device, trials=args.min_abs_radial_trials)

    if len(devices) == 2:
        cpu_worst = results[devices[0]]["max_abs"]
        gpu_worst = results[devices[1]]["max_abs"]
        ratio = gpu_worst / max(cpu_worst, 1e-30)
        print(f"\n=== CPU vs GPU comparison ===")
        print(f"  worst-case max|f1 or f2|: CPU={cpu_worst:.6g}  GPU={gpu_worst:.6g}  ratio={ratio:.3g}")
        if ratio > 10 or ratio < 0.1:
            print("  >> SIGNIFICANT CPU/GPU DIVERGENCE -- this is strong evidence the issue is "
                  "device/precision-dependent (e.g. TF32 matmuls), not a pure logic bug.")


if __name__ == "__main__":
    sys.exit(main())
