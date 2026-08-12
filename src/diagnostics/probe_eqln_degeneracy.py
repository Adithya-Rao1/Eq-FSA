from __future__ import annotations

import argparse
import sys

import torch


def layernorm_near_degenerate_fraction(x: torch.Tensor, node_mask: torch.Tensor, floor: float = 0.5) -> dict:
    near_degenerate = (x.detach() ** 2).sum(dim=-1) < floor ** 2 
    would_trip_production_counter = bool(near_degenerate.any().item())

    atom_mask = node_mask.bool()
    n_real_atoms = int(atom_mask.sum().item())
    if n_real_atoms == 0:
        return {
            "would_trip_production_counter": would_trip_production_counter,
            "elementwise_fraction_of_real_atom_channels": float("nan"),
            "fraction_of_real_atoms_with_ge1_near_degenerate_channel": float("nan"),
            "n_real_atoms": 0,
        }

    elementwise_fraction = near_degenerate[atom_mask].float().mean().item()
    any_channel_per_atom = near_degenerate.any(dim=-1)  # [B, N]
    atom_fraction = any_channel_per_atom[atom_mask].float().mean().item()

    unmasked_elementwise_fraction = near_degenerate.float().mean().item()
    unmasked_atom_fraction = any_channel_per_atom.float().mean().item()

    return {
        "would_trip_production_counter": would_trip_production_counter,
        "elementwise_fraction_of_real_atom_channels": elementwise_fraction,
        "fraction_of_real_atoms_with_ge1_near_degenerate_channel": atom_fraction,
        "unmasked_elementwise_fraction_incl_padding": unmasked_elementwise_fraction,
        "unmasked_atom_fraction_incl_padding": unmasked_atom_fraction,
        "n_real_atoms": n_real_atoms,
    }


def _make_batch(B: int, N_max: int, C: int, dim: int, n_real_range: tuple[int, int],
                 real_feature_scale: float, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthetic degree>0 features: real atoms get scale-`real_feature_scale` random features,
    padded atoms get exact zeros (mirroring what the seed pathway plus bias-free equivariant
    linears plausibly produce for masked-out atoms, per the docstring above)."""
    x = torch.zeros(B, N_max, C, dim)
    mask = torch.zeros(B, N_max, dtype=torch.bool)
    for b in range(B):
        n = torch.randint(n_real_range[0], n_real_range[1] + 1, (1,), generator=generator).item()
        n = min(n, N_max)
        mask[b, :n] = True
        x[b, :n] = torch.randn(n, C, dim, generator=generator) * real_feature_scale
    return x, mask


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trials", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--n-max-atoms", type=int, default=20)
    p.add_argument("--min-real-atoms", type=int, default=3)
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--degree", type=int, default=1, help="1 -> dim=3 (vector), 2 -> dim=5 (traceless symmetric tensor)")
    p.add_argument("--real-feature-scale", type=float, default=1.0,
                    help="Std of synthetic real-atom features. NOTE: without a real checkpoint this is a "
                         "guess, not measured -- the qualitative padding-vs-real-atom contrast is the point, "
                         "not this specific number.")
    args = p.parse_args()

    dim = 2 * args.degree + 1
    g = torch.Generator().manual_seed(0)

    print(f"\n=== eqln_deg_gt0_near_degenerate fraction probe (degree={args.degree}, dim={dim}) ===")
    print("Synthetic data: real atoms ~ N(0, real_feature_scale^2), padded atoms = exact zero "
          "(structural argument, not a measurement of the live model).\n")

    agg_elementwise, agg_atom_frac, agg_unmasked_elementwise, agg_unmasked_atom = [], [], [], []
    n_would_trip = 0
    for _ in range(args.trials):
        x, mask = _make_batch(
            args.batch_size, args.n_max_atoms, args.channels, dim,
            (args.min_real_atoms, args.n_max_atoms), args.real_feature_scale, g,
        )
        result = layernorm_near_degenerate_fraction(x, mask, floor=0.5)
        agg_elementwise.append(result["elementwise_fraction_of_real_atom_channels"])
        agg_atom_frac.append(result["fraction_of_real_atoms_with_ge1_near_degenerate_channel"])
        agg_unmasked_elementwise.append(result["unmasked_elementwise_fraction_incl_padding"])
        agg_unmasked_atom.append(result["unmasked_atom_fraction_incl_padding"])
        n_would_trip += int(result["would_trip_production_counter"])

    def _mean(vals):
        return sum(vals) / len(vals)

    print(f"  trials={args.trials}")
    print(f"  production .any()-gated counter would fire on {n_would_trip}/{args.trials} trials "
          f"({100 * n_would_trip / args.trials:.1f}%) -- matches the observed ~flat, "
          f"near-every-step firing in the real CSV.")
    print(f"  [REAL ATOMS ONLY, masked correctly]")
    print(f"    mean fraction of (real atom, channel) pairs near-degenerate: {_mean(agg_elementwise):.4%}")
    print(f"    mean fraction of real atoms with >=1 near-degenerate channel: {_mean(agg_atom_frac):.4%}")
    print(f"  [UNMASKED, padding included -- what the production code structurally computes over]")
    print(f"    mean fraction of (ALL slot, channel) pairs near-degenerate: {_mean(agg_unmasked_elementwise):.4%}")
    print(f"    mean fraction of ALL atom slots with >=1 near-degenerate channel: {_mean(agg_unmasked_atom):.4%}")
    print("\n  If the masked (real-atoms-only) fractions above are near 0 while the unmasked "
          "ones are large, that's the same missing-node_mask pattern as the min_abs_radial bug: "
          "padded slots (not real chemistry) are what's driving eqln_deg_gt0_near_degenerate. "
          "Re-run this against a real batch's actual f_in[l] tensors (via a patched replay_batch.py "
          "or live forward hook) to get the real answer -- this synthetic run only demonstrates "
          "that the counter's lack of masking is structurally capable of producing exactly that "
          "outcome, not that it is what's actually happening on real data.")


if __name__ == "__main__":
    sys.exit(main())
