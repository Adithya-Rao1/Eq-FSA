from __future__ import annotations

import argparse
import sys

import torch

from stage_1.attention import MasterNodeAttention
from stage_1.model import masked_weighted_com


def build_scenario(
    B: int, N: int, N_a: int, feature_dim: int, box: float, device: torch.device,
    dtype: torch.dtype, seed: int, n_real: int | None = None,
) -> dict:
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(B, N, 3, generator=g).to(device=device, dtype=dtype) * box
    mask = torch.ones(B, N, dtype=torch.bool, device=device)
    if n_real is not None:
        mask[:, n_real:] = False
    masses = mask.to(dtype)
    x_com, _ = masked_weighted_com(x, masses, mask)
    V = (torch.eye(3, dtype=dtype) * box).unsqueeze(0).expand(B, 3, 3).contiguous().to(device)
    g_a = (torch.randn(B, N_a, feature_dim, generator=g) * 0.1).unsqueeze(-1).to(device=device, dtype=dtype)
    f_in0 = torch.randn(B, N, feature_dim, generator=g).unsqueeze(-1).to(device=device, dtype=dtype)
    return {"g_a": g_a, "f_in0": f_in0, "x": x, "x_com": x_com, "mask": mask, "V": V}


def run_scenario(module: MasterNodeAttention, inputs: dict) -> float:
    module.zero_grad(set_to_none=True)
    out = module(inputs["g_a"], inputs["f_in0"], inputs["x"], inputs["x_com"], inputs["mask"], inputs["V"])
    loss = (out ** 2).sum()
    if not torch.isfinite(loss):
        return float("inf")
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(module.parameters(), float("inf"))
    return float(grad_norm)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feature-dim", type=int, default=32)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--n-max", type=int, default=2)
    p.add_argument("--num-master-nodes", type=int, default=4)
    p.add_argument("--box", type=float, default=20.0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--explosion-threshold", type=float, default=50.0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device(args.device) if args.device else (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    print(f"Using device: {device}, dtype: {dtype}", flush=True)

    B, N_a, F, H = args.batch_size, args.num_master_nodes, args.feature_dim, args.hidden_dim

    results = []

    def record(name: str, N: int, n_real: int | None, drift_fn=None, drift_desc: str = "none"):
        torch.manual_seed(args.seed)
        module = MasterNodeAttention(F, args.n_max, hidden_dim=H).to(device=device, dtype=dtype)
        if drift_fn is not None:
            with torch.no_grad():
                drift_fn(module)
        inputs = build_scenario(B, N, N_a, F, args.box, device, dtype, seed=args.seed, n_real=n_real)
        grad_norm = run_scenario(module, inputs)
        bucket = "EXPLOSION" if grad_norm > args.explosion_threshold else "normal"
        print(f"  [{bucket:9s}] {name:40s} N={N:4d} n_real={str(n_real):>5s} drift={drift_desc:20s} "
              f"overall_grad_norm={grad_norm:.6g}", flush=True)
        results.append((name, N, n_real, drift_desc, grad_norm, bucket))

    print("\n=== Baseline scenarios (no adversarial drift) ===")
    record("normal multi-molecule baseline", N=30, n_real=None)
    record("large-N (Round 5's original blowup case)", N=150, n_real=None)
    record("minimum dataset N (real floor, per stage_1/CLAUDE.md)", N=3, n_real=None)
    record("N_real=0 edge case (all-masked)", N=8, n_real=0)

    print("\n=== Adversarial drift on s_ri_norm.gamma (the norm whose interaction with the removed "
          "/N_real this fix directly changes) ===")
    for scale in (50.0, 500.0, 5000.0):
        def drift(module, scale=scale):
            module.s_ri_norm.weight.mul_(scale)
        record(f"s_ri_norm.gamma x{scale:g}, large-N", N=150, n_real=None, drift_fn=drift,
               drift_desc=f"gamma x{scale:g}")

    print("\n=== Adversarial drift on w_vk_norm.gamma (Round 3's noted-but-unexploited gap) ===")
    for scale in (50.0, 500.0):
        def drift(module, scale=scale):
            module.w_vk_norm.gamma.mul_(scale)
        record(f"w_vk_norm.gamma x{scale:g}, large-N", N=150, n_real=None, drift_fn=drift,
               drift_desc=f"gamma x{scale:g}")

    print("\n=== alpha at its floor (Round 1's damping_sum concern) ===")
    def alpha_floor(module):
        module.alpha.fill_(-1e6)
    record("alpha pinned at floor, large-N", N=150, n_real=None, drift_fn=alpha_floor, drift_desc="alpha->floor")

    print("\n=== Fully-escalated combined case (matches Round 5's own worst-case construction) ===")
    def combined_drift(module):
        module.s_ri_norm.weight.mul_(5000.0)
        module.w_vk_norm.gamma.mul_(500.0)
        module.alpha.fill_(-1e6)
    record("combined: gamma x5000 + w_vk_norm gamma x500 + alpha->floor, N=3 (real floor)",
           N=3, n_real=None, drift_fn=combined_drift, drift_desc="combined worst-case")

    n_explosions = sum(1 for r in results if r[-1] == "EXPLOSION")
    print(f"\n=== SUMMARY: {n_explosions}/{len(results)} scenarios exceeded "
          f"overall_grad_norm={args.explosion_threshold} ===")
    for name, N, n_real, drift_desc, grad_norm, bucket in results:
        print(f"  [{bucket:9s}] {name:60s} grad_norm={grad_norm:.6g}")


if __name__ == "__main__":
    sys.exit(main())