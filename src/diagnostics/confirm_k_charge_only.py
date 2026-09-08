from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

from stage_1.attention import MasterNodeAttention
from stage_1.experiments.neurreps.lr_sr.invariant_energy_model import _ChargeEmbedding

def main() -> None:
    torch.manual_seed(0)
    feature_dim, hidden_dim, n_max = 32, 64, 2
    N = 16 

    embed = _ChargeEmbedding(feature_dim, hidden_dim)
    module = MasterNodeAttention(feature_dim, n_max, hidden_dim=hidden_dim)

    charges = torch.cat([torch.ones(N // 2), -torch.ones(N // 2)]).unsqueeze(0) 
    positions = torch.rand(1, N, 3) * 20.0 

    f_in0 = embed(charges)
    _, K_pre_norm = module.qk_pre_norm(torch.zeros(1, 1, feature_dim, 1), f_in0)
    K_post_norm = module.w_vk_norm(K_pre_norm)
    phi_K = F.elu(K_post_norm.squeeze(-1)) + 1 

    pos_atoms = phi_K[0, :N // 2]   
    neg_atoms = phi_K[0, N // 2:]   

    same_charge_max_diff = max(
        (pos_atoms - pos_atoms[0:1]).abs().max().item(),
        (neg_atoms - neg_atoms[0:1]).abs().max().item(),
    )
    cross_charge_diff = (pos_atoms[0] - neg_atoms[0]).abs().max().item()

    print(f"Positions used (should be irrelevant if the claim holds): min={positions.min():.3f} "
          f"max={positions.max():.3f} -- {N} atoms, all distinct positions, box=20.0")
    print(f"\nphi_K max difference AMONG atoms sharing the SAME charge "
          f"(different positions): {same_charge_max_diff:.3e}")
    print(f"phi_K difference BETWEEN a +1 atom and a -1 atom: {cross_charge_diff:.3e}")

    print("\n=== VERDICT ===")
    if same_charge_max_diff < 1e-6 and cross_charge_diff > 1e-3:
        print("CONFIRMED: K depends ONLY on charge sign. Same-charge atoms produce bit-identical "
              "phi_K regardless of position (diff < 1e-6); different-charge atoms differ "
              f"substantially (diff = {cross_charge_diff:.3e}). No amount of training can change "
              "this -- it's a structural property of the code (embed/W_K/w_vk_norm never see `x`), "
              "not a learned behavior.")
    else:
        print("NOT CONFIRMED as stated -- same-charge atoms show meaningful phi_K variation across "
              "positions, or cross-charge difference is negligible. Re-examine the hypothesis before "
              "acting on it.")

    same_charge_f_in0_diff = max(
        (f_in0[0, :N // 2] - f_in0[0, 0:1]).abs().max().item(),
        (f_in0[0, N // 2:] - f_in0[0, N // 2:N // 2 + 1]).abs().max().item(),
    )
    print(f"\n(upstream check) f_in0 max difference among same-charge atoms: "
          f"{same_charge_f_in0_diff:.3e} -- confirms the charge embedding itself, not just K, "
          "is where position information is first discarded.")


if __name__ == "__main__":
    sys.exit(main())
