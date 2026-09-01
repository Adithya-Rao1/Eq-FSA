from __future__ import annotations

import sys

import torch

from stage_1.experiments.neurreps.lr_sr.invariant_energy_data import generate_batch, make_wavevector_grid
from stage_1.experiments.neurreps.lr_sr.invariant_energy_model import EqFSAEncoder
from stage_1.model import masked_mean, masked_weighted_com

def main() -> None:
    torch.manual_seed(0)
    feature_dim, hidden_dim, n_max, num_master_nodes = 32, 64, 2, 4
    N, box = 32, 20.0

    model = EqFSAEncoder(
        num_master_nodes=num_master_nodes, feature_dim=feature_dim, hidden_dim=hidden_dim,
        n_max=n_max, num_outputs=5,
    )
    model.eval()

    kvecs = make_wavevector_grid(n_max, box=box)
    sigmas = torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0])

    # Two genuinely different batches: different seeds -> different specific atom-charge
    # assignments AND different real positions, same N (so same exact 50/50 charge split).
    charges_a, positions_a, V_a, mask_a, _ = generate_batch(
        1, N, kvecs, sigmas, box=box, generator=torch.Generator().manual_seed(1),
    )
    charges_b, positions_b, V_b, mask_b, _ = generate_batch(
        1, N, kvecs, sigmas, box=box, generator=torch.Generator().manual_seed(2),
    )

    print(f"Same charges (both always exactly balanced): "
          f"{torch.equal(charges_a.sort().values, charges_b.sort().values)}")
    print(f"Different specific atom-charge assignment: "
          f"{not torch.equal(charges_a, charges_b)}")
    print(f"Different positions: {not torch.allclose(positions_a, positions_b)}")

    with torch.no_grad():
        f_in0_a = model.embed(charges_a)
        f_in0_b = model.embed(charges_b)
        masses_a, masses_b = mask_a.float(), mask_b.float()
        x_com_a, _ = masked_weighted_com(positions_a, masses_a, mask_a)
        x_com_b, _ = masked_weighted_com(positions_b, masses_b, mask_b)
        dist_a = (positions_a - x_com_a.unsqueeze(1)).norm(dim=-1, keepdim=True)
        dist_b = (positions_b - x_com_b.unsqueeze(1)).norm(dim=-1, keepdim=True)
        pool_feat_a = f_in0_a.squeeze(-1) + model.dist_to_com_embed(dist_a)
        pool_feat_b = f_in0_b.squeeze(-1) + model.dist_to_com_embed(dist_b)
        pooled_a = model.master_pool_ffn(masked_mean(pool_feat_a, mask_a))
        pooled_b = model.master_pool_ffn(masked_mean(pool_feat_b, mask_b))
        g_a_a = model.master_expand_norm(
            model.master_expand(pooled_a).view(1, num_master_nodes, -1).unsqueeze(-1)
        )
        g_a_b = model.master_expand_norm(
            model.master_expand(pooled_b).view(1, num_master_nodes, -1).unsqueeze(-1)
        )

    pooled_diff = (pooled_a - pooled_b).abs().max().item()
    g_a_diff = (g_a_a - g_a_b).abs().max().item()
    g_a_scale = g_a_a.abs().max().item()

    print(f"\npooled_in max difference between the two DIFFERENT examples: {pooled_diff:.3e}")
    print(f"g_a max difference between the two DIFFERENT examples: {g_a_diff:.3e}  "
          f"(g_a's own scale: {g_a_scale:.3e})")

    print("\n=== VERDICT ===")
    if pooled_diff < 1e-6:
        print("STILL COLLAPSED: pooled_in (hence g_a) is still effectively example-independent even "
              "with the dist_to_com_embed fix -- something else is cancelling out the position "
              "signal. Do not trust real-training results until this is re-examined.")
    else:
        print(f"FIXED: pooled_in genuinely differs by {pooled_diff:.3e} between different examples "
              "now (vs. ~1e-6 before the dist_to_com_embed fix) -- g_a's input-dependence is real.")


if __name__ == "__main__":
    sys.exit(main())
