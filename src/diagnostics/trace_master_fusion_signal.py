from __future__ import annotations

import torch

from stage_1.model import minimum_image, masked_weighted_com, safe_minimum_image_cutoff
from stage_1.attention import equivariant_weight_matrix, softmax_over_neighbors
from utils.se3_utils import clebsch_gordan_matrix, get_spherical_harmonics
from utils.pw_bases import make_wavevectors
from diagnostics.trace_eqfsa_classifier_signal import (
    build_dataset, build_model, _one_per_class, report,
)
from stage_1.experiments.neurreps.oversquashing.synthetic_dataset import graph_transfer_collate_fn


def trace_master_fusion(model, dataset, label: str, ablate_g_a_variation: bool = False):
    print(f"\n=== master_fusion trace: {label} ===")
    em = model.energy_model
    mf = em.master_fusion
    batch = graph_transfer_collate_fn(_one_per_class(dataset))
    x, types, masses = batch["x"], batch["types"], batch["masses"]
    mask, V, mol_props, edge_feats = batch["mask"], batch["V"], batch["mol_props"], batch["edge_feats"]
    target_idx = batch["target_idx"]
    seed_masses = batch.get("seed_masses")
    B = x.shape[0]
    n = int(target_idx[0].item())
    assert (target_idx == n).all(), "readout atom index must be identical across the batch"
    batch_ar = torch.arange(B)
    node_mask = mask.bool()

    with torch.no_grad():
        atomic_masses = masses * node_mask.to(masses.dtype)
        sm = atomic_masses if seed_masses is None else seed_masses * node_mask.to(seed_masses.dtype)

        f0_raw = em.input_embedding(types)
        f0_ffn_out = em.f0_ffn(em.ln1(f0_raw))
        mol_ctx = em.mol_prop_emb(mol_props).unsqueeze(1).expand(-1, x.shape[1], -1)
        mol_ffn_out = em.mol_ffn(em.ln2(mol_ctx))
        combined_scalar = em.seed_proj(torch.cat([f0_ffn_out, mol_ffn_out], dim=-1))
        combined_scalar = em.seed_proj_norm(em.seed_proj_out(combined_scalar))
        from stage_1.model import seed_equivariant_features
        seeded = seed_equivariant_features(combined_scalar, em.seed_radial_net, x, sm, node_mask, V)
        f0 = {0: f0_raw.unsqueeze(-1), 1: seeded[1], 2: seeded[2]}
        f = dict(f0)

        x_wrapped = minimum_image(x, V)
        x_com, _ = masked_weighted_com(x_wrapped, atomic_masses, node_mask)

        agg_per_atom = em.master_pool_agg(f0[0], f0[1], f0[2]).squeeze(-1)
        pooled = em.master_pool_ffn(masked_mean_local(agg_per_atom, node_mask))
        B_, = (B,)
        g_a = em.master_expand(pooled).view(B, em.n_master_nodes, -1).unsqueeze(-1)
        g_a = em.master_expand_norm(g_a)
        report("g_a (post master_expand_norm, pre-loop)", g_a.reshape(B, -1))

        N = x.shape[1]
        x_i = x_wrapped.unsqueeze(2).expand(B, N, N, 3)
        x_j = x_wrapped.unsqueeze(1).expand(B, N, N, 3)
        dist = (x_j - x_i).norm(dim=-1)
        self_mask = ~torch.eye(N, dtype=torch.bool, device=x.device).unsqueeze(0)
        effective_cutoff = safe_minimum_image_cutoff(V, em.edge_cutoff).view(B, 1, 1)
        pair_mask = node_mask.unsqueeze(2) & node_mask.unsqueeze(1) & (dist <= effective_cutoff) & self_mask
        edge_b, edge_i, edge_j = pair_mask.nonzero(as_tuple=True)
        x_rel_e = x_wrapped[edge_b, edge_j] - x_wrapped[edge_b, edge_i]
        edge_feat_e = edge_feats[edge_b, edge_i, edge_j]
        kvecs, _, mode_mask = make_wavevectors(V, em.n_max, em.pw_e_cutoff)

        for i, layer in enumerate(em.layers):
            f, g_a, scale_feat = layer(
                f_in=f, f0=f0, g_a_in=g_a, x_wrapped=x_wrapped, x_com=x_com, node_mask=node_mask,
                edge_b=edge_b, edge_i=edge_i, edge_j=edge_j, x_rel_e=x_rel_e, edge_feat_e=edge_feat_e,
                kvecs=kvecs, mode_mask=mode_mask,
                return_scale_feat=True,
            )
            report(f"g_a (after layer {i})", g_a.reshape(B, -1))
            report(f"scale_feat (s_ri_norm side-channel, after layer {i})", scale_feat.reshape(B, -1))
            report(f"f[0] @ readout (after layer {i})", f[0][batch_ar, target_idx].reshape(B, -1))

        if ablate_g_a_variation:
            g_a = g_a.mean(dim=0, keepdim=True).expand_as(g_a).contiguous()
            report("g_a (ABLATED: class-mean, variation removed)", g_a.reshape(B, -1))

        # --- mirror MasterFusionAttention.forward, instrumented ---
        f_atoms = f
        B_, N, C, _ = f_atoms[0].shape
        N_a = g_a.shape[1]
        H, Ch = mf.num_heads, mf.head_dim

        x_rel = minimum_image(x, V) - x_com.unsqueeze(1)
        report("x_rel @ readout (x - x_com)", x_rel[batch_ar, target_idx])
        x_rel_flat = x_rel.reshape(B * N, 3)
        has_dir = (x_rel.norm(dim=-1) > 1e-7).to(x.dtype).view(B, N, 1, 1)

        Q = {l: mf.W_Q[str(l)](f_atoms[l]) for l in range(mf.max_degree + 1)}
        for l in range(mf.max_degree + 1):
            report(f"Q[{l}] @ readout", Q[l][batch_ar, target_idx].reshape(B, -1))

        K_lifted, V_lifted = {}, {}
        for l in range(mf.max_degree + 1):
            K_content = mf.readout_K[str(l)](g_a)
            V_content = mf.readout_V[str(l)](g_a)
            report(f"K_content[{l}] (per master node, shared across atoms)", K_content.reshape(B, -1))
            report(f"V_content[{l}] (per master node, shared across atoms)", V_content.reshape(B, -1))

            r_kv = x_rel_flat.norm(dim=-1, keepdim=True)
            phi_kv = mf.radial_KV[str(l)](r_kv)
            cg_kv = clebsch_gordan_matrix(l, 0, dtype=x.dtype)
            Q_J_kv = cg_kv[l].to(device=x.device, dtype=x.dtype)
            Y_J_kv = get_spherical_harmonics(l, x_rel_flat)
            QTY_kv = torch.einsum("ji,ni->nj", Q_J_kv, Y_J_kv)
            W_K = (phi_kv[0] * QTY_kv).view(B, N, 2 * l + 1, 1)
            W_V = (phi_kv[1] * QTY_kv).view(B, N, 2 * l + 1, 1)
            if l > 0:
                W_K = W_K * has_dir
                W_V = W_V * has_dir
            report(f"W_K[{l}] @ readout (position-lift weight)", W_K[batch_ar, target_idx].reshape(B, -1))

            K_lifted[l] = torch.einsum("bnij,bacj->bnaci", W_K, K_content)
            V_lifted[l] = torch.einsum("bnij,bacj->bnaci", W_V, V_content)
            report(f"K_lifted[{l}] @ readout", K_lifted[l][batch_ar, target_idx].reshape(B, -1))
            report(f"V_lifted[{l}] @ readout", V_lifted[l][batch_ar, target_idx].reshape(B, -1))

        score = None
        for l in range(mf.max_degree + 1):
            Qh = Q[l].view(B, N, H, Ch, 2 * l + 1)
            Kh = K_lifted[l].view(B, N, N_a, H, Ch, 2 * l + 1)
            contrib = torch.einsum("bnhcd,bnahcd->bnah", Qh, Kh)
            score = contrib if score is None else score + contrib
        score = score / mf.scale
        report("score @ readout (pre-softmax, per master node/head)", score[batch_ar, target_idx].reshape(B, -1))

        alpha = softmax_over_neighbors(score.permute(0, 1, 3, 2))
        alpha = alpha.permute(0, 1, 3, 2)
        report("alpha @ readout (post-softmax over master nodes)", alpha[batch_ar, target_idx].reshape(B, -1))

        out = {}
        for l in range(mf.max_degree + 1):
            Vh = V_lifted[l].view(B, N, N_a, H, Ch, 2 * l + 1)
            weighted = torch.einsum("bnah,bnahcd->bnhcd", alpha, Vh)
            out[l] = weighted.reshape(B, N, C, 2 * l + 1)
            report(f"out[{l}] @ readout (master_fusion's own output)", out[l][batch_ar, target_idx].reshape(B, -1))

        fused = {l: em.layer_norm_fusion[str(l)](f[l] + out[l]) for l in f}
        for l in fused:
            report(f"fused[{l}] @ readout (post layer_norm_fusion)", fused[l][batch_ar, target_idx].reshape(B, -1))


def masked_mean_local(feat, mask):
    from stage_1.model import masked_mean
    return masked_mean(feat, mask)


def main():
    torch.manual_seed(0)
    dataset = build_dataset()
    print(f"dataset: n_atoms={dataset.n_atoms} realized_r={dataset.realized_r} "
          f"readout_idx={dataset.readout_idx} label_idx={dataset.label_idx} classes={dataset.classes}")
    model = build_model(zero_init_head=True)
    trace_master_fusion(model, dataset, "zero-init classifier head, untrained")
    trace_master_fusion(model, dataset, "ABLATION: g_a's cross-class variation removed",
                         ablate_g_a_variation=True)


if __name__ == "__main__":
    main()