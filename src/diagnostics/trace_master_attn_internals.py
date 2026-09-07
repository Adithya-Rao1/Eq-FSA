from __future__ import annotations

"""Stage-by-stage trace of `MasterNodeAttention.forward`'s own internal construction of
`master_attn_out`, at every energy layer.

Continuation of the `master_ln` investigation in `oversquashing/CLAUDE.md` ("master_ln
investigated and exonerated"): `master_ln` only rescales magnitude; the actual driver of g_a's
cross-layer relative-signal shrinkage is that `master_attn_out` (each layer's freshly computed
content) itself carries very little cross-class signal (0.003%-0.048%) despite having access to the
real per-atom `f_in[0]` and the real (class-dependent) `x_com`. This file mirrors
`MasterNodeAttention.forward` (`stage_1/attention.py:190-293`) exactly, stage by stage, to find
which specific step is responsible.
"""

import torch
import torch.nn.functional as F

from stage_1.model import (
    minimum_image, masked_weighted_com, masked_mean, seed_equivariant_features,
    safe_minimum_image_cutoff,
)
from utils.pw_bases import make_wavevectors
from diagnostics.trace_eqfsa_classifier_signal import build_dataset, build_model, _one_per_class, report
from stage_1.experiments.neurreps.oversquashing.synthetic_dataset import graph_transfer_collate_fn


def trace_master_attn_internals(model, dataset, label: str):
    print(f"\n=== master_attn internals trace: {label} ===")
    em = model.energy_model
    batch = graph_transfer_collate_fn(_one_per_class(dataset))
    x, types, masses = batch["x"], batch["types"], batch["masses"]
    mask, V, mol_props, edge_feats = batch["mask"], batch["V"], batch["mol_props"], batch["edge_feats"]
    seed_masses = batch.get("seed_masses")
    B = x.shape[0]
    node_mask = mask.bool()
    m = node_mask.to(x.dtype)

    with torch.no_grad():
        atomic_masses = masses * node_mask.to(masses.dtype)
        sm = atomic_masses if seed_masses is None else seed_masses * node_mask.to(seed_masses.dtype)

        f0_raw = em.input_embedding(types)
        f0_ffn_out = em.f0_ffn(em.ln1(f0_raw))
        mol_ctx = em.mol_prop_emb(mol_props).unsqueeze(1).expand(-1, x.shape[1], -1)
        mol_ffn_out = em.mol_ffn(em.ln2(mol_ctx))
        combined_scalar = em.seed_proj(torch.cat([f0_ffn_out, mol_ffn_out], dim=-1))
        combined_scalar = em.seed_proj_norm(em.seed_proj_out(combined_scalar))
        seeded = seed_equivariant_features(combined_scalar, em.seed_radial_net, x, sm, node_mask, V)
        f0 = {0: f0_raw.unsqueeze(-1), 1: seeded[1], 2: seeded[2]}
        f = dict(f0)

        x_wrapped = minimum_image(x, V)
        x_com, _ = masked_weighted_com(x_wrapped, atomic_masses, node_mask)

        agg_per_atom = em.master_pool_agg(f0[0], f0[1], f0[2]).squeeze(-1)
        pooled = em.master_pool_ffn(masked_mean(agg_per_atom, node_mask))
        g_a = em.master_expand(pooled).view(B, em.n_master_nodes, -1).unsqueeze(-1)
        g_a = em.master_expand_norm(g_a)

        for i, layer in enumerate(em.layers):
            print(f"\n  --- layer {i} ---")
            B_, N, _ = x.shape
            x_i = x_wrapped.unsqueeze(2).expand(B_, N, N, 3)
            x_j = x_wrapped.unsqueeze(1).expand(B_, N, N, 3)
            x_rel = x_j - x_i
            dist = x_rel.norm(dim=-1)
            self_mask = ~torch.eye(N, dtype=torch.bool, device=x.device).unsqueeze(0)
            effective_cutoff = safe_minimum_image_cutoff(V, layer.edge_cutoff).view(B_, 1, 1)
            pair_mask = node_mask.unsqueeze(2) & node_mask.unsqueeze(1) & (dist <= effective_cutoff) & self_mask
            edge_b, edge_i, edge_j = pair_mask.nonzero(as_tuple=True)
            x_rel_e = x_wrapped[edge_b, edge_j] - x_wrapped[edge_b, edge_i]
            edge_feat_e = edge_feats[edge_b, edge_i, edge_j]

            f_in = {l: layer.layer_norm_in[str(l)](f[l]) for l in f}
            f_in_i_e = {k: f_in[k][edge_b, edge_i] for k in range(layer.k_max + 1)}
            f0_j_e = {k: f0[k][edge_b, edge_j] for k in range(layer.k_max + 1)}
            alpha_e, Wv_j_sr_flat = layer.sr_attn(
                f_in_i_e, f0_j_e, x_rel_e, edge_feat_e, edge_b, edge_i, batch_size=B_, num_atoms=N,
            )
            receiver_flat = edge_b * N + edge_i
            f_out = {}
            for l in range(layer.max_degree + 1):
                if l not in f_in:
                    continue
                C = f_in[l].shape[2]
                self_term = layer.W_V_self[str(l)](f_in[l])
                contrib_e = alpha_e.view(-1, 1, 1) * Wv_j_sr_flat[l]
                neighbor_raw = torch.zeros(B_ * N, C, 2 * l + 1, device=x.device, dtype=x.dtype).scatter_add(
                    0, receiver_flat.view(-1, 1, 1).expand(-1, C, 2 * l + 1), contrib_e,
                ).view(B_, N, C, 2 * l + 1)
                neighbor_term = layer.sigma_gate[str(l)](neighbor_raw)
                f_out[l] = layer.layer_norm_mid[str(l)](f_in[l] + self_term + neighbor_term)
            f_next = layer.readout(f_out)
            for l in range(layer.max_degree + 1):
                f_next[l] = layer.layer_norm_out[str(l)](f_next[l] + f_out[l])

            # --- mirror MasterNodeAttention.forward, instrumented ---
            ma = layer.master_attn
            f_in0 = f_in[0]

            Q = ma.w_vq_norm(ma.W_Q(g_a)).squeeze(-1)
            K = ma.w_vk_norm(ma.W_K(f_in0)).squeeze(-1)
            phi_Q = F.elu(Q) + 1
            phi_K = (F.elu(K) + 1) * m.unsqueeze(-1)
            report(f"    phi_Q", phi_Q.reshape(B, -1))
            report(f"    phi_K", phi_K.reshape(B, -1))

            phi_K_sum = phi_K.sum(dim=1)
            Z_raw = torch.einsum("bac,bc->ba", phi_Q, phi_K_sum)
            if layer.normalize_z_by_n_real:
                n_real = node_mask.sum(dim=1, keepdim=True).clamp(min=1).to(Z_raw.dtype)
                Z_raw = Z_raw / n_real
            Z = Z_raw.clamp(min=1e-8)
            report(f"    Z", Z.reshape(B, -1))

            kvecs, _, mode_mask = make_wavevectors(V, ma.n_max, ma.pw_e_cutoff)
            mode_mask_f = mode_mask.to(x.dtype)
            if mode_mask_f.dim() == 1:
                mode_mask_f = mode_mask_f.unsqueeze(0).expand(B, -1)

            alpha_p = F.softplus(ma.alpha) + ma.alpha_floor
            k2 = (kvecs ** 2).sum(dim=-1)
            damping = torch.exp(-k2 / (4.0 * alpha_p)) * mode_mask_f
            damping_sum = damping.sum(dim=-1)
            report(f"    damping_sum", damping_sum.reshape(B, -1))

            phase_g = -torch.einsum("bgi,bmi->bgm", x_wrapped, kvecs)
            cos_g = torch.cos(phase_g) * mode_mask_f.unsqueeze(1)
            sin_g = torch.sin(phase_g) * mode_mask_f.unsqueeze(1)

            V_proj0 = ma.W_V0(f_in0.squeeze(-1)).unsqueeze(-1)
            report(f"    V_proj0", V_proj0.reshape(B, -1))

            raw_S_re0 = torch.einsum("bgc,bgnd,bgm->bmcnd", phi_K, V_proj0, cos_g)
            raw_S_im0 = torch.einsum("bgc,bgnd,bgm->bmcnd", phi_K, V_proj0, sin_g)
            report(f"    raw_S_re0", raw_S_re0.reshape(B, -1))
            report(f"    raw_S_im0", raw_S_im0.reshape(B, -1))
            raw_S_ri = torch.cat([raw_S_re0, raw_S_im0], dim=-2)
            S_ri = ma.s_ri_norm(raw_S_ri.transpose(-1, -2)).transpose(-1, -2)
            S_re0, S_im0 = S_ri.split(ma.feature_dim, dim=-2)
            report(f"    S_re0 (post s_ri_norm)", S_re0.reshape(B, -1))
            report(f"    S_im0 (post s_ri_norm)", S_im0.reshape(B, -1))

            phase_com = torch.einsum("bai,bmi->bam", x_com.unsqueeze(1).expand(B, g_a.shape[1], -1), kvecs)
            cos_com = torch.cos(phase_com) * damping.unsqueeze(1)
            sin_com = torch.sin(phase_com) * damping.unsqueeze(1)
            report(f"    cos_com", cos_com.reshape(B, -1))
            report(f"    sin_com", sin_com.reshape(B, -1))

            QS_re0 = torch.einsum("bac,bmcnd->bamnd", phi_Q, S_re0)
            QS_im0 = torch.einsum("bac,bmcnd->bamnd", phi_Q, S_im0)
            report(f"    QS_re0", QS_re0.reshape(B, -1))
            report(f"    QS_im0", QS_im0.reshape(B, -1))

            cos_b = cos_com.unsqueeze(-1).unsqueeze(-1)
            sin_b = sin_com.unsqueeze(-1).unsqueeze(-1)
            pre_Z = (QS_re0 * cos_b - QS_im0 * sin_b).sum(dim=2)
            report(f"    pre_Z (summed over modes, pre /Z /damping_sum)", pre_Z.reshape(B, -1))
            lr_master = pre_Z / Z.view(B, g_a.shape[1], 1, 1)
            report(f"    lr_master (post /Z)", lr_master.reshape(B, -1))
            lr_master = lr_master / damping_sum.view(B, 1, 1, 1)
            report(f"    lr_master (post /Z /damping_sum) = master_attn_out", lr_master.reshape(B, -1))

            self_g_a = layer.W_master_self(g_a)
            g_a_out = layer.master_ln(self_g_a + lr_master)
            report(f"    g_a_out", g_a_out.reshape(B, -1))

            f = f_next
            g_a = g_a_out


def main():
    torch.manual_seed(0)
    dataset = build_dataset()
    print(f"dataset: n_atoms={dataset.n_atoms} realized_r={dataset.realized_r} "
          f"readout_idx={dataset.readout_idx} label_idx={dataset.label_idx} classes={dataset.classes}")
    model = build_model(zero_init_head=True)
    trace_master_attn_internals(model, dataset, "zero-init classifier head, untrained")


if __name__ == "__main__":
    main()
