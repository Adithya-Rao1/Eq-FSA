from __future__ import annotations

import torch

from stage_1.model import minimum_image, masked_weighted_com, masked_mean, seed_equivariant_features
from diagnostics.trace_eqfsa_classifier_signal import build_dataset, build_model, _one_per_class, report
from stage_1.experiments.neurreps.oversquashing.synthetic_dataset import graph_transfer_collate_fn


def trace_master_ln(model, dataset, label: str):
    print(f"\n=== master_ln trace: {label} ===")
    em = model.energy_model
    batch = graph_transfer_collate_fn(_one_per_class(dataset))
    x, types, masses = batch["x"], batch["types"], batch["masses"]
    mask, V, mol_props, edge_feats = batch["mask"], batch["V"], batch["mol_props"], batch["edge_feats"]
    seed_masses = batch.get("seed_masses")
    B = x.shape[0]
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
        seeded = seed_equivariant_features(combined_scalar, em.seed_radial_net, x, sm, node_mask, V)
        f0 = {0: f0_raw.unsqueeze(-1), 1: seeded[1], 2: seeded[2]}
        f = dict(f0)

        x_wrapped = minimum_image(x, V)
        x_com, _ = masked_weighted_com(x_wrapped, atomic_masses, node_mask)

        agg_per_atom = em.master_pool_agg(f0[0], f0[1], f0[2]).squeeze(-1)
        pooled = em.master_pool_ffn(masked_mean(agg_per_atom, node_mask))
        g_a = em.master_expand(pooled).view(B, em.n_master_nodes, -1).unsqueeze(-1)
        g_a = em.master_expand_norm(g_a)
        report("g_a (pre-loop)", g_a.reshape(B, -1))

        for i, layer in enumerate(em.layers):
            # Mirror EnergyLayer.forward exactly, up through the local-feature update, to get
            # the real f_in[0]/f used by master_attn this layer (not a shortcut).
            B_, N, _ = x.shape
            x_i = x_wrapped.unsqueeze(2).expand(B_, N, N, 3)
            x_j = x_wrapped.unsqueeze(1).expand(B_, N, N, 3)
            x_rel = x_j - x_i
            dist = x_rel.norm(dim=-1)
            self_mask = ~torch.eye(N, dtype=torch.bool, device=x.device).unsqueeze(0)
            from stage_1.model import safe_minimum_image_cutoff
            effective_cutoff = safe_minimum_image_cutoff(V, layer.edge_cutoff).view(B_, 1, 1)
            pair_mask = node_mask.unsqueeze(2) & node_mask.unsqueeze(1) & (dist <= effective_cutoff) & self_mask
            edge_b, edge_i, edge_j = pair_mask.nonzero(as_tuple=True)
            x_rel_e = x_wrapped[edge_b, edge_j] - x_wrapped[edge_b, edge_i]
            edge_feat_e = edge_feats[edge_b, edge_i, edge_j]

            f_in = layer.layer_norm_in(f)
            f_in_i_e = {k: f_in[k][edge_b, edge_i] for k in range(layer.k_max + 1)}
            f0_j_e = {k: f0[k][edge_b, edge_j] for k in range(layer.k_max + 1)}
            alpha_e, Wv_j_sr_flat = layer.sr_attn(
                f_in_i_e, f0_j_e, x_rel_e, edge_feat_e, edge_b, edge_i, batch_size=B_, num_atoms=N,
            )
            receiver_flat = edge_b * N + edge_i
            self_terms_raw = {l: layer.W_V_self_linear[str(l)](f_in[l]) for l in range(layer.max_degree + 1)}
            self_terms = layer.W_V_self_norm(self_terms_raw)
            neighbor_raws = {}
            for l in range(layer.max_degree + 1):
                C = f_in[l].shape[2]
                contrib_e = alpha_e.view(-1, 1, 1) * Wv_j_sr_flat[l]
                neighbor_raws[l] = torch.zeros(B_ * N, C, 2 * l + 1, device=x.device, dtype=x.dtype).scatter_add(
                    0, receiver_flat.view(-1, 1, 1).expand(-1, C, 2 * l + 1), contrib_e,
                ).view(B_, N, C, 2 * l + 1)
            neighbor_terms = layer.sigma_gate(neighbor_raws)
            pre_mid = {l: f_in[l] + self_terms[l] + neighbor_terms[l] for l in range(layer.max_degree + 1)}
            f_out = layer.layer_norm_mid(pre_mid)
            f_next = layer.readout(f_out)
            pre_out = {l: f_next[l] + f_out[l] for l in range(layer.max_degree + 1)}
            f_next = layer.layer_norm_out(pre_out)

            # --- master-node update, instrumented ---
            self_g_a = layer.W_master_self(g_a)
            master_attn_out = layer.master_attn(
                g_a, f_in[0], x_wrapped, x_com, node_mask, V,
                normalize_z_by_n_real=layer.normalize_z_by_n_real,
            )
            pre_norm_sum = self_g_a + master_attn_out
            g_a_out = layer.master_ln(pre_norm_sum)

            report(f"  [layer {i}] self_g_a = W_master_self(g_a_in)", self_g_a.reshape(B, -1))
            report(f"  [layer {i}] master_attn_out (this layer's new content)", master_attn_out.reshape(B, -1))
            report(f"  [layer {i}] pre_norm_sum (self + new, pre-master_ln)", pre_norm_sum.reshape(B, -1))
            report(f"  [layer {i}] g_a_out = master_ln(pre_norm_sum)", g_a_out.reshape(B, -1))

            f = f_next
            g_a = g_a_out


def main():
    torch.manual_seed(0)
    dataset = build_dataset()
    print(f"dataset: n_atoms={dataset.n_atoms} realized_r={dataset.realized_r} "
          f"readout_idx={dataset.readout_idx} label_idx={dataset.label_idx} classes={dataset.classes}")
    model = build_model(zero_init_head=True)
    trace_master_ln(model, dataset, "zero-init classifier head, untrained")


if __name__ == "__main__":
    main()