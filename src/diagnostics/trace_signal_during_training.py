from __future__ import annotations

import torch
import torch.nn.functional as F

from diagnostics.trace_eqfsa_classifier_signal import build_dataset, build_model, _one_per_class, report
from stage_1.model import minimum_image, masked_weighted_com, masked_mean, seed_equivariant_features, \
    safe_minimum_image_cutoff
from stage_1.experiments.neurreps.oversquashing.synthetic_dataset import graph_transfer_collate_fn
from stage_1.experiments.neurreps.oversquashing.train_graph_transfer import build_optimizer


@torch.no_grad()
def snapshot(model, dataset, label: str):
    print(f"\n=== snapshot: {label} ===")
    em = model.energy_model
    batch = graph_transfer_collate_fn(_one_per_class(dataset))
    x, types, masses = batch["x"], batch["types"], batch["masses"]
    mask, V, mol_props, edge_feats = batch["mask"], batch["V"], batch["mol_props"], batch["edge_feats"]
    seed_masses = batch.get("seed_masses")
    B = x.shape[0]
    node_mask = mask.bool()

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

    batch_ar = torch.arange(B)
    label_idx_t = torch.full((B,), dataset.label_idx, dtype=torch.long)
    agg_per_atom = em.master_pool_agg(f0[0], f0[1], f0[2]).squeeze(-1)
    report("  agg_per_atom @ label atom (pre-pool)", agg_per_atom[batch_ar, label_idx_t])
    if em.attention_pool:
        g_a_pre_norm = em.pool_expand(agg_per_atom, node_mask).unsqueeze(-1)
        report("  g_a (post attention pool_expand, pre-norm)", g_a_pre_norm.reshape(B, -1))
    else:
        mean_pooled = masked_mean(agg_per_atom, node_mask)
        report("  mean_pooled (post masked_mean, pre-FFN)", mean_pooled)
        pooled = em.master_pool_ffn(mean_pooled)
        report("  pooled (post master_pool_ffn)", pooled)
        g_a_pre_norm = em.master_expand(pooled).view(B, em.n_master_nodes, -1).unsqueeze(-1)
        report("  g_a (post master_expand, pre-norm)", g_a_pre_norm.reshape(B, -1))
    g_a = em.master_expand_norm(g_a_pre_norm)
    report("  g_a (pre-loop, post master_expand_norm)", g_a.reshape(B, -1))
    if hasattr(em.master_expand_norm, "gamma"):
        print(f"    ||master_expand_norm.gamma||={em.master_expand_norm.gamma.norm().item():.4f}")
    pre_norm_per_class_mag = g_a_pre_norm.reshape(B, -1).norm(dim=-1)
    mag_mean = pre_norm_per_class_mag.mean().item()
    mag_cv = pre_norm_per_class_mag.std().item() / (mag_mean + 1e-12)
    print(f"    g_a_pre_norm per-class magnitude: {pre_norm_per_class_mag.tolist()} (CV={mag_cv:.4%})")

    for i, layer in enumerate(em.layers):
        B_, N, _ = x.shape
        x_i = x_wrapped.unsqueeze(2).expand(B_, N, N, 3)
        x_j = x_wrapped.unsqueeze(1).expand(B_, N, N, 3)
        dist = (x_j - x_i).norm(dim=-1)
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

        master_attn_result = layer.master_attn(
            g_a, f_in[0], x_wrapped, x_com, node_mask, V,
            normalize_z_by_n_real=layer.normalize_z_by_n_real,
            decouple_ri_norm=layer.decouple_ri_norm,
        )
        master_attn_out = master_attn_result
        g_a_out = layer.master_ln(layer.W_master_self(g_a) + master_attn_out)

        ma = layer.master_attn
        gamma_q = ma.w_vq_norm.gamma.norm().item()
        gamma_k = ma.w_vk_norm.gamma.norm().item()
        report(f"  [layer {i}] master_attn_out", master_attn_out.reshape(B, -1))
        report(f"  [layer {i}] g_a_out", g_a_out.reshape(B, -1))
        print(f"    [layer {i}] ||w_vq_norm.gamma||={gamma_q:.4f}  ||w_vk_norm.gamma||={gamma_k:.4f}")

        f = f_next
        g_a = g_a_out


def main(deterministic_pool_agg: bool = False, attention_pool: bool = False, skip_master_expand_norm: bool = False):
    print(f"### deterministic_pool_agg={deterministic_pool_agg} attention_pool={attention_pool} "
          f"skip_master_expand_norm={skip_master_expand_norm} ###")
    torch.manual_seed(0)
    dataset = build_dataset()
    model = build_model(
        zero_init_head=True, deterministic_pool_agg=deterministic_pool_agg, attention_pool=attention_pool,
        skip_master_expand_norm=skip_master_expand_norm,
    )
    optimizer = build_optimizer(model, lr=1e-3, qk_gamma_wd=0.0, base_weight_decay=0.0)

    checkpoints = {0, 25, 50, 100, 150, 200, 300}
    steps = 300
    g = torch.Generator().manual_seed(0)
    losses = []

    for step in range(steps + 1):
        if step in checkpoints:
            model.eval()
            snapshot(model, dataset, f"step={step}")
            model.train()
        if step == steps:
            break
        idx = torch.randint(0, len(dataset), (16,), generator=g).tolist()
        batch = graph_transfer_collate_fn([dataset[i] for i in idx])
        optimizer.zero_grad()
        logits = model(
            batch["x"], batch["types"], batch["masses"], batch["mask"], batch["V"],
            batch["mol_props"], batch["edge_feats"], batch["target_idx"],
            seed_masses=batch.get("seed_masses"),
        )
        loss = F.cross_entropy(logits, batch["labels"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

    print("\nloss every 25 steps:", [f"{l:.4f}" for l in losses[::25]])


if __name__ == "__main__":
    import sys
    main(
        deterministic_pool_agg="--deterministic" in sys.argv,
        attention_pool="--attention-pool" in sys.argv,
        skip_master_expand_norm="--skip-master-expand-norm" in sys.argv,
    )