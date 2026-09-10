from __future__ import annotations

import torch
import torch.nn.functional as F

from diagnostics.trace_eqfsa_classifier_signal import build_dataset, _one_per_class, report
from stage_1.experiments.neurreps.oversquashing.eqfsa_classifier import EqFSAClassifier
from stage_1.experiments.neurreps.oversquashing.synthetic_dataset import graph_transfer_collate_fn
from stage_1.model import minimum_image, masked_weighted_com, seed_equivariant_features, safe_minimum_image_cutoff
from utils.se3_utils import RadialNetwork

FEATURE_DIM, HIDDEN_DIM = 24, 32
CUTOFF = 5.0
CLASSES = 5


def build_model(dist_weighted_content: bool) -> EqFSAClassifier:
    torch.manual_seed(123)
    model = EqFSAClassifier(
        radial_net=RadialNetwork, num_atom_types=3, max_degree=2, num_layers=4,
        feature_dim=FEATURE_DIM, hidden_dim=HIDDEN_DIM, edge_cutoff=CUTOFF, n_max=1,
        classes=CLASSES, edge_feat_dim=1, dist_weighted_content=dist_weighted_content,
    )
    with torch.no_grad():
        model.classifier_head[-1].weight.weight.zero_()
    return model


def trace(model: EqFSAClassifier, dataset, label: str):
    print(f"\n=== {label} ===")
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
        report("f0[0] (raw type embedding)", f0[0].reshape(B, -1))

        x_wrapped = minimum_image(x, V)
        x_com, _ = masked_weighted_com(x_wrapped, atomic_masses, node_mask)

        agg_per_atom = em.master_pool_agg(f0[0], f0[1], f0[2]).squeeze(-1)
        g_a = em.pool_expand(agg_per_atom, node_mask).unsqueeze(-1) if em.attention_pool else None
        g_a = em.master_expand_norm(g_a)

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

            f_in = layer.layer_norm_in(f)

            ma = layer.master_attn
            if ma.dist_weighted_content:
                x_rel_to_com_content = x_wrapped - x_com.unsqueeze(1)
                x_rel_to_com_content = minimum_image(x_rel_to_com_content, V)
                dist_to_com_content = x_rel_to_com_content.norm(dim=-1, keepdim=True)
                dist_basis = ma._dist_basis(dist_to_com_content)
                f_in0_kv = torch.cat([f_in[0].squeeze(-1), dist_basis], dim=-1).unsqueeze(-1)
                g_a_kv = F.pad(g_a, (0, 0, 0, ma.dist_rbf_num_basis))
            else:
                f_in0_kv = f_in[0]
                g_a_kv = g_a

            K_pre_norm = ma.qk_pre_norm(g_a_kv, f_in0_kv)[1]
            K = ma.w_vk_norm(K_pre_norm).squeeze(-1)
            phi_K = (F.elu(K) + 1) * m.unsqueeze(-1)
            V_proj0 = ma.W_V0(f_in0_kv.squeeze(-1)).unsqueeze(-1)
            report(f"  [layer {i}] f_in[0]", f_in[0].reshape(B, -1))
            report(f"  [layer {i}] phi_K", phi_K.reshape(B, -1))
            report(f"  [layer {i}] V_proj0", V_proj0.reshape(B, -1))

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

            master_attn_out = layer.master_attn(
                g_a, f_in[0], x_wrapped, x_com, node_mask, V,
                normalize_z_by_n_real=layer.normalize_z_by_n_real,
            )
            g_a_out = layer.master_ln(layer.W_master_self(g_a) + master_attn_out)
            report(f"  [layer {i}] master_attn_out", master_attn_out.reshape(B, -1))

            f = f_next
            g_a = g_a_out


torch.manual_seed(0)
dataset = build_dataset(r=2)
print(f"dataset: n_atoms={dataset.n_atoms} realized_r={dataset.realized_r} classes={dataset.classes}")

trace(build_model(False), dataset, "dist_weighted_content=False (baseline, Finding A)")
trace(build_model(True), dataset, "dist_weighted_content=True (new fix)")
