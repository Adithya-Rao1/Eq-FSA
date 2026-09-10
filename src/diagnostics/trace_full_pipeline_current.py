from __future__ import annotations

import torch
import torch.nn.functional as F

from diagnostics.trace_eqfsa_classifier_signal import build_dataset, _one_per_class, report
from stage_1.experiments.neurreps.oversquashing.eqfsa_classifier import EqFSAClassifier
from stage_1.experiments.neurreps.oversquashing.synthetic_dataset import graph_transfer_collate_fn
from stage_1.model import (
    minimum_image, masked_weighted_com, seed_equivariant_features, safe_minimum_image_cutoff,
)
from utils.se3_utils import RadialNetwork
from utils.pw_bases import make_wavevectors

FEATURE_DIM, HIDDEN_DIM = 24, 32
CUTOFF = 5.0
CLASSES = 5


def build_model() -> EqFSAClassifier:
    torch.manual_seed(123)
    model = EqFSAClassifier(
        radial_net=RadialNetwork, num_atom_types=3, max_degree=2, num_layers=4,
        feature_dim=FEATURE_DIM, hidden_dim=HIDDEN_DIM, edge_cutoff=CUTOFF, n_max=1,
        classes=CLASSES, edge_feat_dim=1,
        use_norms=False, dist_weighted_content=True, linear_com=True,
    )
    with torch.no_grad():
        model.classifier_head[-1].weight.weight.zero_()
    return model


def run(model: EqFSAClassifier, dataset, r_label: str):
    print(f"\n{'='*20} r={r_label} {'='*20}")
    em = model.energy_model
    batch = graph_transfer_collate_fn(_one_per_class(dataset))
    x, types, masses = batch["x"], batch["types"], batch["masses"]
    mask, V, mol_props, edge_feats = batch["mask"], batch["V"], batch["mol_props"], batch["edge_feats"]
    target_idx = batch["target_idx"]
    seed_masses = batch.get("seed_masses")
    B = x.shape[0]
    batch_ar = torch.arange(B)
    node_mask = mask.bool()
    m = node_mask.to(x.dtype)
    label_idx = dataset.label_idx

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
        report("f0[1] (label atom's own seed signal, sanity)", seeded[1][batch_ar, label_idx].reshape(B, -1))

        x_wrapped = minimum_image(x, V)
        x_com, _ = masked_weighted_com(x_wrapped, atomic_masses, node_mask, V=None if em.linear_com else V)

        # --- pooling stage: does attention concentrate on the label atom? ---
        agg_per_atom = em.master_pool_agg(f0[0], f0[1], f0[2]).squeeze(-1)
        report("agg_per_atom @ label atom", agg_per_atom[batch_ar, label_idx].reshape(B, -1))
        Q_pool = em.pool_expand.queries.unsqueeze(0).expand(B, -1, -1)
        scores = torch.einsum("bac,bnc->ban", Q_pool, agg_per_atom) / em.pool_expand.scale
        scores = scores.masked_fill(~node_mask.unsqueeze(1), float("-inf"))
        alpha_pool = torch.softmax(scores, dim=-1)  # (B, n_master_nodes, N)
        n_real = node_mask.sum(dim=1).float()
        uniform_weight = 1.0 / n_real
        print(
            f"  attention-pool weight on label atom (mean over master nodes, per class): "
            f"{alpha_pool[batch_ar, :, label_idx].mean(dim=-1).tolist()}"
        )
        print(f"  uniform weight would be 1/n_atoms = {uniform_weight[0].item():.4f}")
        g_a = em.pool_expand(agg_per_atom, node_mask).unsqueeze(-1)
        g_a = em.master_expand_norm(g_a)
        report("g_a (pre-loop)", g_a.reshape(B, -1))

        # --- layer 0 internals: full MasterNodeAttention decomposition ---
        layer = em.layers[0]
        B_, N, _ = x.shape
        effective_cutoff, kvecs, mode_mask = em._box_derived_quantities(V, B_)
        x_diff_raw = x_wrapped.unsqueeze(1) - x_wrapped.unsqueeze(2)
        x_diff = minimum_image(x_diff_raw.reshape(B_, N * N, 3), V).reshape(B_, N, N, 3)
        dist = x_diff.norm(dim=-1)
        self_mask = ~torch.eye(N, dtype=torch.bool, device=x.device).unsqueeze(0)
        pair_mask = node_mask.unsqueeze(2) & node_mask.unsqueeze(1) & (dist <= effective_cutoff) & self_mask
        edge_b, edge_i, edge_j = pair_mask.nonzero(as_tuple=True)
        x_rel_e = x_diff[edge_b, edge_i, edge_j]
        edge_feat_e = edge_feats[edge_b, edge_i, edge_j]

        f_in = layer.layer_norm_in(f)
        ma = layer.master_attn

        x_rel_to_com_content = minimum_image(x_wrapped - x_com.unsqueeze(1), V)
        dist_to_com_content = x_rel_to_com_content.norm(dim=-1, keepdim=True)
        report("  dist_to_com (raw, all atoms)", dist_to_com_content.squeeze(-1))
        report("  dist_to_com @ label atom", dist_to_com_content[batch_ar, label_idx])

        dist_basis = ma._dist_basis(dist_to_com_content)
        report("  dist_basis (RBF) @ label atom", dist_basis[batch_ar, label_idx].reshape(B, -1))

        f_in0_kv = torch.cat([f_in[0].squeeze(-1), dist_basis], dim=-1).unsqueeze(-1)
        g_a_kv = F.pad(g_a, (0, 0, 0, ma.dist_rbf_num_basis))

        K_pre_norm = ma.qk_pre_norm(g_a_kv, f_in0_kv)[1]
        K = ma.w_vk_norm(K_pre_norm).squeeze(-1)
        phi_K = (F.elu(K) + 1) * m.unsqueeze(-1)
        report("  phi_K (all atoms flat)", phi_K.reshape(B, -1))
        report("  phi_K @ label atom", phi_K[batch_ar, label_idx])

        V_proj0 = ma.W_V0(f_in0_kv.squeeze(-1)).unsqueeze(-1)
        report("  V_proj0 @ label atom", V_proj0[batch_ar, label_idx].reshape(B, -1))

        alpha_p = F.softplus(ma.alpha) + ma.alpha_floor
        k2 = (kvecs ** 2).sum(dim=-1)
        damping = torch.exp(-k2 / (4.0 * alpha_p)) * mode_mask.to(x.dtype)
        damping_sum = damping.sum(dim=-1)
        phase_g = -torch.einsum("bgi,bmi->bgm", x_wrapped, kvecs)
        cos_g = torch.cos(phase_g) * mode_mask.to(x.dtype).unsqueeze(1)
        sin_g = torch.sin(phase_g) * mode_mask.to(x.dtype).unsqueeze(1)

        raw_S_re0 = torch.einsum("bgc,bgnd,bgm->bmcnd", phi_K, V_proj0, cos_g)
        raw_S_im0 = torch.einsum("bgc,bgnd,bgm->bmcnd", phi_K, V_proj0, sin_g)
        report("  raw_S_re0", raw_S_re0.reshape(B, -1))
        report("  raw_S_im0", raw_S_im0.reshape(B, -1))
        S_re0 = ma.s_re_norm(raw_S_re0.transpose(-1, -2)).transpose(-1, -2)
        S_im0 = ma.s_im_norm(raw_S_im0.transpose(-1, -2)).transpose(-1, -2)

        Q_pre_norm = ma.qk_pre_norm(g_a_kv, f_in0_kv)[0]
        Q = ma.w_vq_norm(Q_pre_norm).squeeze(-1)
        phi_Q = F.elu(Q) + 1
        QS_re0 = torch.einsum("bac,bmcnd->bamnd", phi_Q, S_re0)
        QS_im0 = torch.einsum("bac,bmcnd->bamnd", phi_Q, S_im0)
        report("  QS_re0", QS_re0.reshape(B, -1))
        report("  QS_im0", QS_im0.reshape(B, -1))

        x_com_use = x_com.unsqueeze(1).expand(B, g_a.shape[1], -1)
        phase_com = torch.einsum("bai,bmi->bam", x_com_use, kvecs)
        cos_com = torch.cos(phase_com) * damping.unsqueeze(1)
        sin_com = torch.sin(phase_com) * damping.unsqueeze(1)
        report("  cos_com", cos_com.reshape(B, -1))
        report("  sin_com", sin_com.reshape(B, -1))

        cos_b, sin_b = cos_com.unsqueeze(-1).unsqueeze(-1), sin_com.unsqueeze(-1).unsqueeze(-1)
        phi_K_sum = phi_K.sum(dim=1)
        Z = torch.einsum("bac,bc->ba", phi_Q, phi_K_sum).clamp(min=1e-8)
        lr_master = (QS_re0 * cos_b - QS_im0 * sin_b).sum(dim=2) / Z.view(B, -1, 1, 1) / damping_sum.view(B, 1, 1, 1)
        report("  master_attn_out (layer 0)", lr_master.reshape(B, -1))

        # --- run all 4 layers for real, track g_a ---
        for i, real_layer in enumerate(em.layers):
            f, g_a = real_layer(
                f_in=f, f0=f0, g_a_in=g_a, x_wrapped=x_wrapped, x_com=x_com, node_mask=node_mask,
                edge_b=edge_b, edge_i=edge_i, edge_j=edge_j, x_rel_e=x_rel_e, edge_feat_e=edge_feat_e,
                kvecs=kvecs, mode_mask=mode_mask, V=V,
            )
            report(f"  g_a (after layer {i})", g_a.reshape(B, -1))

        # --- MasterFusionAttention decomposition at the readout atom ---
        mf = em.master_fusion
        x_rel = minimum_image(x_wrapped - x_com.unsqueeze(1), V)
        r_norm = x_rel.reshape(B * N, 3).norm(dim=-1, keepdim=True)
        K_content = mf.readout_K["0"](g_a)
        V_content = mf.readout_V["0"](g_a)
        report("  K_content (degree 0)", K_content.reshape(B, -1))
        report("  V_content (degree 0)", V_content.reshape(B, -1))
        report("  x_rel @ readout atom", x_rel[batch_ar, target_idx])

        fused = mf(f_atoms=f, g_a=g_a, x=x_wrapped, x_com=x_com, V=V)
        for l in fused:
            report(f"  fused[{l}] @ readout atom", fused[l][batch_ar, target_idx].reshape(B, -1))

        # --- classifier head, stage by stage ---
        f_target = {l: fused[l][batch_ar, target_idx] for l in fused}
        u_vec = model.norm_readout1(f_target[1])
        u_ten = model.norm_readout2(f_target[2])
        full_vec = f_target[0].squeeze(-1) + u_vec + u_ten
        report("  full_vec (pre classifier_head)", full_vec.reshape(B, -1))
        h = full_vec.unsqueeze(-1)
        for i, mod in enumerate(model.classifier_head):
            h = mod(h)
            report(f"  classifier_head[{i}] ({mod.__class__.__name__})", h.reshape(B, -1))


torch.manual_seed(0)
for r in (6, 8):
    model = build_model()
    dataset = build_dataset(r=r)
    run(model, dataset, str(r))
