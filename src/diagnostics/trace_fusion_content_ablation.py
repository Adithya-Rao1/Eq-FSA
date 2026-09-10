from __future__ import annotations

import torch

from diagnostics.trace_eqfsa_classifier_signal import build_dataset, _one_per_class, report
from stage_1.experiments.neurreps.oversquashing.eqfsa_classifier import EqFSAClassifier
from stage_1.experiments.neurreps.oversquashing.synthetic_dataset import graph_transfer_collate_fn
from stage_1.model import minimum_image, masked_weighted_com, seed_equivariant_features
from utils.se3_utils import RadialNetwork

FEATURE_DIM, HIDDEN_DIM = 24, 32
CUTOFF = 5.0
CLASSES = 5


def build_model() -> EqFSAClassifier:
    torch.manual_seed(123)
    return EqFSAClassifier(
        radial_net=RadialNetwork, num_atom_types=3, max_degree=2, num_layers=4,
        feature_dim=FEATURE_DIM, hidden_dim=HIDDEN_DIM, edge_cutoff=CUTOFF, n_max=1,
        classes=CLASSES, edge_feat_dim=1,
    )


def run(model: EqFSAClassifier, dataset, r_label: str, ablate_g_a_content: bool):
    tag = "ABLATED (g_a -> class-mean at fusion only)" if ablate_g_a_content else "REAL g_a"
    print(f"\n{'='*10} r={r_label}, {tag} {'='*10}")
    em = model.energy_model
    batch = graph_transfer_collate_fn(_one_per_class(dataset))
    x, types, masses = batch["x"], batch["types"], batch["masses"]
    mask, V, mol_props, edge_feats = batch["mask"], batch["V"], batch["mol_props"], batch["edge_feats"]
    seed_masses = batch.get("seed_masses")
    target_idx = batch["target_idx"]
    B = x.shape[0]
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
        seeded = seed_equivariant_features(combined_scalar, em.seed_radial_net, x, sm, node_mask, V)
        f0 = {0: f0_raw.unsqueeze(-1), 1: seeded[1], 2: seeded[2]}
        f = dict(f0)

        x_wrapped = minimum_image(x, V)
        x_com, _ = masked_weighted_com(x_wrapped, atomic_masses, node_mask, V=None if em.linear_com else V)
        x_rel_com = minimum_image(x_wrapped - x_com.unsqueeze(1), V)

        agg_per_atom = em.master_pool_agg(f0[0], f0[1], f0[2], x_rel=x_rel_com).squeeze(-1)
        g_a = em.pool_expand(agg_per_atom, node_mask).unsqueeze(-1)
        g_a = em.master_expand_norm(g_a)

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

        fusion_accum = {l: torch.zeros_like(f0[l]) for l in f0}
        for i, layer in enumerate(em.layers):
            f, g_a = layer(
                f_in=f, f0=f0, g_a_in=g_a, x_wrapped=x_wrapped, x_com=x_com, node_mask=node_mask,
                edge_b=edge_b, edge_i=edge_i, edge_j=edge_j, x_rel_e=x_rel_e, edge_feat_e=edge_feat_e,
                kvecs=kvecs, mode_mask=mode_mask, V=V,
            )
            g_a_for_fusion = g_a.mean(dim=0, keepdim=True).expand_as(g_a) if ablate_g_a_content else g_a
            layer_fusion = em.master_fusion(f_atoms=f, g_a=g_a_for_fusion, x=x_wrapped, x_com=x_com, V=V)
            fusion_accum = {l: fusion_accum[l] + layer_fusion[l] for l in fusion_accum}

        report("  fusion_accum[0] @ readout atom (final)", fusion_accum[0][batch_ar, target_idx].reshape(B, -1))
        f_final = {l: f[l] + fusion_accum[l] for l in f}
        f_final = em.readout(f_final)
        f_target = {l: f_final[l][batch_ar, target_idx] for l in f_final}
        u_vec = model.norm_readout1(f_target[1])
        u_ten = model.norm_readout2(f_target[2])
        full_vec = f_target[0].squeeze(-1) + u_vec + u_ten
        report("  full_vec", full_vec.reshape(B, -1))
        return full_vec


torch.manual_seed(0)
for r in (6, 8):
    dataset = build_dataset(r=r)
    model = build_model()
    real = run(model, dataset, str(r), ablate_g_a_content=False)
    model2 = build_model()
    ablated = run(model2, dataset, str(r), ablate_g_a_content=True)
    rel_change = (real - ablated).norm() / real.norm().clamp(min=1e-12)
    print(f"  >> relative change in full_vec from ablating g_a's content at fusion: {rel_change.item()*100:.2f}%")
