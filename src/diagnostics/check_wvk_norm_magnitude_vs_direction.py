from __future__ import annotations

import torch
import torch.nn.functional as F

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
    model = EqFSAClassifier(
        radial_net=RadialNetwork, num_atom_types=3, max_degree=2, num_layers=4,
        feature_dim=FEATURE_DIM, hidden_dim=HIDDEN_DIM, edge_cutoff=CUTOFF, n_max=1,
        classes=CLASSES, edge_feat_dim=1,
        use_norms=False, dist_weighted_content=True, linear_com=True,
    )
    return model


def run(model: EqFSAClassifier, dataset, r_label: str):
    print(f"\n=== r={r_label} ===")
    em = model.energy_model
    batch = graph_transfer_collate_fn(_one_per_class(dataset))
    x, types, masses = batch["x"], batch["types"], batch["masses"]
    mask, V, mol_props = batch["mask"], batch["V"], batch["mol_props"]
    seed_masses = batch.get("seed_masses")
    B = x.shape[0]
    batch_ar = torch.arange(B)
    node_mask = mask.bool()
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

        x_wrapped = minimum_image(x, V)
        x_com, _ = masked_weighted_com(x_wrapped, atomic_masses, node_mask, V=None if em.linear_com else V)

        layer = em.layers[0]
        ma = layer.master_attn
        f_in = layer.layer_norm_in(f0)

        x_rel_to_com_content = minimum_image(x_wrapped - x_com.unsqueeze(1), V)
        dist_to_com_content = x_rel_to_com_content.norm(dim=-1, keepdim=True)
        dist_basis = ma._dist_basis(dist_to_com_content)
        f_in0_kv = torch.cat([f_in[0].squeeze(-1), dist_basis], dim=-1).unsqueeze(-1)

        agg_per_atom = em.master_pool_agg(f0[0], f0[1], f0[2]).squeeze(-1)
        g_a = em.pool_expand(agg_per_atom, node_mask).unsqueeze(-1)
        g_a = em.master_expand_norm(g_a)
        g_a_kv = F.pad(g_a, (0, 0, 0, ma.dist_rbf_num_basis))

        K_pre_norm = ma.qk_pre_norm(g_a_kv, f_in0_kv)[1]  
        K_pre_norm_label = K_pre_norm[batch_ar, label_idx] 
        flat = K_pre_norm_label.reshape(B, -1)

        overall_scale = flat.norm(dim=-1, keepdim=True)  
        shape_only = flat / overall_scale.clamp(min=1e-12) 

        report("K_pre_norm @ label atom (full: magnitude + direction)", flat)
        report("  -> overall_scale (magnitude only, what w_vk_norm removes)", overall_scale)
        report("  -> shape_only (unit direction, what w_vk_norm preserves)", shape_only)

        K = ma.w_vk_norm(K_pre_norm).squeeze(-1)
        report("K (post-w_vk_norm) @ label atom", K[batch_ar, label_idx])

        phi_K_label = F.elu(K[batch_ar, label_idx]) + 1
        report("phi_K = elu(K)+1 @ label atom", phi_K_label)
        print(f"  K value range @ label atom: min={K[batch_ar, label_idx].min():.4f} max={K[batch_ar, label_idx].max():.4f} "
              f"mean={K[batch_ar, label_idx].mean():.4f} frac_negative={float((K[batch_ar, label_idx] < 0).float().mean()):.3f}")


torch.manual_seed(0)
for r in (6, 8):
    model = build_model()
    dataset = build_dataset(r=r)
    run(model, dataset, str(r))
