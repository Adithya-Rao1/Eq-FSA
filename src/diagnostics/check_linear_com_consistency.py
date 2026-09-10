from __future__ import annotations

import statistics

import torch

from diagnostics.trace_eqfsa_classifier_signal import build_dataset, _one_per_class
from stage_1.experiments.neurreps.oversquashing.eqfsa_classifier import EqFSAClassifier
from stage_1.experiments.neurreps.oversquashing.synthetic_dataset import graph_transfer_collate_fn
from utils.se3_utils import RadialNetwork

FEATURE_DIM, HIDDEN_DIM = 24, 32
CUTOFF = 5.0
CLASSES = 5
SEEDS = [0, 7919, 15838, 23757, 31676, 39595, 47514]


def build_model(linear_com: bool) -> EqFSAClassifier:
    torch.manual_seed(123)
    model = EqFSAClassifier(
        radial_net=RadialNetwork, num_atom_types=3, max_degree=2, num_layers=4,
        feature_dim=FEATURE_DIM, hidden_dim=HIDDEN_DIM, edge_cutoff=CUTOFF, n_max=1,
        classes=CLASSES, edge_feat_dim=1, linear_com=linear_com,
    )
    with torch.no_grad():
        model.classifier_head[-1].weight.weight.zero_()
    return model


def relative_signal(t: torch.Tensor) -> float:
    per_class_norm = t.norm(dim=-1)
    mean_norm = per_class_norm.mean().item()
    deviation = (t - t.mean(dim=0, keepdim=True)).norm(dim=-1).mean().item()
    return deviation / (mean_norm + 1e-12)


def master_attn_out_signal(model: EqFSAClassifier, dataset) -> float:
    em = model.energy_model
    batch = graph_transfer_collate_fn(_one_per_class(dataset))
    x, types, masses = batch["x"], batch["types"], batch["masses"]
    mask, V, mol_props = batch["mask"], batch["V"], batch["mol_props"]
    seed_masses = batch.get("seed_masses")
    B = x.shape[0]
    node_mask = mask.bool()

    from stage_1.model import minimum_image, masked_weighted_com, seed_equivariant_features

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

        agg_per_atom = em.master_pool_agg(f0[0], f0[1], f0[2]).squeeze(-1)
        g_a = em.pool_expand(agg_per_atom, node_mask).unsqueeze(-1)
        g_a = em.master_expand_norm(g_a)

        layer0 = em.layers[0]
        f_in0 = layer0.layer_norm_in(f0)[0]
        master_attn_out = layer0.master_attn(
            g_a, f_in0, x_wrapped, x_com, node_mask, V,
            normalize_z_by_n_real=layer0.normalize_z_by_n_real,
        )
        return relative_signal(master_attn_out.reshape(B, -1))


def main() -> None:
    for r in (6, 8):
        print(f"\n=== r={r} ===")
        for linear_com in (False, True):
            sigs = []
            for seed in SEEDS:
                dataset = build_dataset(r=r, seed=seed)
                model = build_model(linear_com)
                sig = master_attn_out_signal(model, dataset)
                sigs.append(sig)
                print(f"  linear_com={linear_com!s:5} seed={seed:>6} master_attn_out relative_signal={sig:.4%}")
            mean_ = statistics.mean(sigs)
            stdev_ = statistics.pstdev(sigs)
            cv = stdev_ / mean_ if mean_ > 0 else float("inf")
            print(f"  -> linear_com={linear_com!s:5} mean={mean_:.4%} stdev={stdev_:.4%} CV={cv:.3f}")


if __name__ == "__main__":
    main()
