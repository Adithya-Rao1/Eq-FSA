from __future__ import annotations

import torch

from stage_1.experiments.neurreps.oversquashing.eqfsa_classifier import EqFSAClassifier
from stage_1.experiments.neurreps.oversquashing.synthetic_dataset import (
    GRAPH_EDGE_FEAT_DIM, GeometricGraphTransferDataset, graph_transfer_collate_fn,
)
from stage_1.model import seed_equivariant_features, minimum_image, masked_weighted_com
from utils.se3_utils import RadialNetwork

FEATURE_DIM, HIDDEN_DIM = 24, 32
CUTOFF = 5.0
CLASSES = 5


def build_dataset(topology="chain", r=2, classes=CLASSES, samples=60, seed=0, retries=20):
    for attempt in range(retries):
        try:
            return GeometricGraphTransferDataset(
                topology=topology, r=r, classes=classes, samples=samples,
                seed=seed + attempt * 1000003, cutoff=CUTOFF,
            )
        except RuntimeError:
            continue
    raise RuntimeError("no safe seed found")


def build_model(
    zero_init_head: bool = True, deterministic_pool_agg: bool = True, attention_pool: bool = True,
    skip_master_expand_norm: bool = True,
) -> EqFSAClassifier:
    torch.manual_seed(123)
    model = EqFSAClassifier(
        radial_net=RadialNetwork, num_atom_types=3, max_degree=2, num_layers=4,
        feature_dim=FEATURE_DIM, hidden_dim=HIDDEN_DIM, edge_cutoff=CUTOFF, n_max=1,
        classes=CLASSES, edge_feat_dim=GRAPH_EDGE_FEAT_DIM,
        deterministic_pool_agg=deterministic_pool_agg, attention_pool=attention_pool,
        skip_master_expand_norm=skip_master_expand_norm,
    )
    if zero_init_head:
        with torch.no_grad():
            model.classifier_head[-1].weight.weight.zero_()
    return model


def _one_per_class(dataset):
    seen = {}
    for i, label in enumerate(dataset._labels):
        if label not in seen:
            seen[label] = i
        if len(seen) == dataset.classes:
            break
    assert len(seen) == dataset.classes, f"only found {len(seen)} distinct classes in dataset"
    return [dataset[seen[c]] for c in sorted(seen)]


def report(name: str, t: torch.Tensor):
    """t: (num_classes, D). Reports scale-relative cross-class signal."""
    t = t.detach()
    per_class_norm = t.norm(dim=-1)
    mean_norm = per_class_norm.mean().item()
    deviation = (t - t.mean(dim=0, keepdim=True)).norm(dim=-1).mean().item()
    relative = deviation / (mean_norm + 1e-12)
    print(f"  {name:42s} shape={tuple(t.shape)!s:14s} mean_norm={mean_norm:.4e} "
          f"cross_class_dev={deviation:.4e} relative_signal={relative:.4%}")


def trace(model: EqFSAClassifier, dataset, label: str):
    print(f"\n=== trace: {label} ===")
    em = model.energy_model
    batch = graph_transfer_collate_fn(_one_per_class(dataset))
    x, types, masses = batch["x"], batch["types"], batch["masses"]
    mask, V, mol_props, edge_feats = batch["mask"], batch["V"], batch["mol_props"], batch["edge_feats"]
    target_idx = batch["target_idx"]
    seed_masses = batch.get("seed_masses")
    B = x.shape[0]
    batch_ar = torch.arange(B)
    node_mask = mask.bool()

    with torch.no_grad():
        # --- seed stage ---
        masses_m = masses * node_mask.to(masses.dtype)
        sm = masses_m if seed_masses is None else seed_masses * node_mask.to(seed_masses.dtype)
        f0_raw = em.input_embedding(types)
        f0_ffn_out = em.f0_ffn(em.ln1(f0_raw))
        mol_ctx = em.mol_prop_emb(mol_props).unsqueeze(1).expand(-1, x.shape[1], -1)
        mol_ffn_out = em.mol_ffn(em.ln2(mol_ctx))
        combined_scalar = em.seed_proj(torch.cat([f0_ffn_out, mol_ffn_out], dim=-1))
        combined_scalar = em.seed_proj_norm(em.seed_proj_out(combined_scalar))
        seeded = seed_equivariant_features(combined_scalar, em.seed_radial_net, x, sm, node_mask, V)
        report("seed f1 @ target atom", seeded[1][batch_ar, target_idx].reshape(B, -1))
        report("seed f2 @ target atom", seeded[2][batch_ar, target_idx].reshape(B, -1))
        label_idx_t = torch.full((B,), dataset.label_idx, dtype=torch.long)
        report("seed f1 @ LABEL atom (reference)", seeded[1][batch_ar, label_idx_t].reshape(B, -1))
        report("seed f2 @ LABEL atom (reference)", seeded[2][batch_ar, label_idx_t].reshape(B, -1))

        x_wrapped = minimum_image(x, V)
        dist = (x_wrapped[:, dataset.label_idx] - x_wrapped[:, target_idx[0]]).norm(dim=-1)
        print(f"  dist(label, readout) per class: {dist.tolist()} (edge_cutoff={CUTOFF})")

        # --- after full message passing (pre-readout) ---
        f, layer_states = em(x, types, masses, mask, V, mol_props, edge_feats,
                              return_layer_states=True, seed_masses=seed_masses)
        f_target = {l: f[l][batch_ar, target_idx] for l in f}
        report("f_target[0] (final layer)", f_target[0].reshape(B, -1))
        report("f_target[1] (final layer)", f_target[1].reshape(B, -1))
        report("f_target[2] (final layer)", f_target[2].reshape(B, -1))
        for i, layer_f in enumerate(layer_states):
            lt = {l: layer_f[l][batch_ar, target_idx] for l in layer_f}
            report(f"f_target[0] (layer {i})", lt[0].reshape(B, -1))

        # --- readout collapse ---
        u_vec = model.norm_readout1(f_target[1])
        u_ten = model.norm_readout2(f_target[2])
        report("u_vec = norm_readout1(f_target[1])", u_vec.reshape(B, -1))
        report("u_ten = norm_readout2(f_target[2])", u_ten.reshape(B, -1))

        full_vec = f_target[0].squeeze(-1) + u_vec + u_ten
        report("full_vec (pre classifier_head)", full_vec)

        # --- classifier_head, stage by stage ---
        h = full_vec.unsqueeze(-1)
        for i, sub in enumerate(model.classifier_head):
            h = sub(h)
            flat = h.reshape(B, -1)
            report(f"classifier_head[{i}] {sub.__class__.__name__}", flat)

        logits = model._readout_logits(f, batch_ar, target_idx)
        report("final logits (top-level readout, layer 0 only)", logits)


def main():
    torch.manual_seed(0)
    dataset = build_dataset()
    print(f"dataset: n_atoms={dataset.n_atoms} realized_r={dataset.realized_r} "
          f"readout_idx={dataset.readout_idx} label_idx={dataset.label_idx} classes={dataset.classes}")

    model = build_model(zero_init_head=True)
    trace(model, dataset, "zero-init classifier head, untrained")


if __name__ == "__main__":
    main()