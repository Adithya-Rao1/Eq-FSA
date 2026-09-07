from __future__ import annotations

import torch
import torch.nn.functional as F

from stage_1.experiments.neurreps.oversquashing.eqfsa_classifier import EqFSAClassifier
from stage_1.experiments.neurreps.oversquashing.synthetic_dataset import (
    GRAPH_EDGE_FEAT_DIM, GeometricGraphTransferDataset, graph_transfer_collate_fn,
)
from stage_1.experiments.neurreps.oversquashing.train_graph_transfer import build_optimizer
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


def build_model(ablate_lr: bool = False, zero_init_head: bool = False) -> EqFSAClassifier:
    torch.manual_seed(123)
    model = EqFSAClassifier(
        radial_net=RadialNetwork, num_atom_types=3, max_degree=2, num_layers=4,
        feature_dim=FEATURE_DIM, hidden_dim=HIDDEN_DIM, edge_cutoff=CUTOFF, n_max=1,
        classes=CLASSES, edge_feat_dim=GRAPH_EDGE_FEAT_DIM,
    )
    model.energy_model.ablate_lr = ablate_lr
    for layer in model.energy_model.layers:
        layer.ablate_lr = ablate_lr
    if zero_init_head:
        with torch.no_grad():
            model.classifier_head[-1].weight.weight.zero_()
    return model


def call_model(model, batch):
    return model(
        batch["x"], batch["types"], batch["masses"], batch["mask"], batch["V"],
        batch["mol_props"], batch["edge_feats"], batch["target_idx"], seed_masses=batch.get("seed_masses"),
    )


def run_short_training(
    ablate_lr: bool, dataset, steps=150, batch_size=16, log_every=25, zero_init_head=False,
    base_weight_decay=1e-4, lr=1e-3,
):
    model = build_model(ablate_lr=ablate_lr, zero_init_head=zero_init_head)
    optimizer = build_optimizer(model, lr=lr, qk_gamma_wd=0.0, base_weight_decay=base_weight_decay)
    g = torch.Generator().manual_seed(0)
    print(f"\n=== training, ablate_lr={ablate_lr}, zero_init_head={zero_init_head}, "
          f"base_weight_decay={base_weight_decay}, lr={lr} ===")
    losses = []
    for step in range(steps):
        idx = torch.randint(0, len(dataset), (batch_size,), generator=g).tolist()
        batch = graph_transfer_collate_fn([dataset[i] for i in idx])
        optimizer.zero_grad()
        logits = call_model(model, batch)
        loss = F.cross_entropy(logits, batch["labels"])
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
        if step % log_every == 0 or step == steps - 1:
            print(f"  step={step:4d} loss={loss.item():.4f} grad_norm={float(grad_norm):.4f}")
    first_10_avg = sum(losses[:10]) / 10
    last_10_avg = sum(losses[-10:]) / 10
    print(f"  loss avg(first 10)={first_10_avg:.4f} avg(last 10)={last_10_avg:.4f} "
          f"delta={first_10_avg - last_10_avg:+.4f}")
    return model, losses


def _one_per_class(dataset):
    seen = {}
    for i, label in enumerate(dataset._labels):
        if label not in seen:
            seen[label] = i
        if len(seen) == dataset.classes:
            break
    return [dataset[seen[c]] for c in sorted(seen)]


def check_gradient_flow(model, dataset):
    print("\n=== gradient magnitude by submodule (single backward pass, ablate_lr=False) ===")
    print("  (batch: ONE sample per class, so inputs genuinely differ in label-atom position)")
    batch = graph_transfer_collate_fn(_one_per_class(dataset))
    model.zero_grad(set_to_none=True)
    logits = call_model(model, batch)
    print(f"  labels in batch: {batch['labels'].tolist()}")
    print(f"  logits per example:\n{logits.detach()}")
    print(f"  logits std across batch (per class column): {logits.std(dim=0).detach()}")
    loss = F.cross_entropy(logits, batch["labels"])
    print(f"  loss={loss.item():.4f} (chance-level cross-entropy for {CLASSES} classes = {torch.log(torch.tensor(float(CLASSES))).item():.4f})")
    loss.backward()

    groups = {
        "input_embedding": "energy_model.input_embedding",
        "local_SRAttention (layer 0)": "energy_model.layers.0.sr_attn",
        "per-layer master_attn (layer 0)": "energy_model.layers.0.master_attn",
        "master_fusion": "energy_model.master_fusion",
        "classifier_head": "classifier_head",
    }
    for label, prefix in groups.items():
        norms = [
            p.grad.norm().item() for name, p in model.named_parameters()
            if name.startswith(prefix) and p.grad is not None
        ]
        if not norms:
            print(f"  {label:32s}: NO PARAMETERS MATCHED prefix {prefix!r}")
            continue
        total = sum(n ** 2 for n in norms) ** 0.5
        print(f"  {label:32s}: n_params_with_grad={len(norms):3d} total_grad_norm={total:.6e} "
              f"max={max(norms):.6e} min={min(norms):.6e}")


def check_seed_features_vary_with_label_position(model, dataset):
    from stage_1.model import seed_equivariant_features, minimum_image, masked_weighted_com, masked_mean

    print("\n=== do per-atom seed features (f0) actually vary with the label atom's position? ===")
    em = model.energy_model
    batch = graph_transfer_collate_fn(_one_per_class(dataset))
    x, types, masses = batch["x"], batch["types"], batch["masses"]
    mask, V, mol_props = batch["mask"], batch["V"], batch["mol_props"]
    seed_masses = batch.get("seed_masses")
    node_mask = mask.bool()
    masses = masses * node_mask.to(masses.dtype)
    sm = masses if seed_masses is None else seed_masses * node_mask.to(seed_masses.dtype)

    f0_raw = em.input_embedding(types)
    f0_ffn_out = em.f0_ffn(em.ln1(f0_raw))
    mol_ctx = em.mol_prop_emb(mol_props).unsqueeze(1).expand(-1, x.shape[1], -1)
    mol_ffn_out = em.mol_ffn(em.ln2(mol_ctx))
    combined_scalar = em.seed_proj(torch.cat([f0_ffn_out, mol_ffn_out], dim=-1))
    combined_scalar = em.seed_proj_norm(em.seed_proj_out(combined_scalar))
    seeded = seed_equivariant_features(combined_scalar, em.seed_radial_net, x, sm, node_mask, V)

    label_idx, readout_idx = dataset.label_idx, dataset.readout_idx
    for l in (1, 2):
        at_label = seeded[l][:, label_idx].reshape(seeded[l].shape[0], -1)
        at_readout = seeded[l][:, readout_idx].reshape(seeded[l].shape[0], -1)
        print(f"  degree-{l}: label_idx norm per class={at_label.norm(dim=-1).detach().tolist()}, "
              f"std={at_label.norm(dim=-1).std().item():.6e}")
        print(f"            readout_idx norm per class={at_readout.norm(dim=-1).detach().tolist()}, "
              f"std={at_readout.norm(dim=-1).std().item():.6e}")

    x_wrapped = minimum_image(x, V)
    x_com, _ = masked_weighted_com(x_wrapped, masses, node_mask)
    print(f"  master-node x_com per class:\n{x_com.detach()}")
    print(f"  x_com std across class (per coord): {x_com.std(dim=0).detach()}")


def main():
    torch.manual_seed(0)
    dataset = build_dataset()
    print(f"dataset: n_atoms={dataset.n_atoms} realized_r={dataset.realized_r} "
          f"readout_idx={dataset.readout_idx} label_idx={dataset.label_idx} classes={dataset.classes}")

    model_full = build_model(ablate_lr=False)
    check_seed_features_vary_with_label_position(model_full, dataset)
    check_gradient_flow(model_full, dataset)

    run_short_training(ablate_lr=False, dataset=dataset, steps=150)
    run_short_training(ablate_lr=True, dataset=dataset, steps=150)
    run_short_training(ablate_lr=False, dataset=dataset, steps=300, zero_init_head=True)
    run_short_training(ablate_lr=False, dataset=dataset, steps=300, zero_init_head=True, base_weight_decay=0.0)
    run_short_training(
        ablate_lr=False, dataset=dataset, steps=300, zero_init_head=True, base_weight_decay=0.0, lr=1e-2,
    )

if __name__ == "__main__":
    main()