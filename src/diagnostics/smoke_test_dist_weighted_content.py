from __future__ import annotations

import torch
import torch.nn.functional as F

from stage_1.experiments.neurreps.oversquashing.eqfsa_classifier import EqFSAClassifier
from stage_1.experiments.neurreps.oversquashing.synthetic_dataset import (
    GeometricGraphTransferDataset, graph_transfer_collate_fn,
)
from utils.se3_utils import RadialNetwork

FEATURE_DIM, HIDDEN_DIM = 24, 32
CUTOFF = 5.0
CLASSES = 5


def build_dataset(r=6, seed=0):
    for attempt in range(20):
        try:
            return GeometricGraphTransferDataset(
                "chain", r, classes=CLASSES, samples=20, seed=seed + attempt * 1000003, cutoff=CUTOFF,
            )
        except RuntimeError:
            continue
    raise RuntimeError("no safe seed found")


def build_model(dist_weighted_content: bool) -> EqFSAClassifier:
    torch.manual_seed(123)
    return EqFSAClassifier(
        radial_net=RadialNetwork, num_atom_types=3, max_degree=2, num_layers=4,
        feature_dim=FEATURE_DIM, hidden_dim=HIDDEN_DIM, edge_cutoff=CUTOFF, n_max=1,
        classes=CLASSES, edge_feat_dim=1,
        dist_weighted_content=dist_weighted_content,
    )


def run_forward_backward(model: EqFSAClassifier, dataset, label: str):
    print(f"\n=== {label} ===")
    batch = graph_transfer_collate_fn([dataset[i] for i in range(8)])
    x, types, masses = batch["x"], batch["types"], batch["masses"]
    mask, V, mol_props, edge_feats = batch["mask"], batch["V"], batch["mol_props"], batch["edge_feats"]
    seed_masses = batch.get("seed_masses")
    target_idx, labels = batch["target_idx"], batch["labels"]

    logits = model(
        x=x, types=types, masses=masses, mask=mask, V=V, mol_props=mol_props,
        edge_feats=edge_feats, target_idx=target_idx, seed_masses=seed_masses,
    )
    loss = F.cross_entropy(logits, labels)
    print(f"  logits finite: {torch.isfinite(logits).all().item()}, loss: {loss.item():.4f}")
    model.zero_grad()
    loss.backward()
    n_params = 0
    n_nonfinite = 0
    n_none = 0
    for name, p in model.named_parameters():
        n_params += 1
        if p.grad is None:
            n_none += 1
        elif not torch.isfinite(p.grad).all():
            n_nonfinite += 1
            print(f"  NON-FINITE GRAD: {name}")
    print(f"  params={n_params}, none_grad={n_none}, non_finite_grad={n_nonfinite}")
    return loss.item()


def main():
    dataset = build_dataset()
    print(f"dataset: n_atoms={dataset.n_atoms} realized_r={dataset.realized_r} classes={dataset.classes}")

    model_off = build_model(dist_weighted_content=False)
    assert isinstance(model_off.energy_model.layers[0].master_attn.W_QK.gamma0, torch.Tensor)
    in_dim_off = model_off.energy_model.layers[0].master_attn.W_QK.W1.shape[-1]
    print(f"dist_weighted_content=False: W_QK.W1 input dim = {in_dim_off} (should equal feature_dim={FEATURE_DIM})")
    assert in_dim_off == FEATURE_DIM
    run_forward_backward(model_off, dataset, "dist_weighted_content=False (regression check)")

    model_on = build_model(dist_weighted_content=True)
    in_dim_on = model_on.energy_model.layers[0].master_attn.W_QK.W1.shape[-1]
    print(f"\ndist_weighted_content=True: W_QK.W1 input dim = {in_dim_on} (should equal feature_dim+12={FEATURE_DIM + 12})")
    assert in_dim_on == FEATURE_DIM + 12
    run_forward_backward(model_on, dataset, "dist_weighted_content=True (new feature)")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
