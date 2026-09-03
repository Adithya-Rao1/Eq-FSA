from __future__ import annotations

import torch

from stage_1.experiments.neurreps.oversquashing.eqfsa_classifier import EqFSAClassifier
from stage_1.experiments.neurreps.oversquashing.synthetic_dataset import GeometricGraphTransferDataset
from stage_1.model import masked_weighted_com, minimum_image
from utils.se3_utils import RadialNetwork

GRAPH_BOX_HALFWIDTH = 1000.0


def build_dataset(topology="chain", r=6, classes=5, seed=0, retries=20):
    for attempt in range(retries):
        try:
            return GeometricGraphTransferDataset(
                topology=topology, r=r, classes=classes, samples=classes, seed=seed + attempt * 1000003,
            )
        except RuntimeError:
            continue
    raise RuntimeError(f"no safe seed found for topology={topology} r={r} within {retries} retries")


def batch_from_dataset(ds: GeometricGraphTransferDataset):
    items = [ds[c] for c in range(ds.classes)]
    x = torch.stack([it["x"] for it in items], dim=0)
    types = torch.stack([it["types"] for it in items], dim=0)
    mask = torch.stack([it["mask"] for it in items], dim=0)
    seed_masses = torch.stack([it["seed_masses"] for it in items], dim=0)
    atomic_masses = mask.float()
    V = torch.eye(3).unsqueeze(0).expand(ds.classes, 3, 3).contiguous() * GRAPH_BOX_HALFWIDTH
    return x, types, mask, atomic_masses, seed_masses, V


def per_class_readout_com_disp(x, mask, mass, readout_idx):
    x_wrapped = minimum_image(x, V=torch.eye(3).unsqueeze(0).expand(x.shape[0], 3, 3) * GRAPH_BOX_HALFWIDTH)
    com, _ = masked_weighted_com(x_wrapped, mass, mask)
    x_rel_readout = x_wrapped[:, readout_idx] - com
    return x_rel_readout


def main():
    torch.manual_seed(0)
    ds = build_dataset()
    x, types, mask, atomic_masses, seed_masses, V = batch_from_dataset(ds)
    readout_idx = ds.readout_idx
    print(f"dataset: topology={ds.topology} realized_r={ds.realized_r} n_atoms={ds.n_atoms} "
          f"readout_idx={readout_idx} label_idx={ds.label_idx} classes={ds.classes}")

    torch.manual_seed(123)
    model = EqFSAClassifier(
        radial_net=RadialNetwork, num_atom_types=ds.num_atom_types, max_degree=2, num_layers=2,
        feature_dim=16, hidden_dim=24, edge_cutoff=100.0, n_max=1, classes=ds.classes, edge_feat_dim=1,
    )
    model.eval()
    mol_props = torch.stack([ds.mol_props] * ds.classes, dim=0)
    edge_feats = torch.zeros(ds.classes, ds.n_atoms, ds.n_atoms, 1)
    target_idx = torch.full((ds.classes,), readout_idx, dtype=torch.long)

    with torch.no_grad():
        logits_none = model(x, types, atomic_masses, mask, V, mol_props, edge_feats, target_idx, seed_masses=None)
        logits_explicit = model(
            x, types, atomic_masses, mask, V, mol_props, edge_feats, target_idx, seed_masses=atomic_masses,
        )
    identical = torch.equal(logits_none, logits_explicit)
    print(f"\n[Check 1] seed_masses=None bit-identical to seed_masses=atomic_masses: {identical}")
    if not identical:
        print(f"  max abs diff: {(logits_none - logits_explicit).abs().max().item():.3e}")

    disp_unblinded = per_class_readout_com_disp(x, mask, atomic_masses, readout_idx)
    disp_blinded = per_class_readout_com_disp(x, mask, seed_masses, readout_idx)
    x_com_master, _ = masked_weighted_com(
        minimum_image(x, V=torch.eye(3).unsqueeze(0).expand(x.shape[0], 3, 3) * GRAPH_BOX_HALFWIDTH),
        atomic_masses, mask,
    )

    unblinded_spread = (disp_unblinded - disp_unblinded.mean(dim=0, keepdim=True)).norm(dim=-1)
    blinded_spread = (disp_blinded - disp_blinded.mean(dim=0, keepdim=True)).norm(dim=-1)
    master_com_spread = (x_com_master - x_com_master.mean(dim=0, keepdim=True)).norm(dim=-1)

    print(f"\n[Check 2] readout atom's seed-COM-relative position, spread across classes "
          f"(should shrink a lot going unblinded -> blinded):")
    print(f"  WITHOUT fix (seed_masses=atomic_masses): per-class spread = {unblinded_spread.tolist()}")
    print(f"  WITH fix    (seed_masses zeroes label_idx): per-class spread = {blinded_spread.tolist()}")
    ratio = blinded_spread.mean().item() / max(unblinded_spread.mean().item(), 1e-12)
    print(f"  mean spread ratio (blinded/unblinded): {ratio:.4f} (expect << 1, ideally ~0)")

    print(f"\n[Check 3] master-node COM (x_com, always uses atomic_masses -- must NOT be blinded):")
    print(f"  per-class spread = {master_com_spread.tolist()}")
    print(f"  mean spread: {master_com_spread.mean().item():.4f} (expect clearly > 0 -- "
          f"legitimate signal must still reach the master-node channel)")

    torch.manual_seed(456)
    fb_model = EqFSAClassifier(
        radial_net=RadialNetwork, num_atom_types=ds.num_atom_types, max_degree=2, num_layers=2,
        feature_dim=16, hidden_dim=24, edge_cutoff=100.0, n_max=1, classes=ds.classes, edge_feat_dim=1,
    )
    logits = fb_model(
        x[:2], types[:2], atomic_masses[:2], mask[:2], V[:2], mol_props[:2], edge_feats[:2],
        target_idx[:2], seed_masses=seed_masses[:2],
    )
    loss = logits.pow(2).sum()
    loss.backward()
    finite_loss = torch.isfinite(loss).item()
    finite_grads = all(torch.isfinite(p.grad).all() for p in fb_model.parameters() if p.grad is not None)
    print(f"\n[Check 4] forward/backward with seed_masses on a 2-sample batch: "
          f"loss finite={finite_loss}, all grads finite={finite_grads}")

    ok = identical and ratio < 0.05 and master_com_spread.mean().item() > 1e-3 and finite_loss and finite_grads
    print(f"\n=== {'PASS' if ok else 'FAIL'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
