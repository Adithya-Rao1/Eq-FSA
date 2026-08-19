from __future__ import annotations

import argparse
import csv
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from utils.se3_utils import RadialNetwork

from stage_1.experiments.neurreps.synthetic_dataset import (
    GRAPH_EDGE_FEAT_DIM,
    GraphTransferDataset,
    graph_transfer_collate_fn,
)
from stage_1.experiments.neurreps.egnn_baseline import EGNNModel
from stage_1.experiments.neurreps.eqfsa_classifier import EqFSAClassifier
from stage_1.experiments.neurreps.metrics import EmbeddingCapture, jacobian_sensitivity, mi_lower_bound


def detect_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _to_device(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def call_model(model, variant: str, batch: dict) -> torch.Tensor:
    if variant == "egnn":
        return model(batch["x"], batch["types"], batch["mask"], batch["target_idx"], edge_feats=batch["edge_feats"])
    return model(
        batch["x"], batch["types"], batch["masses"], batch["mask"], batch["V"],
        batch["mol_props"], batch["edge_feats"], batch["target_idx"],
    )


def build_model_for_variant(variant: str, realized_r: int, classes: int, num_atom_types: int, config: dict):
    if variant == "egnn":
        num_layers = max(1, realized_r + config["egnn_layer_margin"])
        return EGNNModel(
            num_node_types=num_atom_types, hidden_dim=config["egnn_hidden_dim"],
            num_layers=num_layers, cutoff=config["cutoff"], classes=classes,
            edge_feat_dim=GRAPH_EDGE_FEAT_DIM,
        )
    if variant in ("eqfsa_full", "eqfsa_invariant_only"):
        ablate = 0 if variant == "eqfsa_invariant_only" else None
        return EqFSAClassifier(
            radial_net=RadialNetwork, num_atom_types=num_atom_types, max_degree=2,
            num_layers=config["eqfsa_num_layers"], feature_dim=config["feature_dim"],
            hidden_dim=config["hidden_dim"], edge_cutoff=config["cutoff"], n_max=1,
            classes=classes, ablate_max_degree=ablate, edge_feat_dim=GRAPH_EDGE_FEAT_DIM,
        )
    raise ValueError(f"Unknown variant {variant!r}")


def train_one_config(topology: str, r: int, variant: str, config: dict) -> dict:
    classes = config["classes"]
    samples = config["samples_per_config"]
    seed = config["seed"]

    dataset = GraphTransferDataset(
        topology, r, classes, samples, seed=seed, cutoff=config["cutoff"],
        bond_length=config["bond_length"], confinement_radius=config["confinement_radius"],
    )
    realized_r = dataset.realized_r
    realized_euclid = dataset.realized_euclid
    source_idx_val = dataset.source_idx
    num_atom_types = classes + 1

    n_val = max(1, int(round(len(dataset) * config["val_fraction"])))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed + 2),
    )
    train_loader = DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True, collate_fn=graph_transfer_collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["batch_size"], shuffle=False, collate_fn=graph_transfer_collate_fn,
    )

    device = config["_device"]
    model = build_model_for_variant(variant, realized_r, classes, num_atom_types, config).to(device)
    embedding_module = model.input_embedding_module
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)

    for _ in range(config["epochs"]):
        model.train()
        for batch in train_loader:
            batch = _to_device(batch, device)
            optimizer.zero_grad()
            logits = call_model(model, variant, batch)
            loss = F.cross_entropy(logits, batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    model.eval()
    total_ce, total_correct, total_n = 0.0, 0, 0
    sens_values = []
    for batch in val_loader:
        batch = _to_device(batch, device)
        B = batch["labels"].shape[0]
        source_idx_t = torch.full((B,), source_idx_val, dtype=torch.long, device=device)
        with EmbeddingCapture(embedding_module) as cap:
            logits = call_model(model, variant, batch)
            sens_values.append(jacobian_sensitivity(logits, cap.captured, source_idx_t))
        logits_d = logits.detach()
        total_ce += F.cross_entropy(logits_d, batch["labels"], reduction="sum").item()
        total_correct += (logits_d.argmax(dim=-1) == batch["labels"]).sum().item()
        total_n += B

    val_ce = total_ce / max(1, total_n)
    val_error = 1.0 - total_correct / max(1, total_n)
    val_sensitivity = sum(sens_values) / max(1, len(sens_values))
    val_mi = mi_lower_bound(classes, val_ce)
    n_params = sum(p.numel() for p in model.parameters())

    return {
        "topology": topology,
        "variant": variant,
        "nominal_r": r,
        "r": realized_r,
        "euclid": realized_euclid,
        "val_error": val_error,
        "val_ce": val_ce,
        "jacobian_sensitivity": val_sensitivity,
        "mi_lower_bound": val_mi,
        "n_params": n_params,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--topologies", type=str, nargs="+", default=["chain", "clique_chain"],
    )
    parser.add_argument(
        "--r-values", type=int, nargs="+", default=[8, 10, 12, 14, 16, 18, 20, 24],
    )
    parser.add_argument("--variants", type=str, nargs="+", default=["egnn", "eqfsa_full", "eqfsa_invariant_only"])
    parser.add_argument("--classes", type=int, default=5)
    parser.add_argument("--eqfsa-num-layers", type=int, default=4, help="Fixed depth for both Eq-FSA variants, independent of r")
    parser.add_argument("--egnn-layer-margin", type=int, default=4, help="EGNN depth = realized_r + this margin, to rule out underreaching")
    parser.add_argument("--egnn-hidden-dim", type=int, default=32)
    parser.add_argument("--feature-dim", type=int, default=24)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--bond-length", type=float, default=2.0)
    parser.add_argument("--confinement-radius", type=float, default=25.0)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--samples-per-config", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="stage_1/experiments/neurreps/results")
    parser.add_argument("--csv-path", type=str, default=None)
    parser.add_argument(
        "--device", type=str, default=None,
        help="torch device string (e.g. 'cuda:0'); autodetects CUDA if omitted, else CPU",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Tiny r-values/epochs/samples sanity run before committing to the full sweep",
    )
    args = parser.parse_args()
    config = vars(args)
    config["_device"] = detect_device(config["device"])
    print(f"Using device: {config['_device']}", flush=True)

    if config["smoke_test"]:
        config["r_values"] = [2, 4]
        config["epochs"] = 3
        config["samples_per_config"] = 40
        config["topologies"] = config["topologies"][:1]

    os.makedirs(config["outdir"], exist_ok=True)
    csv_path = config["csv_path"] or os.path.join(config["outdir"], "results.csv")

    rows = []
    for topology in config["topologies"]:
        for r in config["r_values"]:
            for variant in config["variants"]:
                print(f"=== {topology} r={r} variant={variant} ===", flush=True)
                t0 = time.time()
                result = train_one_config(topology, r, variant, config)
                result["train_time_s"] = time.time() - t0
                print(
                    f"  realized_r={result['r']} euclid={result['euclid']:.2f} "
                    f"error={result['val_error']:.3f} jacobian={result['jacobian_sensitivity']:.4g} "
                    f"mi={result['mi_lower_bound']:.4f} ({result['train_time_s']:.1f}s)",
                    flush=True,
                )
                rows.append(result)

    fieldnames = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {csv_path}")

    from stage_1.experiments.neurreps.plot_oversquashing_metrics import plot_oversquashing_metrics
    plot_oversquashing_metrics(csv_path, config["outdir"])


if __name__ == "__main__":
    main()
