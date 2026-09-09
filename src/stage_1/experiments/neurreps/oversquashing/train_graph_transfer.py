from __future__ import annotations

import argparse
import csv
import functools
import math
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from utils.se3_utils import RadialNetwork

from stage_1.experiments.neurreps.oversquashing.synthetic_dataset import (
    GRAPH_EDGE_FEAT_DIM,
    GeometricGraphTransferDataset,
    GraphTransferDataset,
    graph_transfer_collate_fn,
)
from stage_1.experiments.neurreps.oversquashing.eqfsa_classifier import EqFSAClassifier
from stage_1.experiments.neurreps.oversquashing.metrics import EmbeddingCapture, jacobian_sensitivity, mi_lower_bound


def detect_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _to_device(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def call_model(model, variant: str, batch: dict) -> torch.Tensor:
    if variant in ("dimenet", "nequip_allegro"):
        return model(batch["x"], batch["types"], batch["mask"], batch["target_idx"], edge_feats=batch["edge_feats"])
    if variant == "eqfsa_full":
        return model(
            batch["x"], batch["types"], batch["masses"], batch["mask"], batch["V"],
            batch["mol_props"], batch["edge_feats"], batch["target_idx"],
            seed_masses=batch.get("seed_masses"),
        )
    return model(
        batch["x"], batch["types"], batch["masses"], batch["mask"], batch["V"],
        batch["mol_props"], batch["edge_feats"], batch["target_idx"],
    )


def _avg_num_neighbors(positions: torch.Tensor, cutoff: float) -> float:
    
    dist = torch.cdist(positions, positions)
    n = positions.shape[0]
    within_cutoff = (dist <= cutoff) & ~torch.eye(n, dtype=torch.bool, device=positions.device)
    return max(1.0, float(within_cutoff.sum(dim=-1).float().mean().item()))


def build_model_for_variant(
    variant: str, realized_r: int, classes: int, num_atom_types: int, config: dict,
    avg_num_neighbors: float = 1.0,
):
    if variant == "dimenet":
        from stage_1.experiments.neurreps.oversquashing.dimenet_baseline import DimeNetPlusPlusClassifier
        num_layers = max(1, realized_r + config["baseline_layer_margin"])
        return DimeNetPlusPlusClassifier(
            num_node_types=num_atom_types, classes=classes, num_layers=num_layers,
            cutoff=config["cutoff"],
        )
    if variant == "nequip_allegro":
        from stage_1.experiments.neurreps.oversquashing.nequip_allegro_baseline import NequIPAllegroClassifier
        num_layers = max(1, realized_r + config["baseline_layer_margin"])
        return NequIPAllegroClassifier(
            num_node_types=num_atom_types, classes=classes, num_layers=num_layers,
            cutoff=config["cutoff"], avg_num_neighbors=avg_num_neighbors,
        )
    if variant == "eqfsa_full":
        radial_net_cls = RadialNetwork
        if config.get("eqfsa_linear_radial_readout", False):
            radial_net_cls = functools.partial(RadialNetwork, linear_readout=True)
        return EqFSAClassifier(
            radial_net=radial_net_cls, num_atom_types=num_atom_types, max_degree=2,
            num_layers=config["eqfsa_num_layers"], feature_dim=config["feature_dim"],
            hidden_dim=config["hidden_dim"], edge_cutoff=config["cutoff"], n_max=1,
            classes=classes, edge_feat_dim=GRAPH_EDGE_FEAT_DIM,
            normalize_z_by_n_real=config.get("eqfsa_normalize_z_by_n_real", True),
            content_weighted_com=config.get("eqfsa_content_weighted_com", False),
            per_layer_fusion=config.get("eqfsa_per_layer_fusion", True),
            imaginary_only_ri=config.get("eqfsa_imaginary_only_ri", False),
            decouple_ri_norm=config.get("eqfsa_decouple_ri_norm", True),
            reduce_norms=config.get("eqfsa_reduce_norms", True),
        )
    if variant in ("mean_pool", "attn_pool"):
        from stage_1.experiments.neurreps.oversquashing.trivial_pool_classifier import TrivialPoolClassifier
        return TrivialPoolClassifier(
            num_atom_types=num_atom_types, classes=classes, embed_dim=config["feature_dim"],
            hidden_dim=config["hidden_dim"], pool="mean" if variant == "mean_pool" else "attn",
        )
    if variant == "rsa":
        from stage_1.experiments.neurreps.oversquashing.rsa_classifier import RSAClassifier
        return RSAClassifier(
            num_atom_types=num_atom_types, classes=classes, feature_dim=config["feature_dim"],
            hidden_dim=config["hidden_dim"], n_max=config["rsa_n_max"], sigma=config["rsa_sigma"],
        )
    raise ValueError(f"Unknown variant {variant!r}")


def build_optimizer(model: torch.nn.Module, lr: float, qk_gamma_wd: float, base_weight_decay: float = 1e-4):
    decay_params, other_params = [], []
    for name, p in model.named_parameters():
        if (name.endswith("w_vq_norm.gamma") or name.endswith("w_vk_norm.gamma")
                or ".master_attn.W_QK." in name
                or name.endswith("s_ri_norm.weight")):
            decay_params.append(p)
        else:
            other_params.append(p)
    return torch.optim.AdamW([
        {"params": other_params, "weight_decay": base_weight_decay},
        {"params": decay_params, "weight_decay": qk_gamma_wd},
    ], lr=lr)


def train_one_realization(
    topology: str, r: int, variant: str, config: dict, seed: int, weight_seed: int | None = None,
) -> dict:
    classes = config["classes"]
    samples = config["samples_per_config"]

    if config["task"] == "geometric":
        dataset = None
        last_err = None
        for attempt in range(config["geometric_seed_retries"]):
            trial_seed = seed + attempt * 1000003
            try:
                dataset = GeometricGraphTransferDataset(
                    topology, r, classes, samples, seed=trial_seed, cutoff=config["cutoff"],
                    class_spacing=config["geometric_class_spacing"],
                )
                seed = trial_seed
                break
            except RuntimeError as e:
                last_err = e
        if dataset is None:
            raise RuntimeError(
                f"GeometricGraphTransferDataset: no safe geometry found for topology={topology!r} r={r} "
                f"after {config['geometric_seed_retries']} seed attempts starting from {seed}. "
                f"Last error: {last_err}"
            )
    else:
        dataset = GraphTransferDataset(topology, r, classes, samples, seed=seed, cutoff=config["cutoff"])
    realized_r = dataset.realized_r
    realized_euclid = dataset.realized_euclid
    label_idx = dataset.label_idx
    num_atom_types = dataset.num_atom_types

    n_val = max(1, int(round(len(dataset) * config["val_fraction"])))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed + 100000),
    )
    train_loader = DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True, collate_fn=graph_transfer_collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["batch_size"], shuffle=False, collate_fn=graph_transfer_collate_fn,
    )

    device = config["_device"]
    avg_num_neighbors = _avg_num_neighbors(dataset.positions, config["cutoff"])
    torch.manual_seed(seed if weight_seed is None else weight_seed)
    model = build_model_for_variant(
        variant, realized_r, classes, num_atom_types, config, avg_num_neighbors=avg_num_neighbors,
    ).to(device)
    embedding_module = model.input_embedding_module
    optimizer = build_optimizer(model, config["lr"], config.get("qk_gamma_wd", 0.0))

    train_model = model
    if config.get("torch_compile") and variant == "eqfsa_full":
        train_model = torch.compile(model, dynamic=True)

    log_every = config.get("log_every", 0)
    epoch_bar = tqdm(
        range(config["epochs"]), desc=f"{variant} {topology} r={r} seed={seed}",
        leave=False, unit="epoch",
    )
    for epoch in epoch_bar:
        model.train()
        epoch_loss_sum, epoch_grad_norm_sum, epoch_batches = 0.0, 0.0, 0
        for batch in train_loader:
            batch = _to_device(batch, device)
            optimizer.zero_grad()
            logits = call_model(train_model, variant, batch)
            loss = F.cross_entropy(logits, batch["labels"])
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss_sum += loss.item()
            epoch_grad_norm_sum += float(grad_norm)
            epoch_batches += 1
        avg_loss = epoch_loss_sum / max(1, epoch_batches)
        avg_grad_norm = epoch_grad_norm_sum / max(1, epoch_batches)
        epoch_bar.set_postfix(loss=f"{avg_loss:.4f}", grad_norm=f"{avg_grad_norm:.4f}")
        if log_every and (epoch % log_every == 0 or epoch == config["epochs"] - 1):
            tqdm.write(
                f"    [{variant} {topology} r={r} seed={seed}] epoch={epoch} loss={avg_loss:.4f} "
                f"grad_norm={avg_grad_norm:.6f}",
            )

    model.eval()
    total_ce, total_correct, total_n = 0.0, 0, 0
    sens_values = []
    for batch in val_loader:
        batch = _to_device(batch, device)
        B = batch["labels"].shape[0]
        label_idx_t = torch.full((B,), label_idx, dtype=torch.long, device=device)
        with EmbeddingCapture(embedding_module) as cap:
            logits = call_model(model, variant, batch)
            sens_values.append(jacobian_sensitivity(logits, cap.captured, label_idx_t))
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
        "r": realized_r,
        "euclid": realized_euclid,
        "val_error": val_error,
        "val_ce": val_ce,
        "jacobian_sensitivity": val_sensitivity,
        "mi_lower_bound": val_mi,
        "n_params": n_params,
    }


def _mean_std(values: list[float]) -> tuple[float, float]:
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / max(1, n - 1)
    return mean, math.sqrt(var)


def run_config(
    topology: str, r: int, variant: str, config: dict,
    raw_writer: "csv.DictWriter | None" = None, raw_file=None,
) -> dict:
    base_seed = config["seed"]
    explicit_seeds = config.get("seeds")
    if explicit_seeds:
        seeds = list(explicit_seeds)
    else:
        seeds = [base_seed + i * 7919 for i in range(config["num_realizations"])]
    explicit_weight_seeds = config.get("weight_seeds")
    if explicit_weight_seeds:
        assert len(explicit_weight_seeds) == len(seeds), (
            f"--weight-seeds ({len(explicit_weight_seeds)}) must match the number of data seeds "
            f"({len(seeds)}) -- one weight-init seed per realization."
        )
        weight_seeds = list(explicit_weight_seeds)
    else:
        weight_seeds = [None] * len(seeds)
    keys = ["r", "euclid", "val_error", "jacobian_sensitivity", "mi_lower_bound", "n_params"]
    per_realization = {k: [] for k in keys}

    realization_bar = tqdm(
        list(enumerate(zip(seeds, weight_seeds))), desc=f"{variant} {topology} r={r} realizations",
        unit="realization",
    )
    for i, (seed, weight_seed) in realization_bar:
        result = train_one_realization(topology, r, variant, config, seed=seed, weight_seed=weight_seed)
        for k in keys:
            per_realization[k].append(result[k])
        if raw_writer is not None:
            raw_row = {
                "topology": topology, "variant": variant, "nominal_r": r,
                "realization_idx": i, "seed": seed,
                "weight_seed": seed if weight_seed is None else weight_seed,
                **{k: result[k] for k in keys},
            }
            raw_writer.writerow(raw_row)
            raw_file.flush()
            os.fsync(raw_file.fileno())

    row = {"topology": topology, "variant": variant, "nominal_r": r, "num_realizations": len(seeds)}
    for k in keys:
        mean, std = _mean_std(per_realization[k])
        row[k] = mean
        row[f"{k}_std"] = std
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topologies", type=str, nargs="+", default=["chain", "ring", "lollipop"])
    parser.add_argument("--r-values", type=int, nargs="+", default=[2, 4, 6, 8, 10, 12, 16, 20])
    parser.add_argument("--variants", type=str, nargs="+", default=["dimenet", "nequip_allegro", "eqfsa_full"])
    parser.add_argument(
        "--task", type=str, choices=["type", "geometric"], default="type",
        help="'type': class is the label atom's discrete type (original benchmark). "
             "'geometric': class is the label atom's real distance from the readout atom along a "
             "fixed axis, with its type held constant -- mean_pool/attn_pool are structurally "
             "capped at chance since they never see positions.",
    )
    parser.add_argument("--geometric-class-spacing", type=float, default=None,
        help="--task geometric only: distance between adjacent classes' target distances",
    )
    parser.add_argument(
        "--geometric-seed-retries", type=int, default=20,
        help="--task geometric only; number of alternate seeds to try for faulty realization's random-walk",
    )
    parser.add_argument("--classes", type=int, default=5)
    parser.add_argument("--eqfsa-num-layers", type=int, default=4, help="Fixed depth for Eq-FSA, independent of r")
    parser.add_argument(
        "--baseline-layer-margin", type=int, default=4,
        help="DimeNet++/NequIP-Allegro depth = realized_r + this marginp; rules out underreaching",
    )
    parser.add_argument("--feature-dim", type=int, default=24)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument(
        "--rsa-sigma", type=float, default=5.0,
        help="'rsa' variant only; Gaussian mode-weighting bandwidth for RSAAttention (matches lr_sr's default)",
    )
    parser.add_argument(
        "--rsa-n-max", type=int, default=2,
        help="'rsa' variant only: max Fourier mode index for RSAAttention's wavevector grid",
    )
    parser.add_argument("--no-eqfsa-normalize-z-by-n-real", dest="eqfsa_normalize_z_by_n_real", action="store_false",)
    parser.add_argument("--eqfsa-linear-radial-readout", action="store_true",)
    parser.add_argument("--eqfsa-content-weighted-com", action="store_true")
    parser.add_argument("--no-eqfsa-per-layer-fusion", dest="eqfsa_per_layer_fusion", action="store_false")
    parser.add_argument("--eqfsa-imaginary-only-ri", action="store_true")
    parser.add_argument("--no-eqfsa-decouple-ri-norm", dest="eqfsa_decouple_ri_norm", action="store_false")
    parser.add_argument("--no-eqfsa-reduce-norms", dest="eqfsa_reduce_norms", action="store_false")
    parser.add_argument("--qk-gamma-wd", type=float, default=0.0,)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--samples-per-config", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-realizations", type=int, default=3,)
    parser.add_argument("--seeds", type=int, nargs="+", default=None,)
    parser.add_argument("--weight-seeds", type=int, nargs="+", default=None,)
    parser.add_argument("--log-every", type=int, default=0,)
    parser.add_argument("--outdir", type=str, default="stage_1/experiments/neurreps/results")
    parser.add_argument("--csv-path", type=str, default=None)
    parser.add_argument("--device", type=str, default=None,)
    parser.add_argument("--smoke-test", action="store_true",)
    parser.add_argument("--torch-compile", action="store_true",)
    args = parser.parse_args()
    config = vars(args)
    config["_device"] = detect_device(config["device"])
    print(f"Using device: {config['_device']}", flush=True)

    if "rsa" in config["variants"] and config["task"] != "geometric":
        raise ValueError("'rsa' is scoped to --task geometric only")

    if config["smoke_test"]:
        config["r_values"] = [6, 8] if config["task"] == "geometric" else [2, 4]
        config["epochs"] = 3
        config["samples_per_config"] = 40
        config["topologies"] = config["topologies"][:1]
        config["num_realizations"] = 2

    os.makedirs(config["outdir"], exist_ok=True)
    csv_path = config["csv_path"] or os.path.join(config["outdir"], "results.csv")
    raw_csv_path = os.path.join(config["outdir"], "results_raw.csv")
    raw_keys = ["r", "euclid", "val_error", "jacobian_sensitivity", "mi_lower_bound", "n_params"]
    raw_fieldnames = ["topology", "variant", "nominal_r", "realization_idx", "seed", "weight_seed"] + raw_keys

    rows = []
    writer = None
    with open(csv_path, "w", newline="") as csv_file, open(raw_csv_path, "w", newline="") as raw_file:
        raw_writer = csv.DictWriter(raw_file, fieldnames=raw_fieldnames)
        raw_writer.writeheader()
        for topology in config["topologies"]:
            for r in config["r_values"]:
                for variant in config["variants"]:
                    n_realizations_display = len(config["seeds"]) if config.get("seeds") else config["num_realizations"]
                    print(f"=== {topology} r={r} variant={variant} ({n_realizations_display} realizations) ===", flush=True)
                    t0 = time.time()
                    result = run_config(topology, r, variant, config, raw_writer=raw_writer, raw_file=raw_file)
                    result["train_time_s"] = time.time() - t0
                    print(
                        f"  realized_r={result['r']:.1f}+/-{result['r_std']:.1f} "
                        f"euclid={result['euclid']:.2f}+/-{result['euclid_std']:.2f} "
                        f"error={result['val_error']:.3f}+/-{result['val_error_std']:.3f} "
                        f"jacobian={result['jacobian_sensitivity']:.4g}+/-{result['jacobian_sensitivity_std']:.4g} "
                        f"mi={result['mi_lower_bound']:.4f}+/-{result['mi_lower_bound_std']:.4f} "
                        f"({result['train_time_s']:.1f}s)",
                        flush=True,
                    )
                    rows.append(result)
                    if writer is None:
                        writer = csv.DictWriter(csv_file, fieldnames=list(result.keys()))
                        writer.writeheader()
                    writer.writerow(result)
                    csv_file.flush()
                    os.fsync(csv_file.fileno())
    print(f"Wrote {len(rows)} rows to {csv_path}")
    print(f"Wrote raw per-realization rows to {raw_csv_path}")

    from stage_1.experiments.neurreps.oversquashing.plot_oversquashing_metrics import plot_oversquashing_metrics
    plot_oversquashing_metrics(csv_path, config["outdir"])


if __name__ == "__main__":
    main()
