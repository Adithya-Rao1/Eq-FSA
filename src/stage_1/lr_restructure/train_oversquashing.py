import argparse
import csv
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from utils.se3_utils import RadialNetwork
from stage_1.experiments.neurreps.oversquashing.synthetic_dataset import (
    GraphTransferDataset, GeometricGraphTransferDataset, graph_transfer_collate_fn,
)
from stage_1.lr_restructure.oversquashing_classifier import LRRestructureClassifier


def build_dataset(task, topology, r, classes, samples, seed, cutoff, class_spacing, max_retries=20):
    if task != "geometric":
        return GraphTransferDataset(topology, r, classes, samples, seed=seed, cutoff=cutoff)
    last_err = None
    for attempt in range(max_retries):
        try:
            return GeometricGraphTransferDataset(
                topology, r, classes, samples, seed=seed + attempt, cutoff=cutoff,
                class_spacing=class_spacing,
            )
        except RuntimeError as e:
            last_err = e
    raise last_err


def _to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if isinstance(v, torch.Tensor) else v
    return out


def _run_batch(model, batch, device):
    batch = _to_device(batch, device)
    return model(
        batch["x"], batch["types"], batch["masses"], batch["mask"], batch["V"],
        batch["mol_props"], batch["edge_feats"], batch["target_idx"],
        seed_masses=batch["seed_masses"],
    )


def train_one(topology, r, config: dict, device: torch.device) -> float:
    seed = config["seed"]
    torch.manual_seed(seed)
    dataset = build_dataset(
        config["task"], topology, r, config["classes"], config["samples_per_config"], seed,
        config["cutoff"], config["geometric_class_spacing"],
    )
    n_val = max(1, int(len(dataset) * config["val_fraction"]))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(
        train_set, batch_size=config["batch_size"], shuffle=True, collate_fn=graph_transfer_collate_fn,
    )
    val_loader = DataLoader(
        val_set, batch_size=config["batch_size"], shuffle=False, collate_fn=graph_transfer_collate_fn,
    )

    model = LRRestructureClassifier(
        radial_net=RadialNetwork, num_atom_types=dataset.num_atom_types, max_degree=2,
        num_layers=config["num_layers"], feature_dim=config["feature_dim"],
        hidden_dim=config["hidden_dim"], edge_cutoff=config["cutoff"], n_max=1,
        classes=config["classes"], n_master_nodes=config["num_master_nodes"],
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config["lr"])

    loss = None
    for epoch in range(config["epochs"]):
        model.train()
        for batch in train_loader:
            opt.zero_grad()
            logits = _run_batch(model, batch, device)
            loss = F.cross_entropy(logits, batch["labels"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad_norm"])
            opt.step()
        if config["log_every"] and epoch % config["log_every"] == 0 and loss is not None:
            print(f"[{topology} r={r}] epoch={epoch} loss={loss.item():.4f}", flush=True)

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            logits = _run_batch(model, batch, device)
            pred = logits.argmax(dim=-1)
            correct += (pred == batch["labels"].to(device)).sum().item()
            total += pred.shape[0]
    return 1.0 - correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=str, default="geometric", choices=["type", "geometric"])
    parser.add_argument("--topologies", type=str, nargs="+", default=["chain"])
    parser.add_argument("--r-values", type=int, nargs="+", default=[2, 6, 8])
    parser.add_argument("--classes", type=int, default=5)
    parser.add_argument("--samples-per-config", type=int, default=240)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--geometric-class-spacing", type=float, default=None)
    parser.add_argument("--feature-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-master-nodes", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--num-realizations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="stage_1/lr_restructure/results_oversquashing")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        args.r_values = [6] if args.task == "geometric" else [2]
        args.epochs = 2
        args.samples_per_config = 10
        args.topologies = args.topologies[:1]
        args.num_realizations = 1
        args.batch_size = 4
        args.feature_dim = 8
        args.hidden_dim = 16
        args.num_layers = 2
        args.num_master_nodes = 2

    device = torch.device(args.device) if args.device else (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    print(f"Using device: {device}", flush=True)
    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, "results.csv")

    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["topology", "r", "realization", "seed", "val_error"])
        for topology in args.topologies:
            for r in args.r_values:
                for real in range(args.num_realizations):
                    seed = args.seed + real
                    config = dict(vars(args))
                    config["seed"] = seed
                    val_error = train_one(topology, r, config, device)
                    writer.writerow([topology, r, real, seed, val_error])
                    fcsv.flush()
                    print(
                        f"{topology} r={r} realization={real} seed={seed} val_error={val_error:.4f}",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
