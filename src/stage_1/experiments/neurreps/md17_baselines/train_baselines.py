from __future__ import annotations

import argparse
import csv
import os
from typing import Callable, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

torch.serialization.add_safe_globals([slice])

from stage_1.md17_dataset import MOLECULE_SMILES, _read_split_indices, get_pyg_md17_class

KCAL_PER_MOL_TO_EV = 0.0433641
EV_TO_MEV = 1000.0

ATOMIC_NUMBER_VOCAB = 95  # DimeNet++'s fixed Embedding(95, ...) table; SchNet/NequIP/MACE all fine with real Z too
MD17_ELEMENTS = [1, 6, 7, 8]  # H, C, N, O --> every element across all 10 rMD17 molecules (see MOLECULE_SMILES)


class RawMD17(Dataset):
    def __init__(self, molecule: str, root: str, split_idx: int, train: bool, energy_shift: "float | None" = None):
        PyGMD17 = get_pyg_md17_class()

        full = PyGMD17(root=root, name=f"revised {molecule}")
        indices = _read_split_indices(root, split_idx, train)
        self._data = [full[i] for i in indices]

        if energy_shift is None:
            energy_shift = float(
                torch.tensor([float(d.energy) * KCAL_PER_MOL_TO_EV for d in self._data]).mean()
            )
        self.energy_shift = energy_shift

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int):
        d = self._data[idx]
        return (
            d.z, d.pos.float(), float(d.energy) * KCAL_PER_MOL_TO_EV - self.energy_shift,
            d.force.float() * KCAL_PER_MOL_TO_EV,
        )

    def energy_force_stats(self) -> Tuple[float, float, float]:
        energies = torch.tensor(
            [float(d.energy) * KCAL_PER_MOL_TO_EV - self.energy_shift for d in self._data]
        )
        n_atoms = self._data[0].z.shape[0]
        force_sumsq = torch.tensor(
            [float((d.force * KCAL_PER_MOL_TO_EV).pow(2).sum()) for d in self._data]
        )
        return (
            float(energies.mean()), float(energies.std()),
            float((force_sumsq / (n_atoms * 3)).mean().sqrt()),
        )


def raw_collate(batch: List[Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]]):
    zs, poss, energies, forces = zip(*batch)
    batch_idx = torch.cat([torch.full((z.shape[0],), i, dtype=torch.long) for i, z in enumerate(zs)])
    return {
        "z": torch.cat(zs, dim=0),
        "pos": torch.cat(poss, dim=0),
        "batch": batch_idx,
        "energy": torch.tensor(energies, dtype=torch.float32),
        "forces": torch.cat(forces, dim=0),
        "num_graphs": len(batch),
    }


ModelFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] 


def compute_energy_and_forces(
    model_fn: ModelFn, z: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor, create_graph: bool,
):
    pos = pos.clone().requires_grad_(True)
    energy = model_fn(z, pos, batch)
    (grad_pos,) = torch.autograd.grad(energy.sum(), pos, create_graph=create_graph)
    return energy, -grad_pos


def loss_fn(energy, energy_target, forces, forces_target, lambda1: float, lambda2: float):
    energy_term = lambda1 * (energy - energy_target).pow(2).mean()
    force_term = lambda2 * (forces - forces_target).pow(2).sum(dim=-1).mean()
    return energy_term + force_term, energy_term, force_term


class _DenseRadiusInteractionGraph(nn.Module):
    def __init__(self, cutoff: float, max_num_neighbors: int = 32):
        
        super().__init__()
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors

    def forward(self, pos: torch.Tensor, batch: torch.Tensor):
        from stage_1.experiments.neurreps.oversquashing.dimenet_baseline import radius_graph
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch, max_num_neighbors=self.max_num_neighbors)
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        return edge_index, edge_weight


def build_schnet(config: dict) -> Tuple[nn.Module, ModelFn]:
    from torch_geometric.nn.models import SchNet

    model = SchNet(
        hidden_channels=config["hidden_dim"], num_filters=config["hidden_dim"],
        num_interactions=config["num_layers"], num_gaussians=50,
        cutoff=config["cutoff"], readout="add",
        interaction_graph=_DenseRadiusInteractionGraph(cutoff=config["cutoff"]),
    )

    def model_fn(z, pos, batch):
        return model(z, pos, batch).squeeze(-1)

    return model, model_fn


def build_dimenet(config: dict) -> Tuple[nn.Module, ModelFn]:
    from torch_geometric.nn.models import DimeNetPlusPlus
    from stage_1.experiments.neurreps.oversquashing.dimenet_baseline import radius_graph, _triplets

    dimenet = DimeNetPlusPlus(
        hidden_channels=config["hidden_dim"], out_channels=1, num_blocks=config["num_layers"],
        int_emb_size=16, basis_emb_size=8, out_emb_channels=config["hidden_dim"],
        num_spherical=5, num_radial=6, cutoff=config["cutoff"], max_num_neighbors=64,
    )

    def model_fn(z, pos, batch):
        d = dimenet
        edge_index = radius_graph(pos, r=d.cutoff, batch=batch, max_num_neighbors=d.max_num_neighbors)
        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = _triplets(edge_index, num_nodes=z.size(0))

        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()
        pos_jk, pos_ij = pos[idx_j] - pos[idx_k], pos[idx_i] - pos[idx_j]
        a = (pos_ij * pos_jk).sum(dim=-1)
        b = torch.cross(pos_ij, pos_jk, dim=1).norm(dim=-1)
        angle = torch.atan2(b, a)

        rbf = d.rbf(dist)
        sbf = d.sbf(dist, angle, idx_kj)

        h = d.emb(z, rbf, i, j)
        P = d.output_blocks[0](h, rbf, i, num_nodes=pos.size(0))
        for interaction_block, output_block in zip(d.interaction_blocks, d.output_blocks[1:]):
            h = interaction_block(h, rbf, sbf, idx_kj, idx_ji)
            P = P + output_block(h, rbf, i, num_nodes=pos.size(0))

        energy_per_graph = torch.zeros(int(batch.max()) + 1, device=pos.device, dtype=P.dtype)
        energy_per_graph.scatter_add_(0, batch, P.squeeze(-1))
        return energy_per_graph

    return dimenet, model_fn


def build_nequip(config: dict) -> Tuple[nn.Module, ModelFn]:
    from nequip.data import AtomicDataDict
    from nequip.model import NequIPGNNModel
    from nequip.utils.global_state import set_global_state
    from stage_1.experiments.neurreps.oversquashing.dimenet_baseline import radius_graph

    prev_default_dtype = torch.get_default_dtype()
    set_global_state()
    try:
        backbone = NequIPGNNModel(
            seed=0, model_dtype="float32",
            type_names=[f"Z{z}" for z in range(ATOMIC_NUMBER_VOCAB)],
            r_max=config["cutoff"], num_layers=config["num_layers"], l_max=1, parity=True,
            num_features=config["hidden_dim"], radial_mlp_depth=1, radial_mlp_width=config["hidden_dim"],
            avg_num_neighbors=config.get("avg_num_neighbors", 10.0),
            per_type_energy_scales=1.0, per_type_energy_shifts=0.0, do_derivatives=False,
        )
    finally:
        torch.set_default_dtype(prev_default_dtype)

    def model_fn(z, pos, batch):
        row, col = radius_graph(pos, r=config["cutoff"], batch=batch, max_num_neighbors=64)
        edge_index = torch.stack([row, col], dim=0)
        edge_vectors = pos[col] - pos[row]
        data = {
            AtomicDataDict.POSITIONS_KEY: pos,
            AtomicDataDict.EDGE_INDEX_KEY: edge_index,
            AtomicDataDict.EDGE_VECTORS_KEY: edge_vectors,
            AtomicDataDict.ATOM_TYPE_KEY: z,
            AtomicDataDict.BATCH_KEY: batch,
            AtomicDataDict.NUM_NODES_KEY: torch.bincount(batch, minlength=int(batch.max()) + 1),
        }
        out = backbone(data)
        return out[AtomicDataDict.TOTAL_ENERGY_KEY].squeeze(-1)

    return backbone, model_fn


def build_allegro(config: dict) -> Tuple[nn.Module, ModelFn]:
    from allegro.model import AllegroModel
    from nequip.data import AtomicDataDict
    from nequip.utils.global_state import set_global_state
    from stage_1.experiments.neurreps.oversquashing.dimenet_baseline import radius_graph

    prev_default_dtype = torch.get_default_dtype()
    set_global_state()
    try:
        backbone = AllegroModel(
            seed=0, model_dtype="float32",
            type_names=[f"Z{z}" for z in range(ATOMIC_NUMBER_VOCAB)],
            r_max=config["cutoff"], num_layers=config["num_layers"], l_max=1, parity=True,
            radial_chemical_embed={
                "_target_": "allegro.nn.TwoBodyBesselScalarEmbed",
                "num_bessels": 8, "bessel_trainable": False, "polynomial_cutoff_p": 6,
            },
            num_scalar_features=config["hidden_dim"], num_tensor_features=config.get("num_tensor_features", 16),
            avg_num_neighbors=config.get("avg_num_neighbors", 10.0),
            per_type_energy_scales=1.0, per_type_energy_shifts=0.0, do_derivatives=False,
        )
    finally:
        torch.set_default_dtype(prev_default_dtype)

    def model_fn(z, pos, batch):
        row, col = radius_graph(pos, r=config["cutoff"], batch=batch, max_num_neighbors=64)
        edge_index = torch.stack([row, col], dim=0)
        edge_vectors = pos[col] - pos[row]
        data = {
            AtomicDataDict.POSITIONS_KEY: pos,
            AtomicDataDict.EDGE_INDEX_KEY: edge_index,
            AtomicDataDict.EDGE_VECTORS_KEY: edge_vectors,
            AtomicDataDict.ATOM_TYPE_KEY: z,
            AtomicDataDict.BATCH_KEY: batch,
            AtomicDataDict.NUM_NODES_KEY: torch.bincount(batch, minlength=int(batch.max()) + 1),
        }
        out = backbone(data)
        return out[AtomicDataDict.TOTAL_ENERGY_KEY].squeeze(-1)

    return backbone, model_fn


def build_mace(config: dict) -> Tuple[nn.Module, ModelFn]:
    import numpy as np
    from e3nn import o3
    from mace.data import AtomicData, Configuration
    from mace.modules import ScaleShiftMACE, interaction_classes
    from mace.modules.wrapper_ops import CuEquivarianceConfig
    from mace.tools.torch_geometric.batch import Batch
    from mace.tools.utils import AtomicNumberTable

    z_table = AtomicNumberTable(MD17_ELEMENTS)
    device = config.get("_device", torch.device("cpu"))
    cueq_config = CuEquivarianceConfig(
        enabled=True, layout="ir_mul", group="O3_e3nn", optimize_all=True,
        conv_fusion=(device.type == "cuda"),
    )
    model = ScaleShiftMACE(
        r_max=config["cutoff"], num_bessel=8, num_polynomial_cutoff=5,
        max_ell=2, interaction_cls=interaction_classes["RealAgnosticResidualInteractionBlock"],
        interaction_cls_first=interaction_classes["RealAgnosticInteractionBlock"],
        num_interactions=config["num_layers"], num_elements=len(MD17_ELEMENTS),
        hidden_irreps=o3.Irreps(f"{config['hidden_dim']}x0e + {config['hidden_dim']}x1o"),
        MLP_irreps=o3.Irreps(f"{config['hidden_dim']}x0e"),
        atomic_energies=np.zeros(len(MD17_ELEMENTS), dtype=np.float32),
        avg_num_neighbors=config.get("avg_num_neighbors", 10.0),
        atomic_numbers=MD17_ELEMENTS, correlation=3, gate=torch.nn.functional.silu,
        atomic_inter_scale=1.0, atomic_inter_shift=0.0, cueq_config=cueq_config,
    )

    def _graph_for(z_i: torch.Tensor, pos_i: torch.Tensor):
        config_i = Configuration(
            atomic_numbers=z_i.detach().cpu().numpy().astype(int),
            positions=pos_i.detach().cpu().numpy().astype(np.float64),
            properties={}, property_weights={},
            cell=np.zeros((3, 3)), pbc=(False, False, False),
        )
        return AtomicData.from_config(config_i, z_table=z_table, cutoff=config["cutoff"])

    def model_fn(z, pos, batch):
        num_graphs = int(batch.max().item()) + 1
        data_list = [_graph_for(z[batch == g], pos[batch == g]) for g in range(num_graphs)]
        mace_batch = Batch.from_data_list(data_list).to(pos.device)
        mace_batch.positions = pos
        out = model(mace_batch.to_dict(), training=model.training, compute_force=False)
        return out["energy"]

    return model, model_fn


MODEL_BUILDERS = {
    "schnet": build_schnet, "dimenet": build_dimenet, "nequip": build_nequip,
    "allegro": build_allegro, "mace": build_mace,
}


def train_one(model_name: str, config: dict, device: torch.device, seed: int) -> dict:
    torch.manual_seed(seed)
    config["_device"] = device  
    train_ds = RawMD17(config["molecule"], config["data_root"], config["split_idx"], train=True)
    test_ds = RawMD17(
        config["molecule"], config["data_root"], config["split_idx"], train=False,
        energy_shift=train_ds.energy_shift,
    )
    energy_mean, energy_std, force_rms = train_ds.energy_force_stats()

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, collate_fn=raw_collate)
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False, collate_fn=raw_collate)

    model, model_fn = MODEL_BUILDERS[model_name](config)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4, fused=(device.type == "cuda"))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=config.get("plateau_factor", 0.9),
        patience=config.get("plateau_patience", 50), min_lr=config.get("min_lr", 1e-6),
    )

    model.train()
    for epoch in range(config["epochs"]):
        epoch_loss, n_batches = 0.0, 0
        for batch in train_loader:
            z, pos = batch["z"].to(device), batch["pos"].to(device)
            b, E, F = batch["batch"].to(device), batch["energy"].to(device), batch["forces"].to(device)
            optimizer.zero_grad()
            U, F_pred = compute_energy_and_forces(model_fn, z, pos, b, create_graph=True)
            loss, e_term, f_term = loss_fn(
                (U - energy_mean) / energy_std, (E - energy_mean) / energy_std,
                F_pred / force_rms, F / force_rms, config["lambda1"], config["lambda2"],
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        scheduler.step(epoch_loss / max(n_batches, 1))
        if config.get("log_every") and (epoch % config["log_every"] == 0):
            print(f"  [{model_name} seed={seed}] epoch={epoch} loss={loss.item():.4f}", flush=True)

    model.eval()
    total_e_ae, total_f_ae, total_graphs, total_f_components = 0.0, 0.0, 0, 0
    for batch in test_loader:
        z, pos = batch["z"].to(device), batch["pos"].to(device)
        b, E, F = batch["batch"].to(device), batch["energy"].to(device), batch["forces"].to(device)
        U, F_pred = compute_energy_and_forces(model_fn, z, pos, b, create_graph=False)
        total_e_ae += (U.detach() - E).abs().sum().item()
        total_f_ae += (F_pred.detach() - F).abs().sum().item()
        total_graphs += b.max().item() + 1
        total_f_components += F.numel()

    energy_mae_mev = (total_e_ae / total_graphs) * EV_TO_MEV
    force_mae_mev = (total_f_ae / total_f_components) * EV_TO_MEV
    return {"energy_mae_mev": energy_mae_mev, "force_mae_mev_per_angstrom": force_mae_mev}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_BUILDERS))
    parser.add_argument("--molecule", type=str, required=True, choices=sorted(MOLECULE_SMILES))
    parser.add_argument("--split-idx", type=int, default=0, choices=[0, 1, 2, 3, 4])
    parser.add_argument("--data-root", type=str, default="data/rmd17")
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--avg-num-neighbors", type=float, default=10.0, help="[nequip/allegro/mace only]")
    parser.add_argument("--num-tensor-features", type=int, default=16, help="[allegro only]")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--lambda1", type=float, default=1.0)
    parser.add_argument("--lambda2", type=float, default=1000.0)
    parser.add_argument("--min-lr", type=float, default=1e-6, help="ReduceLROnPlateau's LR floor")
    parser.add_argument("--plateau-factor", type=float, default=0.9, help="ReduceLROnPlateau's multiplicative LR reduction on a detected plateau")
    parser.add_argument("--plateau-patience", type=int, default=50, help="Epochs with no train-loss improvement before ReduceLROnPlateau reduces LR")
    parser.add_argument("--num-realizations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="stage_1/experiments/neurreps/md17_baselines/results")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    config = vars(args)

    if config["smoke_test"]:
        config["epochs"] = 1
        config["batch_size"] = 4
        config["hidden_dim"] = 8
        config["num_layers"] = 1
        config["num_realizations"] = 1

    device = torch.device(config["device"]) if config["device"] else (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    print(f"Using device: {device}", flush=True)

    os.makedirs(config["outdir"], exist_ok=True)
    csv_path = os.path.join(config["outdir"], "results.csv")
    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["model", "molecule", "split_idx", "realization", "seed", "energy_mae_mev", "force_mae_mev_per_angstrom"])
        for i in range(config["num_realizations"]):
            seed = config["seed"] + i * 7919
            result = train_one(config["model"], config, device, seed)
            writer.writerow([
                config["model"], config["molecule"], config["split_idx"], i, seed,
                result["energy_mae_mev"], result["force_mae_mev_per_angstrom"],
            ])
            fcsv.flush()
            print(
                f"model={config['model']} molecule={config['molecule']} split={config['split_idx']} "
                f"realization={i} seed={seed} "
                f"energy_mae={result['energy_mae_mev']:.2f}meV "
                f"force_mae={result['force_mae_mev_per_angstrom']:.2f}meV/A", flush=True,
            )


if __name__ == "__main__":
    main()
