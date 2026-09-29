from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import DataLoader

from stage_1.train_spice import (
    SPICEEnergyForceModel,
    build_dataloaders,
    compute_energy_and_forces,
    spice_loss,
)
from utils.se3_utils import RadialNetwork

_MODEL_CONFIG_KEYS = (
    "num_atom_types", "max_degree", "num_layers", "feature_dim", "hidden_dim",
    "edge_cutoff", "n_max", "n_master_nodes", "num_fusion_heads",
)


def load_checkpoint(path: str, device: torch.device) -> dict:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    missing = [k for k in ("config", "model_state_dict") if k not in ckpt]
    if missing:
        raise KeyError(
            f"{path} is missing {missing} -- expected a train_spice.py SPICETrainer.save_checkpoint() "
            f"output (epoch/global_step/model_state_dict/optimizer_state_dict/scheduler_state_dict/"
            f"val_loss/config), not an explosion dump (use replay_batch.py for those instead)."
        )
    return ckpt


def build_model_from_config(config: dict, device: torch.device) -> SPICEEnergyForceModel:
    missing = [k for k in _MODEL_CONFIG_KEYS if k not in config]
    if missing:
        raise KeyError(f"Checkpoint config is missing required model keys: {missing}")
    model = SPICEEnergyForceModel(
        radial_net=RadialNetwork,
        num_atom_types=config["num_atom_types"],
        max_degree=config["max_degree"],
        num_layers=config["num_layers"],
        feature_dim=config["feature_dim"],
        hidden_dim=config["hidden_dim"],
        edge_cutoff=config["edge_cutoff"],
        n_max=config["n_max"],
        pw_e_cutoff=config.get("pw_e_cutoff"),
        n_master_nodes=config.get("n_master_nodes", 4),
        backbone=config.get("backbone", "production"),
        num_fusion_heads=config.get("num_fusion_heads", 4),
        ablate_max_degree=config.get("ablate_max_degree"),
        num_broadcast_heads=config.get("num_broadcast_heads", 4),
        rbf_num_basis=config.get("rbf_num_basis", 12),
        rbf_min=config.get("rbf_min", 0.3),
        rbf_cutoff=config.get("rbf_cutoff", 25.0),
        sigma=config.get("sigma"),
        alpha_floor=config.get("alpha_floor", 1e-4),
        use_norms=config.get("use_norms", True),
    ).to(device)
    return model


def load_test_split(split_path: str) -> dict:
    return torch.load(split_path, map_location="cpu", weights_only=False)


def build_test_dataset_from_split(split: dict, hdf5_paths_override: Optional[list[str]] = None):
    from torch.utils.data import Subset

    from stage_1.spice_dataset import SPICEDataset

    hdf5_paths = hdf5_paths_override if hdf5_paths_override is not None else split["hdf5_paths"]
    dataset = SPICEDataset(
        hdf5_paths=hdf5_paths, subsets=split.get("subsets"), exclude_solvated=True,
        box_padding=split["box_padding"], edge_cutoff=split["edge_cutoff"],
    )
    test_identities = split["test_identities"]
    indices = dataset.indices_for_identities(test_identities)
    if len(indices) != len(test_identities):
        print(
            f"[checkpoint_utils] WARNING: found {len(indices)}/{len(test_identities)} saved "
            f"test-set molecules in the reconstructed dataset "
            f"(hdf5_paths_override={'set' if hdf5_paths_override else 'not set'}) -- proceeding "
            f"with whatever matched. A mismatch usually means the hdf5 files changed since the "
            f"split was saved."
        )
    return Subset(dataset, indices)


def load_model_and_val_loader(
    ckpt_path: str,
    device: torch.device,
    hdf5_paths_override: Optional[list[str]] = None,
    batch_size_override: Optional[int] = None,
    num_workers_override: Optional[int] = None,
) -> tuple[SPICEEnergyForceModel, DataLoader, dict, float, float, float, dict]:
    ckpt = load_checkpoint(ckpt_path, device)
    config = dict(ckpt["config"])
    if hdf5_paths_override is not None:
        config["hdf5_paths"] = hdf5_paths_override
    if batch_size_override is not None:
        config["batch_size"] = batch_size_override
    if num_workers_override is not None:
        config["num_workers"] = num_workers_override

    _, val_loader, energy_mean, energy_std, force_rms = build_dataloaders(config, device=device)

    model = build_model_from_config(config, device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    return model, val_loader, config, energy_mean, energy_std, force_rms, ckpt


def _batch_to_device(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def _accumulate_errors(U: torch.Tensor, F: torch.Tensor, batch: dict) -> tuple[float, float, int, int]:
    mask = batch["all_solute_mask"].bool()
    energy_abs_err = (U.detach() - batch["energy"]).abs().sum().item()
    force_abs_err = ((F.detach() - batch["solute_forces"]).abs() * mask.unsqueeze(-1)).sum().item()
    return energy_abs_err, force_abs_err, U.shape[0], int(mask.sum().item())


def evaluate_validation(
    model: SPICEEnergyForceModel,
    val_loader: DataLoader,
    device: torch.device,
    energy_mean: float,
    energy_std: float,
    force_rms: float,
    lambda1: float,
    lambda2: float,
) -> dict:
    model.eval()
    n_batches = 0
    sum_loss = sum_energy_term = sum_force_term = 0.0
    sum_energy_abs_err = sum_force_abs_err = 0.0
    n_molecules = n_atoms = 0

    for batch in val_loader:
        batch = _batch_to_device(batch, device)
        U, F = compute_energy_and_forces(
            model, batch["all_solute_x"], batch["all_solute_types"], batch["all_solute_masses"],
            batch["all_solute_mask"], batch["V"], batch["mol_props"], batch["all_solute_edge_feats"],
        )

        U_n = (U - energy_mean) / energy_std
        U_target_n = (batch["energy"] - energy_mean) / energy_std
        F_n = F / force_rms
        F_target_n = batch["solute_forces"] / force_rms
        loss, energy_term, force_term = spice_loss(
            U_n, U_target_n, F_n, F_target_n, batch["all_solute_mask"], lambda1, lambda2,
        )

        sum_loss += float(loss.item())
        sum_energy_term += float(energy_term.item())
        sum_force_term += float(force_term.item())
        n_batches += 1

        e_err, f_err, n_mol, n_at = _accumulate_errors(U, F, batch)
        sum_energy_abs_err += e_err
        sum_force_abs_err += f_err
        n_molecules += n_mol
        n_atoms += n_at

    return {
        "normalized_loss": sum_loss / max(n_batches, 1),
        "energy_term": sum_energy_term / max(n_batches, 1),
        "force_term": sum_force_term / max(n_batches, 1),
        "energy_mae_eV": sum_energy_abs_err / max(n_molecules, 1),
        "force_mae_eV_per_A": sum_force_abs_err / max(n_atoms * 3, 1),
        "n_batches": n_batches,
        "n_molecules": n_molecules,
        "n_atoms": n_atoms,
    }