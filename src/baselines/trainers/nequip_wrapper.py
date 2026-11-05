from __future__ import annotations

# nequip 0.6.1 / allegro 0.3.0:
#  - the default builders include StressForceOutput, which needs a cell/virial; this data are non-periodic, so ForceOutput is used;
#  - the default shift is the per-atom mean energy. That is poor for datasets whose composition varies so using a regression shift

import argparse
import json
import os
import subprocess
import sys

import yaml

RUN_NAME = "train"
TRAIN_DONE = "TRAIN_DONE"
DEPLOYED = "deployed.pth"

NEQUIP_MODEL = dict(
    model_builders=["SimpleIrrepsConfig", "EnergyModel", "PerSpeciesRescale", "ForceOutput", "RescaleEnergyEtc"],
    num_layers=4, l_max=2, parity=True, num_features=32, nonlinearity_type="gate", resnet=False,
    invariant_layers=2, invariant_neurons=64, num_basis=8, BesselBasis_trainable=True, PolynomialCutoff_p=6,
    avg_num_neighbors="auto",
)
ALLEGRO_MODEL = dict(
    model_builders=["allegro.model.Allegro", "PerSpeciesRescale", "ForceOutput", "RescaleEnergyEtc"],
    num_layers=2, l_max=2, parity="o3_full", num_tensor_features=64,
    radial_basis_num_bessels_per_basis=8, radial_basis_trainable=True,
    two_body_latent_mlp_latent_dimensions=[128, 256, 512, 1024], two_body_latent_mlp_nonlinearity="silu",
    two_body_latent_mlp_initialization="uniform", latent_mlp_latent_dimensions=[1024, 1024, 1024], latent_mlp_nonlinearity="silu",
    latent_mlp_initialization="uniform", latent_resnet=True, env_embed_mlp_latent_dimensions=[], env_embed_mlp_nonlinearity=None,
    env_embed_mlp_initialization="uniform", edge_eng_mlp_latent_dimensions=[128], edge_eng_mlp_nonlinearity=None,
    edge_eng_mlp_initialization="uniform", PolynomialCutoff_p=6, avg_num_neighbors="auto",
)
MODELS = {"nequip": NEQUIP_MODEL, "allegro": ALLEGRO_MODEL}


def read_split_record(data_dir: str) -> dict:
    with open(os.path.join(data_dir, "split.json")) as f:
        rec = json.load(f)
    missing = [k for k in ("species", "fixed_composition", "splits") if k not in rec]
    if missing:
        raise KeyError(f"{data_dir}/split.json lacks {missing}; re-export with the current export_splits")
    return rec


def build_config(
    model: str, data_dir: str, run_dir: str, seed: int, *, device: str = "cuda", epochs: int = 1, batch_size: int = 5, lr: float = 0.005,
    r_max: float = 5.0, forces_coeff: float = 1.0, energy_coeff: float = 1.0, model_dtype: str = "float32", ema_decay: float = 0.99,
    overrides: dict | None = None,
) -> dict:
    if model not in MODELS:
        raise ValueError(f"model must be one of {sorted(MODELS)}, got {model!r}")
    rec = read_split_record(data_dir)
    sp = rec["splits"]
    has_forces = all(sp[k]["has_forces"] for k in ("train", "val") if sp[k]["n"])
    key_mapping = {"REF_energy": "total_energy"}
    loss = {"total_energy": [energy_coeff, "PerAtomMSELoss"]}
    if has_forces:
        key_mapping["REF_forces"] = "forces"
        loss["forces"] = [forces_coeff, "MSELoss"]
    ds = lambda name: os.path.abspath(os.path.join(data_dir, f"{name}.xyz"))  # noqa: E731
    ase_args = {"format": "extxyz"}
    cfg: dict = {
        "root": os.path.abspath(run_dir), "run_name": RUN_NAME, "append": True, "seed": seed, "dataset_seed": seed,
        "device": device, "default_dtype": "float64", "model_dtype": model_dtype, "allow_tf32": False,
        "r_max": r_max, "chemical_symbols": rec["species"],
        "dataset": "ase", "dataset_file_name": ds("train"), "dataset_ase_args": ase_args,
        "dataset_include_keys": list(key_mapping), "dataset_key_mapping": key_mapping,
        "validation_dataset": "ase", "validation_dataset_file_name": ds("val"), "validation_dataset_ase_args": ase_args,
        "validation_dataset_include_keys": list(key_mapping), "validation_dataset_key_mapping": key_mapping,
        "n_train": sp["train"]["n"], "n_val": sp["val"]["n"], "train_val_split": "sequential", "shuffle": True,
        "batch_size": batch_size, "validation_batch_size": batch_size, "max_epochs": epochs,
        "learning_rate": lr, "optimizer_name": "Adam", "lr_scheduler_name": "ReduceLROnPlateau",
        "lr_scheduler_patience": 100, "lr_scheduler_factor": 0.5, "use_ema": True, "ema_decay": ema_decay,
        "loss_coeffs": loss, "metrics_key": "validation_loss", "log_epoch_freq": 1,
    }
    cfg.update(MODELS[model])
    if not has_forces:
        cfg["global_rescale_scale"] = "dataset_total_energy_std"
    if not rec["fixed_composition"]:
        cfg["per_species_rescale_shifts"] = "dataset_per_species_total_energy_mean"
        cfg["per_species_rescale_scales"] = None
    cfg.update(overrides or {})
    return cfg


def write_config(cfg: dict, run_dir: str) -> str:
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "config.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path


def deployed_path(run_dir: str) -> str:
    return os.path.join(run_dir, DEPLOYED)


def run(model: str, data_dir: str, run_dir: str, seed: int, dry_run: bool = False, **kw) -> int:
    if os.path.exists(deployed_path(run_dir)):
        print(f"[{model}] deployed model exists in {run_dir}; nothing to do")
        return 0
    cfg_path = write_config(build_config(model, data_dir, run_dir, seed, **kw), run_dir)
    train = [sys.executable, "-m", "nequip.scripts.train", cfg_path]
    deploy = [sys.executable, "-m", "nequip.scripts.deploy", "build", "--train-dir", os.path.join(run_dir, RUN_NAME), deployed_path(run_dir)]
    print(f"[{model}] " + " ".join(train))
    if dry_run:
        return 0
    marker = os.path.join(run_dir, TRAIN_DONE)
    if not os.path.exists(marker):
        rc = subprocess.call(train)
        if rc != 0:
            return rc
        open(marker, "w").close()
    print(f"[{model}] " + " ".join(deploy))
    return subprocess.call(deploy)


def load_calculator(deployed_model: str, species: list[str], device: str = "cpu"):
    # Type names equal the chemical symbols because the config uses `chemical_symbols`.
    from nequip.ase import NequIPCalculator

    return NequIPCalculator.from_deployed_model(deployed_model, device=device, species_to_type_name={s: s for s in species})


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train NequIP or Allegro with the native trainer on exported extxyz splits.")
    p.add_argument("--model", required=True, choices=sorted(MODELS))
    p.add_argument("--data-dir", required=True, help="directory written by export/export_splits.py")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=5)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--r-max", type=float, default=5.0)
    p.add_argument("--model-dtype", default="float32", choices=["float32", "float64"])
    p.add_argument("--dry-run", action="store_true", help="write the config, print the command, do not train")
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return run(a.model, a.data_dir, a.run_dir, a.seed, a.dry_run, device=a.device, epochs=a.epochs, batch_size=a.batch_size,
               lr=a.lr, r_max=a.r_max, model_dtype=a.model_dtype)


if __name__ == "__main__":
    raise SystemExit(main())
