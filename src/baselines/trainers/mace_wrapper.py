from __future__ import annotations

# mace-torch 0.3.9 source:
#  - frames without REF_forces get forces_weight 0 (data/utils.py::config_from_atoms), so QM9 (energy only) needs no special flag;
#  - `--restart_latest` is a no-op when no checkpoint exists, so it is always passed and a requeued job resumes by itself;
#  - the final model is saved as <model_dir>/<name>.model; model_dir defaults to work_dir; checkpoints go to <work_dir>/checkpoints;
#  - `--E0s average` fits per-element reference energies by least squares on the training set --> atomization-style referencing that ~1e4 eV totals need = the default here.

import argparse
import os
import subprocess
import sys

MODEL_NAME = "mace"


def final_model_path(run_dir: str, name: str = MODEL_NAME) -> str:
    return os.path.join(run_dir, f"{name}.model")


def build_command(
    data_dir: str, run_dir: str, seed: int, *, device: str = "cuda", epochs: int = 1, batch_size: int = 32, lr: float = 0.01,
    r_max: float = 5.0, num_channels: int = 128, max_l: int = 2, num_interactions: int = 2, energy_weight: float = 1.0,
    forces_weight: float = 100.0, e0s: str = "average", dtype: str = "float64", name: str = MODEL_NAME, ema: bool = True,
    extra: list[str] | None = None,
) -> list[str]:
    # The 128x0e+128x1o (+128x2e) hidden_irreps follow from num_channels and max_l inside MACE
    if num_interactions < 2:
        # mace-torch 0.3.9 trains a 1-interaction model but then crashes saving it
        raise ValueError("mace-torch 0.3.9 cannot save a model with num_interactions < 2; use 2 or more")
    hidden = "+".join(f"{num_channels}x{l}{'e' if l % 2 == 0 else 'o'}" for l in range(max_l + 1))
    cmd = [
        sys.executable, "-m", "mace.cli.run_train",
        "--name", name, "--work_dir", run_dir, "--seed", str(seed),
        "--train_file", os.path.join(data_dir, "train.xyz"), "--valid_file", os.path.join(data_dir, "val.xyz"),
        "--test_file", os.path.join(data_dir, "test.xyz"),
        "--energy_key", "REF_energy", "--forces_key", "REF_forces", "--E0s", e0s,
        "--model", "MACE", "--hidden_irreps", hidden, "--r_max", str(r_max), "--num_interactions", str(num_interactions),
        "--batch_size", str(batch_size), "--valid_batch_size", str(batch_size), "--max_num_epochs", str(epochs), "--lr", str(lr),
        "--energy_weight", str(energy_weight), "--forces_weight", str(forces_weight),
        "--default_dtype", dtype, "--device", device, "--restart_latest", "--error_table", "PerAtomMAE",
    ]
    if ema:
        cmd += ["--ema", "--ema_decay", "0.99"]
    return cmd + list(extra or [])


def run(data_dir: str, run_dir: str, seed: int, dry_run: bool = False, **kw) -> int:
    # Already trained -> nothing to do, so `sbatch --dependency=afterany` chains end quietly.
    if os.path.exists(final_model_path(run_dir, kw.get("name", MODEL_NAME))):
        print(f"[mace] final model exists in {run_dir}; nothing to do")
        return 0
    os.makedirs(run_dir, exist_ok=True)
    cmd = build_command(data_dir, run_dir, seed, **kw)
    print("[mace] " + " ".join(cmd))
    return 0 if dry_run else subprocess.call(cmd)


def load_calculator(model_path: str, device: str = "cpu", dtype: str = "float64"):
    from mace.calculators import MACECalculator

    return MACECalculator(model_paths=model_path, device=device, default_dtype=dtype)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train MACE with its native trainer on exported extxyz splits.")
    p.add_argument("--data-dir", required=True, help="directory written by export/export_splits.py")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--r-max", type=float, default=5.0)
    p.add_argument("--num-channels", type=int, default=128)
    p.add_argument("--max-l", type=int, default=2)
    p.add_argument("--num-interactions", type=int, default=2)
    p.add_argument("--forces-weight", type=float, default=100.0)
    p.add_argument("--energy-weight", type=float, default=1.0)
    p.add_argument("--dry-run", action="store_true", help="print the command and exit")
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return run(
        a.data_dir, a.run_dir, a.seed, a.dry_run, device=a.device, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr, r_max=a.r_max,
        num_channels=a.num_channels, max_l=a.max_l, num_interactions=a.num_interactions, forces_weight=a.forces_weight,
        energy_weight=a.energy_weight,
    )


if __name__ == "__main__":
    raise SystemExit(main())
