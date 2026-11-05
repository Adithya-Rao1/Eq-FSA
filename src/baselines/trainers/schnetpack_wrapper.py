from __future__ import annotations

# schnetpack 2.0.4:
#  - RemoveOffsets(remove_atomrefs=True) reads per-element reference energies from the database

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import yaml

RUN_ID = "spk"
ENERGY, FORCES = "energy", "forces"
REPRESENTATIONS = {"schnet": "schnet", "painn": "painn"}


def run_subdir(run_dir: str) -> str:
    return os.path.join(run_dir, RUN_ID)


def best_model_path(run_dir: str) -> str:
    return os.path.join(run_subdir(run_dir), "best_model")


def read_split_record(data_dir: str) -> dict:
    with open(os.path.join(data_dir, "split.json")) as f:
        rec = json.load(f)
    missing = [k for k in ("species", "fixed_composition", "splits") if k not in rec]
    if missing:
        raise KeyError(f"{data_dir}/split.json lacks {missing}; re-export with the current export_splits")
    return rec


def fit_atomrefs(frames, zmax: int = 100) -> list[float]:
    # Per-element reference energies (eV) by least squares of total energy on element counts; elements absent from the training set stay 0.
    from baselines.export.extxyz import ENERGY_KEY

    counts = np.zeros((len(frames), zmax), dtype=np.float64)
    for i, a in enumerate(frames):
        for z in a.get_atomic_numbers():
            counts[i, z] += 1.0
    energies = np.array([a.info[ENERGY_KEY] for a in frames], dtype=np.float64)
    present = counts.any(axis=0)
    solution, *_ = np.linalg.lstsq(counts[:, present], energies, rcond=None)  # minimum-norm if rank-deficient
    refs = np.zeros(zmax)
    refs[present] = solution
    return refs.tolist()


def build_database(data_dir: str, out_dir: str) -> dict:
    # Writes <out_dir>/data.db and <out_dir>/split.npz (train, then val, then test, in exported order). Returns a summary.
    from baselines.trainers.spk_launch import ensure_schnetpack_importable

    ensure_schnetpack_importable()  # must precede any schnetpack import in this process
    from ase import Atoms
    from schnetpack.data import ASEAtomsData

    from baselines.export.extxyz import ENERGY_KEY, FORCES_KEY, read_extxyz

    rec = read_split_record(data_dir)
    frames = {name: read_extxyz(os.path.join(data_dir, f"{name}.xyz")) for name in ("train", "val", "test")}
    has_forces = all(FORCES_KEY in a.arrays for a in frames["train"] + frames["val"] + frames["test"])
    atomrefs = None if rec["fixed_composition"] else {ENERGY: fit_atomrefs(frames["train"])}
    os.makedirs(out_dir, exist_ok=True)
    db_path = os.path.join(out_dir, "data.db")
    if os.path.exists(db_path):
        os.remove(db_path)  # rebuilt from the exported files every time, so it can never drift from them
    units = {ENERGY: "eV", **({FORCES: "eV/Ang"} if has_forces else {})}
    db = ASEAtomsData.create(db_path, distance_unit="Ang", property_unit_dict=units, atomrefs=atomrefs)
    bounds, start = {}, 0
    for name in ("train", "val", "test"):
        atoms_list, props = [], []
        for a in frames[name]:
            atoms_list.append(Atoms(numbers=a.get_atomic_numbers(), positions=a.get_positions(), pbc=False))
            p = {ENERGY: np.array([a.info[ENERGY_KEY]], dtype=np.float64)}
            if has_forces:
                p[FORCES] = np.asarray(a.arrays[FORCES_KEY], dtype=np.float64)
            props.append(p)
        db.add_systems(props, atoms_list)
        bounds[name] = list(range(start, start + len(atoms_list)))
        start += len(atoms_list)
    split_path = os.path.join(out_dir, "split.npz")
    np.savez(split_path, train_idx=bounds["train"], val_idx=bounds["val"], test_idx=bounds["test"])
    return {"db": db_path, "split": split_path, "has_forces": has_forces, "fixed_composition": rec["fixed_composition"],
            "n_train": len(bounds["train"]), "n_val": len(bounds["val"]), "n_test": len(bounds["test"])}


def _output(name: str, weight: float) -> dict:
    metric = lambda target, **kw: {"_target_": target, **kw}  # noqa: E731
    return {
        "_target_": "schnetpack.task.ModelOutput", "name": name, "loss_fn": {"_target_": "torch.nn.MSELoss"},
        "metrics": {"mae": metric("torchmetrics.regression.MeanAbsoluteError"),
                    "rmse": metric("torchmetrics.regression.MeanSquaredError", squared=False)},
        "loss_weight": weight,
    }


def build_experiment(
    summary: dict, run_dir: str, seed: int, *, device: str = "cuda", epochs: int = 1, batch_size: int = 10, lr: float = 1e-3,
    cutoff: float = 5.0, energy_weight: float = 0.01, forces_weight: float = 0.99, num_workers: int = 0, overrides: dict | None = None,
) -> dict:
    has_forces, fixed = summary["has_forces"], summary["fixed_composition"]
    offsets = {"_target_": "schnetpack.transform.RemoveOffsets", "property": ENERGY, "remove_mean": True}
    post_offsets = {"_target_": "schnetpack.transform.AddOffsets", "property": ENERGY, "add_mean": True}
    if not fixed:
        offsets["remove_atomrefs"] = True
        post_offsets["add_atomrefs"] = True
    output_modules = [{"_target_": "schnetpack.atomistic.Atomwise", "output_key": ENERGY, "n_in": "${model.representation.n_atom_basis}",
                       "aggregation_mode": "sum"}]
    outputs = [_output(ENERGY, energy_weight if has_forces else 1.0)]
    if has_forces:
        output_modules.append({"_target_": "schnetpack.atomistic.Forces", "energy_key": ENERGY, "force_key": FORCES})
        outputs.append(_output(FORCES, forces_weight))
    exp = {
        "defaults": [{"override /model": "nnp"}, {"override /data": "custom"}, "_self_"],
        "seed": seed,
        "run": {"id": RUN_ID, "path": os.path.abspath(run_dir), "data_dir": os.path.join(os.path.abspath(run_dir), "spk_data")},
        "globals": {"cutoff": cutoff, "lr": lr, "energy_key": ENERGY, "forces_key": FORCES},
        "data": {
            "datapath": summary["db"], "split_file": summary["split"], "batch_size": batch_size, "num_workers": num_workers,
            "num_train": summary["n_train"], "num_val": summary["n_val"], "num_test": summary["n_test"], "distance_unit": "Ang",
            "property_units": {ENERGY: "eV", **({FORCES: "eV/Ang"} if has_forces else {})},
            "transforms": [
                {"_target_": "schnetpack.transform.SubtractCenterOfMass"}, offsets,
                {"_target_": "schnetpack.transform.MatScipyNeighborList", "cutoff": "${globals.cutoff}"},
                {"_target_": "schnetpack.transform.CastTo32"},
            ],
        },
        "model": {"output_modules": output_modules,
                  "postprocessors": [{"_target_": "schnetpack.transform.CastTo64"}, post_offsets]},
        "task": {"outputs": outputs},
        "trainer": {"max_epochs": epochs, "accelerator": "gpu" if device.startswith("cuda") else "cpu", "devices": 1},
    }
    for dotted, value in (overrides or {}).items():
        node = exp
        *parents, leaf = dotted.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = value
    return exp


def write_experiment(exp: dict, run_dir: str) -> str:
    cfg_dir = os.path.join(os.path.abspath(run_dir), "spk_configs")
    os.makedirs(os.path.join(cfg_dir, "experiment"), exist_ok=True)
    with open(os.path.join(cfg_dir, "experiment", "baseline.yaml"), "w") as f:
        f.write("# @package _global_\n")
        yaml.safe_dump(exp, f, sort_keys=False)
    return cfg_dir


def build_command(cfg_dir: str, model: str) -> list[str]:
    if model not in REPRESENTATIONS:
        raise ValueError(f"model must be one of {sorted(REPRESENTATIONS)}, got {model!r}")
    return [sys.executable, "-m", "baselines.trainers.spk_launch", f"--config-dir={cfg_dir}", "experiment=baseline",
            f"model/representation={REPRESENTATIONS[model]}"]


def run(model: str, data_dir: str, run_dir: str, seed: int, dry_run: bool = False, **kw) -> int:
    if os.path.exists(best_model_path(run_dir)):
        print(f"[{model}] best_model exists in {run_dir}; nothing to do")
        return 0
    if model not in REPRESENTATIONS:
        raise ValueError(f"model must be one of {sorted(REPRESENTATIONS)}, got {model!r}")
    os.makedirs(run_dir, exist_ok=True)
    summary = build_database(data_dir, os.path.join(run_dir, "spk_data"))
    exp = build_experiment(summary, run_dir, seed, **kw)
    cfg_dir = write_experiment(exp, run_dir)
    with open(os.path.join(run_dir, "wrapper.json"), "w") as f:
        json.dump({"model": model, "cutoff": exp["globals"]["cutoff"], "has_forces": summary["has_forces"]}, f)
    cmd = build_command(cfg_dir, model)
    print(f"[{model}] " + " ".join(cmd))
    return 0 if dry_run else subprocess.call(cmd)


def load_calculator(run_dir: str, device: str = "cpu"):
    from baselines.trainers.spk_launch import ensure_schnetpack_importable

    ensure_schnetpack_importable()
    import schnetpack.transform as trn
    import torch
    from schnetpack.interfaces import SpkCalculator

    with open(os.path.join(run_dir, "wrapper.json")) as f:
        meta = json.load(f)
    return SpkCalculator(
        model_file=best_model_path(run_dir), neighbor_list=trn.MatScipyNeighborList(cutoff=meta["cutoff"]), energy_key=ENERGY,
        force_key=FORCES, energy_unit="eV", position_unit="Ang", device=device, dtype=torch.float32,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train SchNet or PaiNN with SchNetPack's native trainer on exported extxyz splits.")
    p.add_argument("--model", required=True, choices=sorted(REPRESENTATIONS))
    p.add_argument("--data-dir", required=True, help="directory written by export/export_splits.py")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--cutoff", type=float, default=5.0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--dry-run", action="store_true", help="build the database and experiment file, print the command, do not train")
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return run(a.model, a.data_dir, a.run_dir, a.seed, a.dry_run, device=a.device, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
               cutoff=a.cutoff, num_workers=a.num_workers)


if __name__ == "__main__":
    raise SystemExit(main())
