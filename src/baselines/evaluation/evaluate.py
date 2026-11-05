from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Sequence

import numpy as np

from baselines.export.extxyz import ENERGY_KEY, FORCES_KEY, read_extxyz

EV_TO_MEV = 1000.0


def _rms(x: np.ndarray) -> float:
    return float(math.sqrt(np.mean(np.square(x))))


def predict(calculator, frames: Sequence) -> dict:
    # Runs the calculator on a copy of every frame, so the reference data on the inputs is never overwritten.
    energies, forces = [], []
    for ref in frames:
        atoms = ref.copy()
        atoms.calc = calculator
        energies.append(float(atoms.get_potential_energy()))
        forces.append(np.asarray(atoms.get_forces(), dtype=np.float64))
    return {"energy": np.array(energies, dtype=np.float64), "forces": forces}


def score(frames: Sequence, pred_energy: np.ndarray, pred_forces: Sequence[np.ndarray] | None, relative: bool = False) -> dict:
    # relative=True also reports energy errors after removing the single best constant offset (the mean error). 
    # This is the correct comparison for a classical force field, whose absolute energies reference a different zero than the DFT references.
    if len(frames) == 0:
        raise ValueError("cannot score an empty test set")
    ref_e = np.array([a.info[ENERGY_KEY] for a in frames], dtype=np.float64)
    n_atoms = np.array([len(a) for a in frames], dtype=np.float64)
    err = (np.asarray(pred_energy, dtype=np.float64) - ref_e) * EV_TO_MEV
    out = {
        "n_structures": len(frames),
        "energy_mae_meV": float(np.mean(np.abs(err))),
        "energy_rmse_meV": _rms(err),
        "energy_mae_meV_per_atom": float(np.mean(np.abs(err) / n_atoms)),
        "energy_rmse_meV_per_atom": _rms(err / n_atoms),
    }
    if relative:
        centered = err - np.mean(err)
        out["energy_relative_mae_meV"] = float(np.mean(np.abs(centered)))
        out["energy_relative_rmse_meV"] = _rms(centered)
        out["energy_offset_meV"] = float(np.mean(err))
    if pred_forces is not None and all(FORCES_KEY in a.arrays for a in frames):
        diff = np.concatenate([(np.asarray(p) - a.arrays[FORCES_KEY]).ravel() for p, a in zip(pred_forces, frames)]) * EV_TO_MEV
        out["force_mae_meV_per_A"] = float(np.mean(np.abs(diff)))
        out["force_rmse_meV_per_A"] = _rms(diff)
        out["n_force_components"] = int(diff.size)
    return out


def evaluate_calculator(calculator, frames: Sequence, relative: bool = False, with_forces: bool | None = None) -> dict:
    # with_forces=None --> score forces iff every frame carries REF_forces (QM9 has none).
    if with_forces is None:
        with_forces = bool(len(frames)) and all(FORCES_KEY in a.arrays for a in frames)
    if with_forces:
        pred = predict(calculator, frames)
        return score(frames, pred["energy"], pred["forces"], relative)
    energies = []
    for ref in frames:
        atoms = ref.copy()
        atoms.calc = calculator
        energies.append(float(atoms.get_potential_energy()))
    return score(frames, np.array(energies), None, relative)


def evaluate_file(calculator, xyz_path: str, out_path: str | None = None, relative: bool = False, extra: dict | None = None) -> dict:
    metrics = evaluate_calculator(calculator, read_extxyz(xyz_path), relative=relative)
    metrics["test_file"] = xyz_path
    metrics.update(extra or {})
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=1)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Score an ASE calculator on an exported extxyz test file.")
    p.add_argument("--test-file", required=True)
    p.add_argument("--out", required=True, help="metrics JSON path")
    p.add_argument("--calculator", required=True, help="module:function; the function takes no arguments and returns an ASE calculator")
    p.add_argument("--relative", action="store_true", help="also report offset-removed energy errors (for classical force fields)")
    return p


def main(argv: list[str] | None = None) -> int:
    import importlib

    args = build_parser().parse_args(argv)
    module_name, _, func_name = args.calculator.partition(":")
    if not func_name:
        raise SystemExit("--calculator must look like module:function")
    calculator = getattr(importlib.import_module(module_name), func_name)()
    metrics = evaluate_file(calculator, args.test_file, args.out, args.relative)
    print(json.dumps(metrics, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
