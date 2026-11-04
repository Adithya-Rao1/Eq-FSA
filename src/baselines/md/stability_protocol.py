# MLIPAudit-style NVT stability protocol plus NVE-conservation exclusion for non-conservative models.
# MLIPAudit (github.com/instadeepai/mlipaudit) stability benchmark; Orb non-conservative forces: Neumann et al., "Orb: A Fast, Scalable Neural Network Potential" (arXiv:2410.22570);
# conservation test: Fu et al., "Forces are not Enough" (arXiv:2210.07237);
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Callable

import numpy as np

from baselines.md import trajectory_metrics as tm
from baselines.provenance import env_manifest
from baselines.provenance.seeding import set_seed

# Orb forces are predicted directly (not -grad E).
NVE_EXCLUDED_MODELS: frozenset[str] = frozenset({"orb", "orb_v2", "orb_v3", "orb-v2", "orb-v3"})
NVE_EXCLUSION_REASON = "Orb models have non-conservative (directly predicted) forces; NVE energy conservation is not meaningful (artifact section 07)."


def is_nve_excluded(model_name: str) -> bool:
    name = model_name.strip().lower()
    return name in NVE_EXCLUDED_MODELS or name.startswith("orb")


def require_nve_eligible(model_name: str) -> None:
    if is_nve_excluded(model_name):
        raise ValueError(f"{model_name}: {NVE_EXCLUSION_REASON}")


def nve_conservation_report(
    model_name: str, times_fs: np.ndarray, total_energy_ev: np.ndarray, n_atoms: int, max_abs_slope_ev_per_atom_per_ps: float = 0.01
) -> dict[str, Any]:
    if is_nve_excluded(model_name):
        return {"model": model_name, "skipped": True, "passed": None, "reason": NVE_EXCLUSION_REASON}
    drift = tm.energy_drift(times_fs, total_energy_ev, n_atoms)
    passed = bool(np.isfinite(drift["slope_ev_per_atom_per_ps"]) and abs(drift["slope_ev_per_atom_per_ps"]) <= max_abs_slope_ev_per_atom_per_ps)
    return {"model": model_name, "skipped": False, "passed": passed, "max_abs_slope_ev_per_atom_per_ps": max_abs_slope_ev_per_atom_per_ps, **drift}


def infer_bonds(positions0: np.ndarray, cutoff: float = 1.8) -> np.ndarray:
    d = np.linalg.norm(positions0[:, None] - positions0[None], axis=-1)
    i, j = np.where(np.triu(d < cutoff, k=1))
    return np.stack([i, j], axis=1).astype(np.int64)


def collect_nvt(advance: Callable[[int], tuple[np.ndarray, float, float]], n_steps: int, report_interval: int) -> dict[str, np.ndarray]:
    # advance(k) must integrate k NVT steps and return (positions (N,3) in Angstrom, total energy in eV, temperature in K).
    pos, en, temp = [], [], []
    done = 0
    while done < n_steps:
        k = min(report_interval, n_steps - done)
        p, e, t = advance(k)
        pos.append(np.asarray(p, dtype=np.float64))
        en.append(e)
        temp.append(t)
        done += k
        if not (np.isfinite(pos[-1]).all() and np.isfinite(e) and np.isfinite(t)):
            break
    return {"positions": np.stack(pos), "energies": np.asarray(en, dtype=np.float64), "temperatures": np.asarray(temp, dtype=np.float64)}


def stability_report(
    positions: np.ndarray,
    energies: np.ndarray,
    temperatures: np.ndarray,
    initial_positions: np.ndarray,
    target_temperature: float,
    bonds: np.ndarray | None = None,
    bond_excursion_frac: float = 0.5,
    energy_runaway_ev_per_atom: float = 2.0,
    temperature_runaway_factor: float = 3.0,
) -> dict[str, Any]:
    positions = np.asarray(positions, dtype=np.float64)
    energies = np.asarray(energies, dtype=np.float64)
    temperatures = np.asarray(temperatures, dtype=np.float64)
    initial_positions = np.asarray(initial_positions, dtype=np.float64)
    n_frames, n_atoms = positions.shape[0], positions.shape[1]
    if bonds is None:
        bonds = infer_bonds(initial_positions)
    bonds = np.asarray(bonds, dtype=np.int64).reshape(-1, 2)

    bad_frames = ~(np.isfinite(positions).all(axis=(1, 2)) & np.isfinite(energies) & np.isfinite(temperatures))
    thresholds = {
        "bond_excursion_frac": bond_excursion_frac,
        "energy_runaway_ev_per_atom": energy_runaway_ev_per_atom,
        "temperature_runaway_factor": temperature_runaway_factor,
        "target_temperature_k": target_temperature,
    }
    explosion = {"passed": not bool(bad_frames.any()), "first_bad_frame": int(np.argmax(bad_frames)) if bad_frames.any() else None}
    report: dict[str, Any] = {"n_frames": int(n_frames), "n_atoms": int(n_atoms), "n_bonds": int(len(bonds)), "thresholds": thresholds}
    checks: dict[str, Any] = {"explosion": explosion}

    if not explosion["passed"]:
        reason = "skipped: non-finite positions/energies/temperatures"
        for name in ("bond_excursion", "energy_runaway", "temperature_runaway"):
            checks[name] = {"passed": False, "skipped": True, "reason": reason}
    else:
        if len(bonds):
            d0 = np.linalg.norm(initial_positions[bonds[:, 0]] - initial_positions[bonds[:, 1]], axis=-1)
            d = np.linalg.norm(positions[:, bonds[:, 0]] - positions[:, bonds[:, 1]], axis=-1)
            rel = np.abs(d - d0) / d0
            viol = rel > bond_excursion_frac
            checks["bond_excursion"] = {
                "passed": not bool(viol.any()),
                "max_relative_excursion": float(rel.max()),
                "n_violating_bonds": int(viol.any(axis=0).sum()),
                "first_violating_frame": int(np.argmax(viol.any(axis=1))) if viol.any() else None,
            }
        else:
            checks["bond_excursion"] = {"passed": True, "max_relative_excursion": 0.0, "n_violating_bonds": 0, "first_violating_frame": None}
        e_dev = float(np.max(np.abs(energies - energies[0])) / n_atoms)
        checks["energy_runaway"] = {"passed": e_dev <= energy_runaway_ev_per_atom, "max_abs_deviation_ev_per_atom": e_dev}
        t_max = float(temperatures.max())
        checks["temperature_runaway"] = {"passed": t_max <= temperature_runaway_factor * target_temperature, "max_temperature_k": t_max}

    report["checks"] = checks
    report["passed"] = all(c["passed"] for c in checks.values())
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an NVT trajectory (.npz with positions, energies, temperatures) with the stability protocol.")
    parser.add_argument("--trajectory", required=True, help=".npz with positions (T,N,3) Angstrom, energies (T,) eV, temperatures (T,) K")
    parser.add_argument("--target-temperature", type=float, required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bond-excursion-frac", type=float, default=0.5)
    parser.add_argument("--energy-runaway-ev-per-atom", type=float, default=2.0)
    parser.add_argument("--temperature-runaway-factor", type=float, default=3.0)
    args = parser.parse_args()
    config = vars(args)
    set_seed(config["seed"])
    os.makedirs(config["run_dir"], exist_ok=True)
    env_manifest.write(config["run_dir"])
    data = np.load(config["trajectory"])
    report = stability_report(
        data["positions"], data["energies"], data["temperatures"], data["positions"][0], config["target_temperature"],
        bonds=data["bonds"] if "bonds" in data.files else None,
        bond_excursion_frac=config["bond_excursion_frac"],
        energy_runaway_ev_per_atom=config["energy_runaway_ev_per_atom"],
        temperature_runaway_factor=config["temperature_runaway_factor"],
    )
    with open(os.path.join(config["run_dir"], "stability_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
