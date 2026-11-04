# Writes LAMMPS input decks (NVT/NVE) for deployed NequIP / Allegro / MACE potentials and optionally invokes `lmp` when --run is given.
# Syntax follows mir-group/pair_nequip, mir-group/pair_allegro (Musaelian et al., arXiv:2204.05249) and ACEsuit/lammps-mace (MACE docs, "Using MACE with LAMMPS");
from __future__ import annotations

import argparse
import os
import subprocess

from baselines.provenance import env_manifest
from baselines.provenance.seeding import set_seed

PAIR_STYLES: dict[str, str] = {
    "nequip": "pair_style nequip",
    "allegro": "pair_style allegro",
    # lammps-mace needs this flag for single-process runs.
    "mace": "pair_style mace no_domain_decomposition",
}


def build_input_deck(
    model: str,
    data_file: str,
    potential_path: str,
    elements: list[str],
    ensemble: str = "nvt",
    timestep_fs: float = 1.0,
    temperature_k: float = 300.0,
    n_steps: int = 10000,
    thermostat: str = "nose-hoover",
    dump_every: int = 100,
    seed: int = 0,
    dump_path: str = "traj.lammpstrj",
    thermo_path: str = "thermo.log",
) -> str:
    if model not in PAIR_STYLES:
        raise ValueError(f"model must be one of {sorted(PAIR_STYLES)}")
    if ensemble not in ("nvt", "nve"):
        raise ValueError("ensemble must be 'nvt' or 'nve'")
    if thermostat not in ("nose-hoover", "langevin"):
        raise ValueError("thermostat must be 'nose-hoover' or 'langevin'")
    dt_ps = timestep_fs / 1000.0  # LAMMPS 'metal' units use ps for time
    tdamp = 100.0 * dt_ps
    lines = [
        "units metal",
        "atom_style atomic",
        "boundary p p p",
        f"read_data {data_file}",
        PAIR_STYLES[model],
        f"pair_coeff * * {potential_path} {' '.join(elements)}",
        f"timestep {dt_ps:.6f}",
        "neighbor 1.0 bin",
        f"velocity all create {temperature_k} {seed + 1} mom yes rot yes dist gaussian",  # LAMMPS rejects seed 0
        "thermo_style custom step time temp pe ke etotal press",
        f"thermo {dump_every}",
        f"dump traj all custom {dump_every} {dump_path} id type element x y z vx vy vz fx fy fz",
        f"dump_modify traj element {' '.join(elements)}",
        f"log {thermo_path}",
    ]
    if ensemble == "nvt" and thermostat == "nose-hoover":
        lines.append(f"fix integ all nvt temp {temperature_k} {temperature_k} {tdamp:.6f}")
    elif ensemble == "nvt":
        lines += ["fix integ all nve", f"fix thermo all langevin {temperature_k} {temperature_k} {tdamp:.6f} {seed + 1}"]
    else:
        lines.append("fix integ all nve")
    lines.append(f"run {n_steps}")
    return "\n".join(lines) + "\n"


def write_input_deck(path: str, **kwargs) -> str:
    deck = build_input_deck(**kwargs)
    with open(path, "w") as f:
        f.write(deck)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a LAMMPS input deck for a deployed MLIP; --run additionally invokes lmp.")
    parser.add_argument("--model", required=True, choices=sorted(PAIR_STYLES))
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--potential", required=True, help="deployed/compiled potential file")
    parser.add_argument("--elements", nargs="+", required=True, help="element symbol per LAMMPS atom type, in type order")
    parser.add_argument("--ensemble", default="nvt", choices=["nvt", "nve"])
    parser.add_argument("--thermostat", default="nose-hoover", choices=["nose-hoover", "langevin"])
    parser.add_argument("--timestep-fs", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--dump-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--lmp", default="lmp", help="LAMMPS binary")
    parser.add_argument("--run", action="store_true", help="actually run lmp (default: only write the input file)")
    args = parser.parse_args()
    config = vars(args)
    set_seed(config["seed"])
    os.makedirs(config["run_dir"], exist_ok=True)
    env_manifest.write(config["run_dir"])
    if config["ensemble"] == "nve":
        from baselines.md.stability_protocol import require_nve_eligible

        require_nve_eligible(config["model"])
    input_path = os.path.join(config["run_dir"], f"in.{config['model']}_{config['ensemble']}")
    write_input_deck(
        input_path, model=config["model"], data_file=config["data_file"], potential_path=config["potential"],
        elements=config["elements"], ensemble=config["ensemble"], timestep_fs=config["timestep_fs"],
        temperature_k=config["temperature"], n_steps=config["steps"], thermostat=config["thermostat"],
        dump_every=config["dump_every"], seed=config["seed"],
        dump_path=os.path.join(config["run_dir"], "traj.lammpstrj"), thermo_path=os.path.join(config["run_dir"], "thermo.log"),
    )
    print(f"wrote {input_path}")
    if config["run"]:
        subprocess.run([config["lmp"], "-in", input_path], check=True)


if __name__ == "__main__":
    main()
