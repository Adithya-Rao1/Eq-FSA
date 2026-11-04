# Runs NVT (Langevin) / NVE (Verlet) MD in OpenMM with a torch potential wrapped via openmm-torch's TorchForce, or an openmm-ml MLPotential.
# OpenMM (Eastman et al., PLoS Comput. Biol. 2017), openmm-torch (github.com/openmm/openmm-torch), openmm-ml (github.com/openmm/openmm-ml).
from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np

from baselines.provenance import env_manifest
from baselines.provenance.seeding import set_seed

# OpenMM works in nm and kJ/mol; the potential works in Angstrom and eV. 1 eV = 96.48533212 kJ/mol (CODATA 2018); 1 nm = 10 Angstrom.
EV_TO_KJ_PER_MOL = 96.48533212331002
NM_TO_ANGSTROM = 10.0


def build_torch_force(model_path: str, atomic_numbers: np.ndarray, device: str = "cpu") -> Any:
    # model_path is a TorchScript module with forward(positions_angstrom (N,3), atomic_numbers (N,)) -> scalar energy in eV.
    import torch
    import openmmtorch

    class UnitWrapper(torch.nn.Module):
        def __init__(self, inner: torch.nn.Module, numbers: torch.Tensor) -> None:
            super().__init__()
            self.inner = inner
            self.register_buffer("numbers", numbers)

        def forward(self, positions: torch.Tensor) -> torch.Tensor:
            return self.inner(positions * 10.0, self.numbers) * 96.48533212331002

    inner = torch.jit.load(model_path, map_location=device)
    numbers = torch.as_tensor(np.asarray(atomic_numbers), dtype=torch.long, device=device)
    return openmmtorch.TorchForce(torch.jit.script(UnitWrapper(inner, numbers)))


def build_simulation(
    atomic_numbers: np.ndarray,
    positions_angstrom: np.ndarray,
    ensemble: str,
    timestep_fs: float,
    temperature_k: float,
    friction_per_ps: float = 1.0,
    model_path: str | None = None,
    mlpotential_name: str | None = None,
    device: str = "cpu",
    seed: int = 0,
) -> Any:
    if (model_path is None) == (mlpotential_name is None):
        raise ValueError("provide exactly one of model_path or mlpotential_name")
    if ensemble not in ("nvt", "nve"):
        raise ValueError("ensemble must be 'nvt' or 'nve'")
    import openmm
    from openmm import app, unit

    topology = app.Topology()
    chain = topology.addChain()
    for z in atomic_numbers:
        element = app.Element.getByAtomicNumber(int(z))
        residue = topology.addResidue("MOL", chain)
        topology.addAtom(element.symbol, element, residue)
    positions = unit.Quantity(np.asarray(positions_angstrom, dtype=np.float64), unit.angstrom)

    if mlpotential_name is not None:
        from openmmml import MLPotential

        system = MLPotential(mlpotential_name).createSystem(topology)
    else:
        system = openmm.System()
        for z in atomic_numbers:
            system.addParticle(app.Element.getByAtomicNumber(int(z)).mass)
        system.addForce(build_torch_force(model_path, atomic_numbers, device))

    if ensemble == "nvt":
        integrator = openmm.LangevinMiddleIntegrator(temperature_k * unit.kelvin, friction_per_ps / unit.picosecond, timestep_fs * unit.femtosecond)
        integrator.setRandomNumberSeed(seed)
    else:
        integrator = openmm.VerletIntegrator(timestep_fs * unit.femtosecond)
    simulation = app.Simulation(topology, system, integrator, openmm.Platform.getPlatformByName("CUDA" if device.startswith("cuda") else "CPU"))
    simulation.context.setPositions(positions)
    simulation.context.setVelocitiesToTemperature(temperature_k * unit.kelvin, seed + 1)
    return simulation


def run(simulation: Any, n_steps: int, report_every: int, out_dir: str) -> str:
    # Writes traj.dcd, energies.csv (StateDataReporter) and trajectory.npz in the units of trajectory_metrics (Angstrom, fs, eV, K).
    from openmm import app, unit

    simulation.reporters.append(app.DCDReporter(os.path.join(out_dir, "traj.dcd"), report_every))
    simulation.reporters.append(
        app.StateDataReporter(os.path.join(out_dir, "energies.csv"), report_every, step=True, time=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True, temperature=True)
    )
    dt_fs = simulation.integrator.getStepSize().value_in_unit(unit.femtosecond)
    pos, vel, pe, ke, times = [], [], [], [], []
    for _ in range(n_steps // report_every):
        simulation.step(report_every)
        state = simulation.context.getState(getPositions=True, getVelocities=True, getEnergy=True)
        pos.append(state.getPositions(asNumpy=True).value_in_unit(unit.angstrom))
        vel.append(state.getVelocities(asNumpy=True).value_in_unit(unit.angstrom / unit.femtosecond))
        pe.append(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) / EV_TO_KJ_PER_MOL)
        ke.append(state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole) / EV_TO_KJ_PER_MOL)
        times.append(simulation.currentStep * dt_fs)
    pe_a, ke_a = np.asarray(pe), np.asarray(ke)
    path = os.path.join(out_dir, "trajectory.npz")
    np.savez(path, positions=np.asarray(pos), velocities=np.asarray(vel), potential_energy=pe_a, kinetic_energy=ke_a, energies=pe_a + ke_a, times_fs=np.asarray(times))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NVT/NVE MD in OpenMM with a TorchScript potential or an openmm-ml MLPotential.")
    parser.add_argument("--structure", required=True, help=".npz with atomic_numbers (N,) and positions (N,3) in Angstrom")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--mlpotential", default=None)
    parser.add_argument("--model-name", default="custom", help="used for the Orb NVE exclusion")
    parser.add_argument("--ensemble", default="nvt", choices=["nvt", "nve"])
    parser.add_argument("--timestep-fs", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--friction-per-ps", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--report-every", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    config = vars(args)
    set_seed(config["seed"])
    os.makedirs(config["run_dir"], exist_ok=True)
    env_manifest.write(config["run_dir"])
    if config["ensemble"] == "nve":
        from baselines.md.stability_protocol import require_nve_eligible

        require_nve_eligible(config["model_name"])
    structure = np.load(config["structure"])
    simulation = build_simulation(
        structure["atomic_numbers"], structure["positions"], config["ensemble"], config["timestep_fs"], config["temperature"],
        config["friction_per_ps"], config["model_path"], config["mlpotential"], config["device"], config["seed"],
    )
    print(run(simulation, config["steps"], config["report_every"], config["run_dir"]))


if __name__ == "__main__":
    main()
