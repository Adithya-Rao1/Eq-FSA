from __future__ import annotations

# Format: https://wiki.fysik.dtu.dk/ase/ase/io/formats.html#extxyz

import os
import re
from collections.abc import Iterable, Sequence

import numpy as np

ENERGY_KEY = "REF_energy"
FORCES_KEY = "REF_forces"


def _symbol(atomic_number: int) -> str:
    from ase.data import chemical_symbols

    if not 0 < atomic_number < len(chemical_symbols):
        raise ValueError(f"atomic number {atomic_number} is outside the periodic table")
    return chemical_symbols[atomic_number]


_PLAIN_TAG = re.compile(r"^[A-Za-z0-9_.+-]+$")


def _quote(value: str) -> str:
    # ASE's header parser reads an unquoted "[...]" as an array and silently drops the brackets
    value = str(value)
    if _PLAIN_TAG.match(value):
        return value
    if '"' in value or "\n" in value:
        raise ValueError(f"tag value may not contain a double quote or newline: {value!r}")
    return f'"{value}"'


def format_frame(
    atomic_numbers: Sequence[int], positions: np.ndarray, energy: float, forces: np.ndarray | None = None, tags: dict[str, str] | None = None,
) -> str:
    positions = np.asarray(positions, dtype=np.float64)
    n_atoms = len(atomic_numbers)
    if positions.shape != (n_atoms, 3):
        raise ValueError(f"positions must have shape ({n_atoms}, 3), got {positions.shape}")
    properties = "species:S:1:pos:R:3"
    if forces is not None:
        forces = np.asarray(forces, dtype=np.float64)
        if forces.shape != (n_atoms, 3):
            raise ValueError(f"forces must have shape ({n_atoms}, 3), got {forces.shape}")
        properties += f":{FORCES_KEY}:R:3"
    # repr() of a Python float round-trips float64 exactly; fixed-point formatting would drop digits on ~1e4 eV totals.
    header = f'Properties={properties} {ENERGY_KEY}={float(energy)!r} pbc="F F F"'
    for key, value in (tags or {}).items():
        header += f" {key}={_quote(value)}"
    lines = [str(n_atoms), header]
    for i in range(n_atoms):
        row = f"{_symbol(int(atomic_numbers[i]))} " + " ".join(f"{x:.10f}" for x in positions[i])
        if forces is not None:
            row += " " + " ".join(f"{x:.10f}" for x in forces[i])
        lines.append(row)
    return "\n".join(lines) + "\n"


def write_extxyz(path: str, frames: Iterable[dict]) -> int:
    # frames: dicts with atomic_numbers, positions, energy (eV), optional forces (eV/A) and tags. Returns the number written.
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    count = 0
    with open(path, "w") as f:
        for frame in frames:
            f.write(format_frame(frame["atomic_numbers"], frame["positions"], frame["energy"], frame.get("forces"), frame.get("tags")))
            count += 1
    return count


def read_extxyz(path: str) -> list:
    from ase.io import read

    return read(path, index=":", format="extxyz")
