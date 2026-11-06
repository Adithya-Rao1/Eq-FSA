from __future__ import annotations

# Constants: CODATA 2018 (physics.nist.gov/cuu/Constants); 1 kcal = 4.184 J.

from typing import TYPE_CHECKING, TypeVar, Union

import numpy as np

if TYPE_CHECKING:  # annotation only: lets torch-free environments (the classical force-field image) import the constants
    import torch

Number = TypeVar("Number", bound=Union[float, np.ndarray, "torch.Tensor"])

HARTREE_TO_EV: float = 27.211386245988
BOHR_TO_ANGSTROM: float = 0.529177210903
_AVOGADRO_PER_MOL: float = 6.02214076e23
_ELEMENTARY_CHARGE_C: float = 1.602176634e-19
KCAL_PER_MOL_TO_EV: float = 4184.0 / (_AVOGADRO_PER_MOL * _ELEMENTARY_CHARGE_C)
# 1 eV per particle = e * N_A joule per mole (exact in the 2019 SI), i.e. e * N_A / 1000 kJ/mol. OpenMM works in kJ/mol.
EV_TO_KJ_PER_MOL: float = _AVOGADRO_PER_MOL * _ELEMENTARY_CHARGE_C / 1000.0


def hartree_to_ev(energy: Number) -> Number:
    return energy * HARTREE_TO_EV


def ev_to_hartree(energy: Number) -> Number:
    return energy / HARTREE_TO_EV


def bohr_to_angstrom(length: Number) -> Number:
    return length * BOHR_TO_ANGSTROM


def angstrom_to_bohr(length: Number) -> Number:
    return length / BOHR_TO_ANGSTROM


def kcal_per_mol_to_ev(energy: Number) -> Number:
    return energy * KCAL_PER_MOL_TO_EV


def ev_to_kcal_per_mol(energy: Number) -> Number:
    return energy / KCAL_PER_MOL_TO_EV


def hartree_per_bohr_to_ev_per_angstrom(force: Number) -> Number:
    # Converting forces needs the energy factor divided by the length factor.
    return force * (HARTREE_TO_EV / BOHR_TO_ANGSTROM)


def ev_per_angstrom_to_hartree_per_bohr(force: Number) -> Number:
    return force / (HARTREE_TO_EV / BOHR_TO_ANGSTROM)


def kcal_per_mol_per_angstrom_to_ev_per_angstrom(force: Number) -> Number:
    # Length unit (Angstrom) is already shared on both sides, so only the energy factor applies.
    return force * KCAL_PER_MOL_TO_EV


UnitConvention = str  # one of "hartree_bohr", "kcal_mol_angstrom", "ev_angstrom"

_ENERGY_CONVERTERS = {
    "hartree_bohr": hartree_to_ev,
    "kcal_mol_angstrom": kcal_per_mol_to_ev,
    "ev_angstrom": lambda x: x,
}
_FORCE_CONVERTERS = {
    "hartree_bohr": hartree_per_bohr_to_ev_per_angstrom,
    "kcal_mol_angstrom": kcal_per_mol_per_angstrom_to_ev_per_angstrom,
    "ev_angstrom": lambda x: x,
}


def to_ev_angstrom(energy: Number, forces: Number, source: UnitConvention) -> tuple[Number, Number]:
    if source not in _ENERGY_CONVERTERS:
        raise ValueError(f"unknown unit convention {source!r}; expected one of {sorted(_ENERGY_CONVERTERS)}")
    return _ENERGY_CONVERTERS[source](energy), _FORCE_CONVERTERS[source](forces)
