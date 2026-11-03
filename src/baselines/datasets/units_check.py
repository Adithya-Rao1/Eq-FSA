from __future__ import annotations

from baselines.config.units import (
    BOHR_TO_ANGSTROM,
    HARTREE_TO_EV,
    KCAL_PER_MOL_TO_EV,
    Number,
    to_ev_angstrom,
)

__all__ = ["to_ev_angstrom", "Number", "HARTREE_TO_EV", "BOHR_TO_ANGSTROM", "KCAL_PER_MOL_TO_EV"]

# CODATA 2018 sanity bounds (physics.nist.gov/cuu/Constants); catches an accidental constant edit in
# baselines/config/units.py at import time.
assert 27.0 < HARTREE_TO_EV < 27.3, f"HARTREE_TO_EV looks wrong: {HARTREE_TO_EV}"
assert 0.52 < BOHR_TO_ANGSTROM < 0.53, f"BOHR_TO_ANGSTROM looks wrong: {BOHR_TO_ANGSTROM}"
assert 0.04 < KCAL_PER_MOL_TO_EV < 0.05, f"KCAL_PER_MOL_TO_EV looks wrong: {KCAL_PER_MOL_TO_EV}"
assert to_ev_angstrom(1.0, 1.0, "ev_angstrom") == (1.0, 1.0), "ev_angstrom identity conversion must be a no-op"
