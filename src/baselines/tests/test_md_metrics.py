# Tests for src/baselines/md/: trajectory_metrics on synthetic data, the Orb NVE exclusion, and stability_report logic.
from __future__ import annotations

import sys
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import json

import numpy as np
import pytest

from baselines.md import trajectory_metrics as tm
from baselines.md.stability_protocol import (
    NVE_EXCLUDED_MODELS,
    nve_conservation_report,
    require_nve_eligible,
    stability_report,
)

RNG = np.random.default_rng(0)
BOX = np.array([10.0, 10.0, 10.0])
N_ATOMS = 40
N_FRAMES = 200
IDEAL_GAS = RNG.uniform(0.0, 10.0, size=(N_FRAMES, N_ATOMS, 3))
VELOCITY = np.array([0.01, -0.02, 0.005])  # Angstrom/fs
START = RNG.uniform(0.0, 10.0, size=(1, N_ATOMS, 3))
FREE_PARTICLES = START + np.arange(50)[:, None, None] * 2.0 * VELOCITY
TIMES_FS = np.arange(100) * 1.0
CHAIN = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]])
CHAIN_TRAJ = np.repeat(CHAIN[None], 20, axis=0)
BONDS = np.array([[0, 1], [1, 2]])


def test_ideal_gas_rdf_is_close_to_one():
    r, g = tm.rdf(IDEAL_GAS, BOX, r_max=5.0, n_bins=10)
    assert r.shape == g.shape == (10,)
    assert np.allclose(g, 1.0, atol=0.1)


def test_free_particle_msd_is_quadratic_in_time_for_ballistic_motion():
    out = tm.msd(FREE_PARTICLES, max_lag=20)
    expected = (np.arange(21) * 2.0) ** 2 * np.sum(VELOCITY**2)
    assert np.allclose(out, expected)


def test_random_walk_msd_is_linear_in_time():
    steps = RNG.normal(size=(2000, 100, 3))
    out = tm.msd(np.cumsum(steps, axis=0), max_lag=10)
    assert np.allclose(out[1:] / np.arange(1, 11), 3.0, rtol=0.1)


def test_vacf_of_constant_velocity_is_one_and_starts_at_one():
    v = np.broadcast_to(VELOCITY, (30, N_ATOMS, 3))
    out = tm.vacf(v, max_lag=10)
    assert out[0] == 1.0
    assert np.allclose(out, 1.0)


def test_constant_energy_has_zero_drift():
    drift = tm.energy_drift(TIMES_FS, np.full(100, -5.0), n_atoms=10)
    assert drift["slope_ev_per_fs"] == pytest.approx(0.0, abs=1e-15)
    assert drift["relative_drift"] == pytest.approx(0.0, abs=1e-15)


def test_linear_energy_drift_recovers_slope():
    drift = tm.energy_drift(TIMES_FS, -5.0 + 1e-4 * TIMES_FS, n_atoms=10)
    assert drift["slope_ev_per_fs"] == pytest.approx(1e-4, rel=1e-9)
    assert drift["slope_ev_per_atom_per_ps"] == pytest.approx(1e-4 * 1000 / 10, rel=1e-9)


def test_kinetic_temperature_matches_equipartition_formula():
    v = np.full((1, 4, 3), 0.01)
    m = np.full(4, 12.0)
    ke = 0.5 * 12.0 * 4 * 3 * 0.01**2 * tm.AMU_ANG2_PER_FS2_TO_EV
    assert tm.kinetic_temperature(v, m)[0] == pytest.approx(2 * ke / (9 * tm.KB_EV_PER_K))


def test_orb_models_are_excluded_from_nve_check():
    assert "orb" in NVE_EXCLUDED_MODELS
    with pytest.raises(ValueError, match="non-conservative"):
        require_nve_eligible("Orb-v3")
    out = nve_conservation_report("orb_v2", TIMES_FS, np.full(100, -5.0), 10)
    assert out["skipped"] is True and out["passed"] is None and "non-conservative" in out["reason"]


def test_conservative_model_nve_report_runs_and_passes_for_constant_energy():
    out = nve_conservation_report("mace", TIMES_FS, np.full(100, -5.0), 10)
    assert out["skipped"] is False and out["passed"] is True


def test_stable_trajectory_passes_and_report_is_json_serialisable():
    report = stability_report(CHAIN_TRAJ, np.full(20, -3.0), np.full(20, 300.0), CHAIN, 300.0, bonds=BONDS)
    assert report["passed"] is True
    json.dumps(report)


def test_nan_positions_flag_explosion():
    traj = CHAIN_TRAJ.copy()
    traj[7, 0, 0] = np.nan
    report = stability_report(traj, np.full(20, -3.0), np.full(20, 300.0), CHAIN, 300.0, bonds=BONDS)
    assert report["passed"] is False
    assert report["checks"]["explosion"]["first_bad_frame"] == 7
    assert report["checks"]["bond_excursion"]["skipped"] is True


def test_stretched_bond_flags_excursion():
    traj = CHAIN_TRAJ.copy()
    traj[12:, 2, 0] += 3.0
    report = stability_report(traj, np.full(20, -3.0), np.full(20, 300.0), CHAIN, 300.0, bonds=BONDS)
    assert report["checks"]["bond_excursion"]["passed"] is False
    assert report["checks"]["bond_excursion"]["first_violating_frame"] == 12


def test_energy_and_temperature_runaway_flagged():
    energies = np.full(20, -3.0)
    energies[-1] = 100.0
    temps = np.full(20, 300.0)
    temps[-1] = 5000.0
    report = stability_report(CHAIN_TRAJ, energies, temps, CHAIN, 300.0, bonds=BONDS)
    assert report["checks"]["energy_runaway"]["passed"] is False
    assert report["checks"]["temperature_runaway"]["passed"] is False
