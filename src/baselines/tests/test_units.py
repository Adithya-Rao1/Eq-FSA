# Tests for src/baselines/config/units.py
# references: CODATA 2018 (physics.nist.gov/cuu/Constants) -- E_h = 27.211386245988(52) eV, a_0 = 0.529177210903(80) Angstrom, alpha = 7.2973525693(11)e-3, m_e c^2 = 0.51099895000(15) MeV; exact SI-2019 defining constants h, c, e, N_A; thermochemical 1 kcal = 4184 J.
from __future__ import annotations

import math
import sys
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import numpy as np
import pytest
import torch

from baselines.config import units

# CODATA 2018 recommended values.
HARTREE_EV_CODATA = 27.211386245988
BOHR_ANGSTROM_CODATA = 0.529177210903

# Exact SI-2019 defining constants.
PLANCK_J_S = 6.62607015e-34
LIGHT_M_S = 299792458.0
ELEMENTARY_CHARGE_C = 1.602176634e-19
AVOGADRO_PER_MOL = 6.02214076e23
KCAL_J = 4184.0

# CODATA 2018 measured constants (finite uncertainty).
FINE_STRUCTURE_ALPHA = 7.2973525693e-3
ELECTRON_REST_ENERGY_EV = 510998.95000

# 1 kcal/mol = 4184 J / (N_A * e).
KCAL_PER_MOL_EV_EXACT = KCAL_J / (AVOGADRO_PER_MOL * ELEMENTARY_CHARGE_C)
HARTREE_PER_BOHR_EV_PER_ANGSTROM = HARTREE_EV_CODATA / BOHR_ANGSTROM_CODATA

# float64: one multiply/divide rounds by <= 2^-53 ~ 1.1e-16 relative, and parsing a decimal literal adds one more;
RTOL_F64 = 1e-15
# Float32 (eps 1.19e-7, unit roundoff 6e-8): a single conversion = scalar cast to float32 + one product rounding <= 1.2e-7.
RTOL_F32_SINGLE = 2e-7
# Float32 round trip = forward (cast + product) + inverse (cast or reciprocal + product) <= 4 x 6e-8 = 2.4e-7; 5e-7 gives ~2x margin.
RTOL_F32_ROUND_TRIP = 5e-7
# Reference values quoted to 6-7 significant figures: 0.0433641 vs true 0.04336410424 is off by 9.8e-8 relative,
# 23.06055 vs true 23.0605478 is off by 9.5e-8 relative; 1e-6 is ~10x above both.
RTOL_QUOTED_DIGITS = 1e-6

SCALARS = [0.0, 1.0, -1.0, 0.3, -2.75, 1234.5678, 1e-9, -1e6]

FORWARD_INVERSE_PAIRS = [
    (units.hartree_to_ev, units.ev_to_hartree),
    (units.bohr_to_angstrom, units.angstrom_to_bohr),
    (units.kcal_per_mol_to_ev, units.ev_to_kcal_per_mol),
    (units.hartree_per_bohr_to_ev_per_angstrom, units.ev_per_angstrom_to_hartree_per_bohr),
]
ALL_CONVERTERS = [fn for pair in FORWARD_INVERSE_PAIRS for fn in pair] + [
    units.kcal_per_mol_per_angstrom_to_ev_per_angstrom,
]


def _assert_close(actual, expected, rtol: float) -> None:
    np.testing.assert_allclose(np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64), rtol=rtol, atol=0.0)


def test_reference_constants_agree_with_first_principles_derivation():
    # E_h = alpha^2 m_e c^2 and a_0 = hbar c / (alpha m_e c^2) --> the CODATA literals must follow from the measured alpha and m_e c^2.
    hartree_ev = FINE_STRUCTURE_ALPHA**2 * ELECTRON_REST_ENERGY_EV
    hbar_c_ev_m = PLANCK_J_S * LIGHT_M_S / (2 * math.pi * ELEMENTARY_CHARGE_C)
    bohr_angstrom = hbar_c_ev_m / (FINE_STRUCTURE_ALPHA * ELECTRON_REST_ENERGY_EV) * 1e10
    # alpha is quoted to 11 significant digits (rounding <= 7e-12 relative, doubled by squaring), so 1e-10 is >5x above that.
    assert hartree_ev == pytest.approx(HARTREE_EV_CODATA, rel=1e-10, abs=0.0)
    assert bohr_angstrom == pytest.approx(BOHR_ANGSTROM_CODATA, rel=1e-10, abs=0.0)


def test_hartree_to_ev_matches_codata():
    for x in SCALARS:
        _assert_close(units.hartree_to_ev(x), x * HARTREE_EV_CODATA, RTOL_F64)
    assert units.hartree_to_ev(1.0) == pytest.approx(27.211386245988, rel=RTOL_F64, abs=0.0)


def test_ev_to_hartree_matches_codata():
    for x in SCALARS:
        _assert_close(units.ev_to_hartree(x), x / HARTREE_EV_CODATA, RTOL_F64)
    assert units.ev_to_hartree(27.211386245988) == pytest.approx(1.0, rel=RTOL_F64, abs=0.0)


def test_bohr_to_angstrom_matches_codata():
    for x in SCALARS:
        _assert_close(units.bohr_to_angstrom(x), x * BOHR_ANGSTROM_CODATA, RTOL_F64)
    assert units.bohr_to_angstrom(1.0) == pytest.approx(0.529177210903, rel=RTOL_F64, abs=0.0)


def test_angstrom_to_bohr_matches_codata():
    for x in SCALARS:
        _assert_close(units.angstrom_to_bohr(x), x / BOHR_ANGSTROM_CODATA, RTOL_F64)
    # 1 Angstrom = 1.8897261246... Bohr (reciprocal of a_0 in Angstrom).
    assert units.angstrom_to_bohr(1.0) == pytest.approx(1.8897261246, rel=1e-10, abs=0.0)


def test_kcal_per_mol_scale_matches_six_figure_reference():
    assert units.kcal_per_mol_to_ev(1.0) == pytest.approx(0.0433641, rel=RTOL_QUOTED_DIGITS, abs=0.0)
    assert units.ev_to_kcal_per_mol(1.0) == pytest.approx(23.06055, rel=RTOL_QUOTED_DIGITS, abs=0.0)


def test_kcal_per_mol_constant_matches_exact_si_derivation():
    # 4184 J/mol / (N_A e)
    assert units.kcal_per_mol_to_ev(1.0) == pytest.approx(KCAL_PER_MOL_EV_EXACT, rel=RTOL_F64, abs=0.0)
    assert units.ev_to_kcal_per_mol(KCAL_PER_MOL_EV_EXACT) == pytest.approx(1.0, rel=RTOL_F64, abs=0.0)


def test_hartree_per_bohr_force_factor_is_energy_over_length_factor():
    # CODATA atomic unit of electric field, 5.14220674763e11 V/m = 51.4220674763 V/Angstrom, to 12 digits (rounding ~1e-12).
    assert units.hartree_per_bohr_to_ev_per_angstrom(1.0) == pytest.approx(51.4220674763, rel=1e-11, abs=0.0)
    for x in SCALARS:
        _assert_close(units.hartree_per_bohr_to_ev_per_angstrom(x), x * HARTREE_PER_BOHR_EV_PER_ANGSTROM, RTOL_F64)
        _assert_close(units.ev_per_angstrom_to_hartree_per_bohr(x), x / HARTREE_PER_BOHR_EV_PER_ANGSTROM, RTOL_F64)


def test_hartree_per_bohr_factor_is_not_the_bare_energy_factor():
    assert units.hartree_per_bohr_to_ev_per_angstrom(1.0) != pytest.approx(units.hartree_to_ev(1.0), rel=1e-3)


def test_kcal_per_mol_per_angstrom_force_uses_energy_factor_only():
    for x in SCALARS:
        # Same Angstrom on both sides, so the force factor = energy factor.
        assert units.kcal_per_mol_per_angstrom_to_ev_per_angstrom(x) == units.kcal_per_mol_to_ev(x)
    assert units.kcal_per_mol_per_angstrom_to_ev_per_angstrom(1.0) == pytest.approx(0.0433641, rel=RTOL_QUOTED_DIGITS, abs=0.0)


def test_force_conversion_is_negative_gradient_of_converted_energy():
    # Harmonic well E = k x^2 / 2 in Hartree/Bohr. The converted force must be -dE_eV/dx_Angstrom of the converted energy.
    k_hartree_bohr2, x_bohr = 0.3, 0.7
    x_angstrom = units.bohr_to_angstrom(x_bohr)
    h_angstrom = 1e-2

    def energy_ev_at(x_ang: float) -> float:
        return units.hartree_to_ev(0.5 * k_hartree_bohr2 * units.angstrom_to_bohr(x_ang) ** 2)

    # cancellation error ~ eps * |E| / (h * |F|) ~ 1e-14
    numeric_force = -(energy_ev_at(x_angstrom + h_angstrom) - energy_ev_at(x_angstrom - h_angstrom)) / (2 * h_angstrom)
    converted_force = units.hartree_per_bohr_to_ev_per_angstrom(-k_hartree_bohr2 * x_bohr)
    assert converted_force == pytest.approx(numeric_force, rel=1e-9, abs=0.0)


@pytest.mark.parametrize("forward, inverse", FORWARD_INVERSE_PAIRS)
def test_round_trip_identity_python_float(forward, inverse):
    for x in SCALARS:
        # forward then inverse (and the reverse order) each round at most ~3 times in float64.
        _assert_close(inverse(forward(x)), x, RTOL_F64 * 3)
        _assert_close(forward(inverse(x)), x, RTOL_F64 * 3)


@pytest.mark.parametrize("forward, inverse", FORWARD_INVERSE_PAIRS)
def test_round_trip_identity_float64_numpy_and_torch(forward, inverse):
    values = np.array([[0.5, -1.25, 3.0], [1e-7, -4e5, 0.0]], dtype=np.float64)
    _assert_close(inverse(forward(values)), values, RTOL_F64 * 3)
    _assert_close(inverse(forward(torch.from_numpy(values))).numpy(), values, RTOL_F64 * 3)


@pytest.mark.parametrize("forward, inverse", FORWARD_INVERSE_PAIRS)
def test_round_trip_identity_float32(forward, inverse):
    values = np.array([[0.5, -1.25, 3.0], [1e-7, -4e5, 0.0]], dtype=np.float32)
    _assert_close(inverse(forward(values)), values, RTOL_F32_ROUND_TRIP)
    _assert_close(inverse(forward(torch.from_numpy(values))).numpy(), values, RTOL_F32_ROUND_TRIP)


@pytest.mark.parametrize("fn", ALL_CONVERTERS)
def test_python_float_in_python_float_out(fn):
    assert isinstance(fn(1.5), float)


@pytest.mark.parametrize("fn", ALL_CONVERTERS)
def test_numpy_preserves_shape_and_dtype(fn):
    for dtype in (np.float32, np.float64):
        array = np.arange(24, dtype=dtype).reshape(2, 3, 4) - 7
        result = fn(array)
        assert isinstance(result, np.ndarray)
        assert result.shape == array.shape
        assert result.dtype == dtype


@pytest.mark.parametrize("fn", ALL_CONVERTERS)
def test_torch_preserves_shape_dtype_and_type(fn):
    for dtype in (torch.float32, torch.float64):
        tensor = torch.arange(24, dtype=dtype).reshape(2, 3, 4) - 7
        result = fn(tensor)
        assert isinstance(result, torch.Tensor)
        assert result.shape == tensor.shape
        assert result.dtype == dtype


def test_numpy_and_torch_values_match_python_float_reference():
    values = [0.5, -1.25, 3.0, 17.0]
    for fn in ALL_CONVERTERS:
        expected = np.array([fn(v) for v in values])
        _assert_close(fn(np.array(values, dtype=np.float64)), expected, RTOL_F64)
        _assert_close(fn(torch.tensor(values, dtype=torch.float64)).numpy(), expected, RTOL_F64)
        _assert_close(fn(np.array(values, dtype=np.float32)), expected, RTOL_F32_SINGLE)
        _assert_close(fn(torch.tensor(values, dtype=torch.float32)).numpy(), expected, RTOL_F32_SINGLE)


def test_torch_conversion_keeps_autograd_graph():
    x = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
    units.hartree_to_ev(x).sum().backward()
    _assert_close(x.grad.numpy(), [HARTREE_EV_CODATA, HARTREE_EV_CODATA], RTOL_F64)


def test_to_ev_angstrom_hartree_bohr_numpy():
    energy = np.array([-40.5, 0.25], dtype=np.float64)
    forces = np.array([[0.1, -0.2, 0.3], [0.0, 1.0, -1.0]], dtype=np.float64)
    energy_ev, forces_ev = units.to_ev_angstrom(energy, forces, "hartree_bohr")
    _assert_close(energy_ev, energy * HARTREE_EV_CODATA, RTOL_F64)
    _assert_close(forces_ev, forces * HARTREE_PER_BOHR_EV_PER_ANGSTROM, RTOL_F64)


def test_to_ev_angstrom_kcal_mol_angstrom_numpy():
    energy = np.array([-400.0, 12.5], dtype=np.float64)
    forces = np.array([[10.0, -20.0, 30.0], [0.0, 1.0, -1.0]], dtype=np.float64)
    energy_ev, forces_ev = units.to_ev_angstrom(energy, forces, "kcal_mol_angstrom")
    _assert_close(energy_ev, energy * 0.0433641, RTOL_QUOTED_DIGITS)
    _assert_close(forces_ev, forces * 0.0433641, RTOL_QUOTED_DIGITS)
    assert np.array_equal(energy_ev, units.kcal_per_mol_to_ev(energy))
    assert np.array_equal(forces_ev, units.kcal_per_mol_per_angstrom_to_ev_per_angstrom(forces))


def test_to_ev_angstrom_ev_angstrom_is_identity_for_every_container():
    energy = np.array([-1.5, 2.0], dtype=np.float32)
    forces = np.arange(6, dtype=np.float32).reshape(2, 3)
    energy_out, forces_out = units.to_ev_angstrom(energy, forces, "ev_angstrom")
    assert np.array_equal(energy_out, energy) and energy_out.dtype == np.float32
    assert np.array_equal(forces_out, forces) and forces_out.dtype == np.float32

    energy_t, forces_t = torch.tensor([-1.5, 2.0]), torch.arange(6, dtype=torch.float32).reshape(2, 3)
    energy_t_out, forces_t_out = units.to_ev_angstrom(energy_t, forces_t, "ev_angstrom")
    assert torch.equal(energy_t_out, energy_t) and torch.equal(forces_t_out, forces_t)

    assert units.to_ev_angstrom(1.0, 1.0, "ev_angstrom") == (1.0, 1.0)


@pytest.mark.parametrize("source", ["hartree_bohr", "kcal_mol_angstrom", "ev_angstrom"])
def test_to_ev_angstrom_torch_preserves_shape_and_dtype(source):
    energy = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    forces = torch.ones(3, 4, 3, dtype=torch.float32)
    energy_ev, forces_ev = units.to_ev_angstrom(energy, forces, source)
    assert energy_ev.shape == energy.shape and energy_ev.dtype == torch.float32
    assert forces_ev.shape == forces.shape and forces_ev.dtype == torch.float32


def test_to_ev_angstrom_torch_float32_matches_reference():
    energy = torch.tensor([-40.5, 0.25], dtype=torch.float32)
    forces = torch.tensor([[0.1, -0.2, 0.3]], dtype=torch.float32)
    energy_ev, forces_ev = units.to_ev_angstrom(energy, forces, "hartree_bohr")
    _assert_close(energy_ev.numpy(), energy.double().numpy() * HARTREE_EV_CODATA, RTOL_F32_SINGLE)
    _assert_close(forces_ev.numpy(), forces.double().numpy() * HARTREE_PER_BOHR_EV_PER_ANGSTROM, RTOL_F32_SINGLE)


def test_to_ev_angstrom_python_floats():
    energy_ev, forces_ev = units.to_ev_angstrom(2.0, 3.0, "hartree_bohr")
    _assert_close(energy_ev, 2.0 * HARTREE_EV_CODATA, RTOL_F64)
    _assert_close(forces_ev, 3.0 * HARTREE_PER_BOHR_EV_PER_ANGSTROM, RTOL_F64)


def test_to_ev_angstrom_returns_a_two_tuple():
    result = units.to_ev_angstrom(1.0, 1.0, "hartree_bohr")
    assert isinstance(result, tuple) and len(result) == 2


@pytest.mark.parametrize("source", ["", "hartree", "Hartree_Bohr", "HARTREE_BOHR", "ev", "eV_angstrom", "kcal", " ev_angstrom", None])
def test_to_ev_angstrom_unknown_convention_raises_value_error(source):
    with pytest.raises(ValueError, match="unknown unit convention"):
        units.to_ev_angstrom(1.0, 1.0, source)


def test_to_ev_angstrom_error_lists_supported_conventions():
    with pytest.raises(ValueError) as excinfo:
        units.to_ev_angstrom(np.zeros(2), np.zeros((2, 3)), "bogus")
    message = str(excinfo.value)
    for name in ("hartree_bohr", "kcal_mol_angstrom", "ev_angstrom", "bogus"):
        assert name in message


def test_ev_to_kj_per_mol_is_exactly_avogadro_times_charge_over_1000():
    # 1 eV per particle = e * N_A J/mol = the Faraday constant, 96485.33212... C/mol * 1 V; exact in SI-2019.
    assert units.EV_TO_KJ_PER_MOL == pytest.approx(ELEMENTARY_CHARGE_C * AVOGADRO_PER_MOL / 1000.0, rel=RTOL_F64)
    assert units.EV_TO_KJ_PER_MOL == pytest.approx(96.48533212331, rel=1e-12)  # CODATA 2018 F = 96485.33212... C/mol
    assert units.EV_TO_KJ_PER_MOL / 4.184 == pytest.approx(1.0 / units.KCAL_PER_MOL_TO_EV, rel=RTOL_F64)  # same eV, kcal vs kJ
