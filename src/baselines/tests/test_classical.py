from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("openmm")
pytest.importorskip("rdkit")
pytest.importorskip("openff.toolkit")
pytest.importorskip("openmmforcefields")
pytest.importorskip("ase")

from ase import Atoms  # noqa: E402
from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

from baselines.classical import forcefield as ff  # noqa: E402
from baselines.classical import run as crun  # noqa: E402
from baselines.config.units import EV_TO_KJ_PER_MOL  # noqa: E402
from baselines.export.extxyz import read_extxyz, write_extxyz  # noqa: E402


def _have_antechamber() -> bool:
    import shutil

    return shutil.which("antechamber") is not None and shutil.which("sqm") is not None


needs_ambertools = pytest.mark.skipif(not _have_antechamber(), reason="AmberTools (antechamber, sqm) not on PATH; activate the env")


def _molecule(smiles: str, seed: int = 1):
    # An embedded molecule with atom-map numbers (index + 1) on every atom, like SPICE, plus its coordinates.
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=seed) == 0
    AllChem.MMFFOptimizeMolecule(mol)
    positions = mol.GetConformer().GetPositions()
    for a in mol.GetAtoms():
        a.SetAtomMapNum(a.GetIdx() + 1)
    return Chem.MolToSmiles(mol), [a.GetAtomicNum() for a in mol.GetAtoms()], positions


def _jitter(positions, n, scale=0.03, seed=0):
    # Rounded to the 10 decimals the extxyz writer keeps: reference forces must come from the stored coordinates. Otherwise the 5e-11 A
    # rounding, amplified by stiff bonds (k ~ 30 eV/A^2), shows up as a ~1e-6 meV/A "force error" (measured 1.2e-6) that is a fixture artifact.
    g = np.random.default_rng(seed)
    return [np.round(positions + g.normal(scale=scale, size=positions.shape), 10) for _ in range(n)]


def test_kj_per_mol_per_ev_is_the_faraday_constant_over_1000():
    # Faraday constant F = N_A * e = 96485.33212... C/mol (CODATA 2018 value 96485.33212 is the rounded exact SI-2019 product).
    assert EV_TO_KJ_PER_MOL == pytest.approx(96.48533212, abs=5e-9)


def test_openmm_calculator_matches_a_hand_computed_harmonic_bond():
    import openmm

    k, r0 = 3000.0, 0.15  # kJ/mol/nm^2, nm
    system = openmm.System()
    for _ in range(2):
        system.addParticle(12.0)
    bond = openmm.HarmonicBondForce()
    bond.addBond(0, 1, r0, k)
    system.addForce(bond)
    calc = ff.OpenMMCalculator(system)
    atoms = Atoms("CC", positions=[[0, 0, 0], [1.7, 0, 0]], calculator=calc)  # 1.7 A = 0.17 nm: stretched by 0.02 nm
    e_kj = 0.5 * k * 0.02**2
    f_kj_nm = k * 0.02  # magnitude, pulling the atoms together
    assert atoms.get_potential_energy() == pytest.approx(e_kj / EV_TO_KJ_PER_MOL, rel=1e-12)
    f_ev_a = f_kj_nm / EV_TO_KJ_PER_MOL / 10.0
    np.testing.assert_allclose(atoms.get_forces(), [[f_ev_a, 0, 0], [-f_ev_a, 0, 0]], rtol=1e-9, atol=1e-15)


def test_mapped_smiles_reorders_atoms_to_the_dataset_order():
    # Atom maps are deliberately scrambled relative to the SMILES string order.
    mol = ff.rdmol_from_mapped_smiles("[H:3][O:1][C:2]([H:4])([H:5])[H:6]", 6)
    assert [a.GetAtomicNum() for a in mol.GetAtoms()] == [8, 6, 1, 1, 1, 1]
    assert mol.GetBondBetweenAtoms(0, 1) is not None and mol.GetBondBetweenAtoms(0, 2) is not None  # O-C and O-H
    with pytest.raises(ff.TopologyError, match="atom maps"):
        ff.rdmol_from_mapped_smiles("[H:3][O:1][C:2]([H:4])([H:5])[H:6]", 7)


def test_geometry_topology_recovers_bonds_and_orders():
    smiles, numbers, pos = _molecule("CC(=O)O")  # acetic acid: has a C=O double bond
    mol = ff.rdmol_from_geometry(numbers, pos)
    ref = ff.rdmol_from_mapped_smiles(smiles, len(numbers))
    got = {(min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()), max(b.GetBeginAtomIdx(), b.GetEndAtomIdx())): str(b.GetBondType()) for b in mol.GetBonds()}
    want = {(min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()), max(b.GetBeginAtomIdx(), b.GetEndAtomIdx())): str(b.GetBondType()) for b in ref.GetBonds()}
    assert got == want
    assert ff.bonds_from_geometry(numbers, pos) == set(got)


def test_geometry_topology_refuses_oversize_molecules(monkeypatch):
    _, numbers, pos = _molecule("CCO")
    monkeypatch.setattr(ff, "MAX_GEOMETRY_ATOMS", 5)
    with pytest.raises(ff.TopologyError, match="limit"):
        ff.rdmol_from_geometry(numbers, pos)


@needs_ambertools
def test_charges_are_cached_by_graph_and_reused(tmp_path):
    smiles, numbers, pos = _molecule("CCO")
    a = ff.to_openff(ff.rdmol_from_mapped_smiles(smiles, len(numbers)), pos)
    assert ff.assign_charges(a, str(tmp_path)) is False
    b = ff.to_openff(ff.rdmol_from_mapped_smiles(smiles, len(numbers)), pos)
    assert ff.assign_charges(b, str(tmp_path)) is True
    np.testing.assert_array_equal(a.partial_charges.m, b.partial_charges.m)
    assert abs(float(a.partial_charges.m.sum())) < 1e-6  # neutral molecule


@needs_ambertools
@pytest.mark.parametrize("name", ["gaff2", "openff"])
def test_forces_are_the_negative_energy_gradient(tmp_path, name):
    smiles, numbers, pos = _molecule("CCO")
    calc = ff.ClassicalCalculator(name, str(tmp_path))
    atoms = Atoms(numbers=numbers, positions=pos + 0.02, info={"smiles": smiles}, calculator=calc)
    analytic = atoms.get_forces()
    h = 1e-5  # Angstrom
    numeric = np.zeros_like(analytic)
    for i in range(len(atoms)):
        for d in range(3):
            e = []
            for sign in (+1, -1):
                p = atoms.get_positions().copy()
                p[i, d] += sign * h
                a2 = Atoms(numbers=numbers, positions=p, info={"smiles": smiles}, calculator=calc)
                e.append(a2.get_potential_energy())
            numeric[i, d] = -(e[0] - e[1]) / (2 * h)
    err = np.abs(analytic - numeric).max()
    print(f"{name}: max |analytic - central difference| = {err:.3e} eV/A (max |F| = {np.abs(analytic).max():.3f})")
    # The residual is central-difference truncation error, not a force error: measured for gaff2/ethanol it falls by exactly 4x per halving
    # of h (2.1e-6, 5.3e-7, 1.3e-7, 3.3e-8, 8.2e-9, 1.3e-9 eV/A at h = 4e-4 ... 1e-5 A). At h = 1e-5 A the truncation error is 1.3e-9 and
    # double-precision roundoff (~1e-16 * |E| / h ~ 1e-11) is negligible, so 1e-8 leaves 7x headroom over the measured value.
    assert err < 1e-8


def _write_frames(path, rows):
    write_extxyz(str(path), rows)


@needs_ambertools
def test_evaluate_classical_recovers_known_offsets_and_perfect_forces(tmp_path):
    smiles, numbers, pos = _molecule("CCO")
    calc = ff.ClassicalCalculator("gaff2", str(tmp_path / "q"))
    rows = []
    for p in _jitter(pos, 4):
        a = Atoms(numbers=numbers, positions=p, info={"smiles": smiles}, calculator=calc)
        rows.append({"atomic_numbers": numbers, "positions": p, "energy": a.get_potential_energy() + 5.0, "forces": a.get_forces(), "tags": {"smiles": smiles}})
    _write_frames(tmp_path / "test.xyz", rows)
    m = crun.evaluate_classical(str(tmp_path / "test.xyz"), "gaff2", str(tmp_path / "m.json"), str(tmp_path / "q"))
    assert m["energy_mae_meV"] == pytest.approx(5000.0, abs=1e-5)  # +5 eV on every frame, exactly (energies stored to 17 digits)
    assert m["energy_relative_mae_meV"] == pytest.approx(0.0, abs=1e-5)
    # Only the 10-decimal rounding of the stored reference forces remains: <= 5e-11 eV/A = 5e-8 meV/A per component.
    assert m["force_mae_meV_per_A"] < 5e-8
    assert m["n_molecules_parameterized"] == 1 and m["n_structures_skipped"] == 0
    assert json.load(open(tmp_path / "m.json"))["forcefield"] == "gaff-2.11"


@needs_ambertools
def test_per_molecule_offsets_need_the_group_key(tmp_path):
    calc = ff.ClassicalCalculator("openff", str(tmp_path / "q"))
    rows = []
    for smi, shift in (("CCO", 5.0), ("CO", -3.0)):
        smiles, numbers, pos = _molecule(smi)
        for p in _jitter(pos, 3):
            a = Atoms(numbers=numbers, positions=p, info={"smiles": smiles}, calculator=calc)
            rows.append({"atomic_numbers": numbers, "positions": p, "energy": a.get_potential_energy() + shift, "forces": a.get_forces(), "tags": {"smiles": smiles}})
    _write_frames(tmp_path / "test.xyz", rows)
    m = crun.evaluate_classical(str(tmp_path / "test.xyz"), "openff", None, str(tmp_path / "q"))
    assert m["n_relative_groups"] == 2 and m["n_relative_structures"] == 6
    assert m["energy_relative_mae_meV"] == pytest.approx(0.0, abs=1e-5)  # one offset per molecule removes the +5 / -3 eV exactly
    from baselines.evaluation.evaluate import evaluate_calculator

    pooled = evaluate_calculator(calc, read_extxyz(str(tmp_path / "test.xyz")), relative=True)
    assert pooled["energy_relative_mae_meV"] > 1000.0  # a single pooled offset cannot absorb two different shifts


@needs_ambertools
def test_unparameterizable_molecules_are_reported_not_hidden(tmp_path):
    smiles, numbers, pos = _molecule("CCO")
    rows = [{"atomic_numbers": numbers, "positions": p, "energy": 0.0, "forces": np.zeros((len(numbers), 3)), "tags": {"smiles": smiles}} for p in _jitter(pos, 2)]
    g = np.random.default_rng(0)
    rows.append({"atomic_numbers": [14, 1, 1, 1, 1], "positions": g.normal(size=(5, 3)), "energy": 0.0, "forces": np.zeros((5, 3)),
                 "tags": {"smiles": "[H:2][Si:1]([H:3])([H:4])[H:5]"}})  # element outside the GAFF2 organic set
    rows.append({"atomic_numbers": [6, 1], "positions": g.normal(size=(2, 3)), "energy": 0.0, "forces": np.zeros((2, 3)),
                 "tags": {"smiles": "[C:1][H:5]"}})  # atom maps are not 1..n: the graph cannot be aligned to the frame
    _write_frames(tmp_path / "test.xyz", rows)
    m = crun.evaluate_classical(str(tmp_path / "test.xyz"), "gaff2", None, str(tmp_path / "q"))
    assert m["n_structures_total"] == 4 and m["n_structures_skipped"] == 2 and m["n_molecules_skipped"] == 2
    assert m["n_molecules_parameterized"] == 1 and m["n_structures"] == 2
    reasons = " | ".join(m["skipped_reasons"].values())
    assert "organic set" in reasons and "atom maps" in reasons


@needs_ambertools
def test_charge_failure_is_reported_as_a_skip(tmp_path, monkeypatch):
    smiles, numbers, pos = _molecule("CCO")

    def boom(off_mol, cache_dir=None):
        raise ff.ParameterizationError("AM1-BCC failed: SCF did not converge")

    monkeypatch.setattr(ff, "assign_charges", boom)
    rows = [{"atomic_numbers": numbers, "positions": pos, "energy": 0.0, "forces": np.zeros((len(numbers), 3)), "tags": {"smiles": smiles}}]
    _write_frames(tmp_path / "test.xyz", rows)
    with pytest.raises(RuntimeError, match="no molecule could be parameterized.*SCF"):
        crun.evaluate_classical(str(tmp_path / "test.xyz"), "gaff2")


@needs_ambertools
def test_geometry_topology_path_for_fixed_composition_datasets(tmp_path):
    _, numbers, pos = _molecule("CC(=O)O")
    calc = ff.ClassicalCalculator("gaff2", str(tmp_path / "q"))
    rows = []
    for p in _jitter(pos, 5):
        a = Atoms(numbers=numbers, positions=p, calculator=calc)
        rows.append({"atomic_numbers": numbers, "positions": p, "energy": a.get_potential_energy() - 2.0, "forces": a.get_forces()})
    _write_frames(tmp_path / "test.xyz", rows)
    m = crun.evaluate_classical(str(tmp_path / "test.xyz"), "gaff2", None, str(tmp_path / "q"))
    assert m["energy_mae_meV"] == pytest.approx(2000.0, abs=1e-5) and m["energy_relative_mae_meV"] == pytest.approx(0.0, abs=1e-5)
    assert m["topology_frames_checked"] == 5 and m["topology_frames_mismatched"] == 0
    assert "n_relative_groups" not in m  # no smiles tag -> one pooled offset, as for rMD17 / MD22


@needs_ambertools
def test_topology_drift_counts_frames_whose_bonds_changed(tmp_path):
    _, numbers, pos = _molecule("CCO")
    calc = ff.ClassicalCalculator("gaff2", str(tmp_path / "q"))
    good = Atoms(numbers=numbers, positions=pos)
    calc.prepare(good)
    broken_pos = pos.copy()
    broken_pos[-1] += [5.0, 0, 0]  # pull the hydroxyl hydrogen far away: that bond is gone
    d = crun.topology_drift(calc, [good, Atoms(numbers=numbers, positions=broken_pos)])
    assert d == {"topology_frames_checked": 2, "topology_frames_mismatched": 1}


def test_mixed_compositions_without_smiles_are_refused(tmp_path):
    rows = [{"atomic_numbers": z, "positions": np.random.default_rng(0).normal(size=(len(z), 3)), "energy": 0.0} for z in ([8, 1, 1], [6, 1, 1, 1, 1])]
    _write_frames(tmp_path / "test.xyz", rows)
    with pytest.raises(ValueError, match="different compositions"):
        crun.evaluate_classical(str(tmp_path / "test.xyz"), "gaff2")


@pytest.mark.parametrize("dataset", ["ani1x", "qm9"])
def test_cli_refuses_datasets_without_topology(tmp_path, dataset):
    (tmp_path / "split.json").write_text(json.dumps({"dataset": dataset, "splits": {"test": {"file": "test.xyz"}}}))
    with pytest.raises(SystemExit, match="no bond topology"):
        crun.main(["--data-dir", str(tmp_path), "--forcefield", "gaff2", "--out", str(tmp_path / "m.json")])


def test_unknown_forcefield_is_rejected():
    with pytest.raises(ValueError, match="forcefield must be"):
        ff.ClassicalCalculator("amber99")
