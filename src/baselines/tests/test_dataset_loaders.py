# Tests for src/baselines/datasets/{qm9,rmd17,md22,ani1x,units_check}.py on tiny synthetic on-disk fixtures with hand-computed expected values.
# Formats: QM9 extended-XYZ (Ramakrishnan et al., Sci. Data 1, 140022 (2014): line 1 atom count, line 2 "gdb idx" + 15 properties with U0 in Hartree,
# then one "element x y z charge" line per atom; Mathematica "*^" exponents); sGDML/MD22 npz keys R,z,E,F (sgdml.org); ANI-1x HDF5 groups with
# atomic_numbers/coordinates (Angstrom)/<method>.energy (Hartree)/<method>.forces (Hartree/Angstrom) (Smith et al., Sci. Data 7, 134 (2020)).
# Unit references: CODATA 2018 E_h = 27.211386245988 eV; 1 kcal/mol = 4184 J / (N_A e) with exact SI-2019 N_A and e.
from __future__ import annotations

import importlib
import sys
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import numpy as np
import pytest
import torch

from baselines.datasets import md22, qm9, rmd17

HARTREE_EV = 27.211386245988
KCAL_PER_MOL_EV = 4184.0 / (6.02214076e23 * 1.602176634e-19)


RTOL_F64_ENERGY = 1e-14
RTOL_F32_ONE_CONVERSION = 2e-7
# ANI-1x forces conversion
RTOL_F32_TWO_STEP = 5e-7

QM9_ATOMS_METHANE_LIKE = [
    ("C", ("-0.0126981359", "1.0858041578", "0.0080009958")),
    ("H", ("0.002150416", "-0.0060313176", "1.5*^-5")),
    ("H", ("1.0117308433", "1.4637511618", "-2.0*^-6")),
    ("H", ("-0.540815069", "1.4475266138", "-0.8766437152")),
    ("H", ("-0.5238136345", "1.4379326443", "2.5*^1")),
]
QM9_ATOMS_METHANE_LIKE_PARSED = [
    [-0.0126981359, 1.0858041578, 0.0080009958],
    [0.002150416, -0.0060313176, 1.5e-5],
    [1.0117308433, 1.4637511618, -2.0e-6],
    [-0.540815069, 1.4475266138, -0.8766437152],
    [-0.5238136345, 1.4379326443, 25.0],
]
QM9_OTHER_PROPS = [
    "157.7118", "157.70997", "157.70699", "0.", "13.21", "-0.3877", "0.1171", "0.5048", "35.3641", "0.044749",
]  # A B C mu alpha homo lumo gap r2 zpve, in file order, before U0


def _write_qm9_xyz(path: Path, atoms: list[tuple[str, tuple[str, str, str]]], u0: str, extra_props: list[str] | None = None) -> None:
    props = QM9_OTHER_PROPS + [u0] + (["-40.476062", "-40.475117", "-40.498597", "6.469"] if extra_props is None else extra_props)
    lines = [str(len(atoms)), "gdb\t1\t" + "\t".join(props)]
    lines += [f"{el}\t{x}\t{y}\t{z}\t-0.535689" for el, (x, y, z) in atoms]
    lines += ["1341.307\t1341.3284\t1341.365", "C\tC", "InChI=1S/CH4/h1H4\tInChI=1S/CH4/h1H4"]
    path.write_text("\n".join(lines) + "\n")


def _write_qm9_directory(root: Path, n: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        # Molecule i has U0 = -(1 + i) Hartree so its index can be decoded back from the converted energy.
        _write_qm9_xyz(root / f"dsgdb9nsd_{i:06d}.xyz", [("H", ("0.0", "0.0", "0.0")), ("H", ("0.0", "0.0", "0.74"))], f"{-(1.0 + i):.6f}")
    (root / "README.txt").write_text("not a molecule file")


def _qm9_index(item: dict) -> int:
    return int(round(-float(item["energy"]) / HARTREE_EV)) - 1


def _indices(dataset) -> list[int]:
    return [_qm9_index(dataset[i]) for i in range(len(dataset))]


def _expected_permutation(n: int, seed: int) -> list[int]:
    return torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()


def _close(actual, expected, rtol: float) -> None:
    np.testing.assert_allclose(np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64), rtol=rtol, atol=0.0)


def test_parse_qm9_molecule_file_positions_numbers_and_energy(tmp_path):
    path = tmp_path / "mol.xyz"
    _write_qm9_xyz(path, QM9_ATOMS_METHANE_LIKE, "-40.47893")
    mol = qm9.parse_qm9_molecule_file(str(path))
    assert mol.atomic_numbers.dtype == torch.long
    assert mol.atomic_numbers.tolist() == [6, 1, 1, 1, 1]
    assert mol.positions.dtype == torch.float32
    assert mol.positions.shape == (5, 3)
    assert torch.equal(mol.positions, torch.tensor(QM9_ATOMS_METHANE_LIKE_PARSED, dtype=torch.float32))
    assert mol.energy.dtype == torch.float64
    assert mol.energy.shape == ()
    _close(mol.energy.item(), -40.47893 * HARTREE_EV, RTOL_F64_ENERGY)


def test_parse_qm9_molecule_file_handles_fortran_style_star_caret_exponents(tmp_path):
    path = tmp_path / "mol.xyz"
    _write_qm9_xyz(path, [("O", ("1.5*^-5", "-2.0*^-6", "3.25*^2")), ("H", ("-1.0*^-1", "0.0", "7.0*^0"))], "-76.4")
    mol = qm9.parse_qm9_molecule_file(str(path))
    assert torch.equal(mol.positions, torch.tensor([[1.5e-5, -2.0e-6, 325.0], [-0.1, 0.0, 7.0]], dtype=torch.float32))


def test_parse_qm9_molecule_file_reads_u0_not_neighbouring_columns(tmp_path):
    path = tmp_path / "mol.xyz"
    # U, H, G, Cv deliberately differ wildly from U0.
    _write_qm9_xyz(path, [("H", ("0", "0", "0"))], "-0.5", extra_props=["-111.0", "-222.0", "-333.0", "444.0"])
    mol = qm9.parse_qm9_molecule_file(str(path))
    _close(mol.energy.item(), -0.5 * HARTREE_EV, RTOL_F64_ENERGY)


def test_parse_qm9_molecule_file_maps_all_five_elements(tmp_path):
    path = tmp_path / "mol.xyz"
    atoms = [(el, ("0.0", "0.0", str(float(i)))) for i, el in enumerate(["H", "C", "N", "O", "F"])]
    _write_qm9_xyz(path, atoms, "-1.0")
    assert qm9.parse_qm9_molecule_file(str(path)).atomic_numbers.tolist() == [1, 6, 7, 8, 9]


def test_parse_qm9_molecule_file_ignores_trailer_lines_after_the_atoms(tmp_path):
    path = tmp_path / "mol.xyz"
    _write_qm9_xyz(path, [("H", ("0", "0", "0")), ("H", ("0", "0", "1"))], "-1.0")
    mol = qm9.parse_qm9_molecule_file(str(path))
    assert mol.positions.shape == (2, 3)


def test_parse_qm9_molecule_file_wrong_property_count_raises_value_error(tmp_path):
    path = tmp_path / "mol.xyz"
    _write_qm9_xyz(path, [("H", ("0", "0", "0"))], "-1.0", extra_props=["-1.0", "-1.0", "-1.0"])
    with pytest.raises(ValueError, match="expected 15 scalar properties"):
        qm9.parse_qm9_molecule_file(str(path))


def test_parse_qm9_molecule_file_unsupported_element_raises_key_error(tmp_path):
    path = tmp_path / "mol.xyz"
    _write_qm9_xyz(path, [("S", ("0", "0", "0"))], "-1.0")
    with pytest.raises(KeyError):
        qm9.parse_qm9_molecule_file(str(path))


def test_qm9_property_layout_constants():
    assert len(qm9.QM9_PROPERTY_NAMES) == 15
    assert qm9.QM9_U0_INDEX == 10
    assert qm9.QM9_TRAIN_SIZE + qm9.QM9_VAL_SIZE + qm9.QM9_TEST_SIZE == qm9.QM9_FULL_SIZE_AFTER_EXCLUSION == 130_831
    assert (qm9.QM9_TRAIN_SIZE, qm9.QM9_VAL_SIZE, qm9.QM9_TEST_SIZE) == (110_000, 10_000, 10_831)


def test_qm9_dataset_item_keys_and_length(tmp_path):
    path = tmp_path / "mol.xyz"
    _write_qm9_xyz(path, QM9_ATOMS_METHANE_LIKE, "-40.47893")
    mol = qm9.parse_qm9_molecule_file(str(path))
    dataset = qm9.QM9Dataset([mol, mol, mol])
    assert len(dataset) == 3
    item = dataset[1]
    assert set(item) == {"x", "atomic_numbers", "energy"}
    assert item["x"] is mol.positions and item["energy"] is mol.energy


def test_qm9_fixed_permutation_is_a_permutation_and_deterministic():
    perm = qm9.qm9_fixed_permutation(1000)
    assert sorted(perm) == list(range(1000))
    assert perm != list(range(1000))
    assert perm == qm9.qm9_fixed_permutation(1000)
    assert all(type(i) is int for i in perm)


def test_qm9_fixed_permutation_default_seed_is_zero_and_matches_torch_randperm():
    assert qm9.qm9_fixed_permutation(200) == _expected_permutation(200, seed=0)
    assert qm9.qm9_fixed_permutation(200, seed=5) == _expected_permutation(200, seed=5)


def test_qm9_fixed_permutation_different_seeds_differ_and_do_not_touch_global_rng():
    torch.manual_seed(321)
    before = torch.get_rng_state()
    assert qm9.qm9_fixed_permutation(200, seed=0) != qm9.qm9_fixed_permutation(200, seed=1)
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.parametrize("n, expected", [(0, []), (1, [0])])
def test_qm9_fixed_permutation_degenerate_sizes(n, expected):
    assert qm9.qm9_fixed_permutation(n) == expected


def test_load_qm9_split_sizes_follow_the_110000_10000_10831_proportions(tmp_path):
    # For n=50: round(50*110000/130831 = 42.04) = 42, round(50*10000/130831 = 3.82) = 4, rest = 4.
    _write_qm9_directory(tmp_path / "qm9", 50)
    train, val, test = qm9.load_qm9(str(tmp_path / "qm9"))
    assert (len(train), len(val), len(test)) == (42, 4, 4)
    assert len(train) + len(val) + len(test) == 50


def test_load_qm9_split_follows_fixed_permutation_with_no_overlap(tmp_path):
    _write_qm9_directory(tmp_path / "qm9", 50)
    train, val, test = qm9.load_qm9(str(tmp_path / "qm9"))
    perm = _expected_permutation(50, seed=0)
    assert _indices(train) == perm[:42]
    assert _indices(val) == perm[42:46]
    assert _indices(test) == perm[46:]
    assert sorted(_indices(train) + _indices(val) + _indices(test)) == list(range(50))


def test_load_qm9_is_deterministic_and_seed_sensitive(tmp_path):
    _write_qm9_directory(tmp_path / "qm9", 50)
    first = [_indices(d) for d in qm9.load_qm9(str(tmp_path / "qm9"))]
    second = [_indices(d) for d in qm9.load_qm9(str(tmp_path / "qm9"))]
    other_seed = [_indices(d) for d in qm9.load_qm9(str(tmp_path / "qm9"), seed=1)]
    assert first == second
    assert first != other_seed
    assert first == [_indices(d) for d in qm9.load_qm9(str(tmp_path / "qm9"), seed=qm9.QM9_SPLIT_SEED)]


def test_load_qm9_energy_is_converted_hartree_to_ev_and_files_sorted_by_name(tmp_path):
    root = tmp_path / "qm9"
    _write_qm9_directory(root, 5)
    train, val, test = qm9.load_qm9(str(root))
    perm = _expected_permutation(5, seed=0)
    everything = [d[i] for d in (train, val, test) for i in range(len(d))]
    for item, molecule_index in zip(everything, perm):
        _close(item["energy"].item(), -(1.0 + molecule_index) * HARTREE_EV, RTOL_F64_ENERGY)
        assert item["energy"].dtype == torch.float64


def test_load_qm9_ignores_non_xyz_files(tmp_path):
    _write_qm9_directory(tmp_path / "qm9", 50)
    (tmp_path / "qm9" / "notes.xyz.bak").write_text("junk")
    assert sum(len(d) for d in qm9.load_qm9(str(tmp_path / "qm9"))) == 50


def test_load_qm9_missing_files_raise_file_not_found_with_actionable_message(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match=r"No QM9 '\*\.xyz' files found"):
        qm9.load_qm9(str(tmp_path / "empty"))
    with pytest.raises(FileNotFoundError, match="No QM9"):
        qm9.load_qm9(str(tmp_path / "does_not_exist"))


RMD17_N_FRAMES = 2000
RMD17_N_ATOMS = 3
RMD17_ATOMIC_NUMBERS = [8, 1, 1]


def _frame_offsets(n_atoms: int) -> np.ndarray:
    return np.arange(n_atoms * 3, dtype=np.float64).reshape(n_atoms, 3) * 0.25


def _synthetic_trajectory(n_frames: int, n_atoms: int) -> dict[str, np.ndarray]:
    # Every value is a small multiple of 0.25, so float64 -> float32 is exact to isolate unit conversion errors.
    frames = np.arange(n_frames, dtype=np.float64)
    return {
        "R": frames[:, None, None] + _frame_offsets(n_atoms)[None],
        "z": np.array(RMD17_ATOMIC_NUMBERS[:n_atoms] + [1] * max(0, n_atoms - len(RMD17_ATOMIC_NUMBERS)), dtype=np.uint8),
        "E": frames * 0.25,
        "F": -(frames[:, None, None] + 2.0 * _frame_offsets(n_atoms)[None]),
    }


def _write_split_files(root: Path, split_index: int, train: list[int], test: list[int]) -> None:
    split_dir = root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / f"index_train_{split_index + 1:02d}.csv").write_text("\n".join(map(str, train)) + "\n")
    (split_dir / f"index_test_{split_index + 1:02d}.csv").write_text("\n".join(map(str, test)) + "\n")


def _write_rmd17_fixture(root: Path, molecule: str = "aspirin") -> None:
    (root / "npz_data").mkdir(parents=True)
    arrays = _synthetic_trajectory(RMD17_N_FRAMES, RMD17_N_ATOMS)
    np.savez(root / "npz_data" / f"rmd17_{molecule}.npz", R=arrays["R"], z=arrays["z"], E=arrays["E"], F=arrays["F"])
    even, odd = list(range(0, RMD17_N_FRAMES, 2)), list(range(1, RMD17_N_FRAMES, 2))
    _write_split_files(root, 0, train=even, test=odd)
    _write_split_files(root, 2, train=odd, test=even)


def _frame_of(item: dict) -> int:
    return int(round(float(item["x"][0, 0])))


def test_load_rmd17_split_sizes_and_empty_validation(tmp_path):
    _write_rmd17_fixture(tmp_path)
    train, val, test = rmd17.load_rmd17(str(tmp_path), "aspirin")
    assert (len(train), len(val), len(test)) == (1000, 0, 1000)


def test_load_rmd17_indexes_frames_through_the_fixed_split_files(tmp_path):
    _write_rmd17_fixture(tmp_path)
    train, _, test = rmd17.load_rmd17(str(tmp_path), "aspirin", split_index=0)
    assert [_frame_of(train[i]) for i in range(len(train))] == list(range(0, RMD17_N_FRAMES, 2))
    assert [_frame_of(test[i]) for i in range(len(test))] == list(range(1, RMD17_N_FRAMES, 2))
    assert not {_frame_of(train[i]) for i in range(len(train))} & {_frame_of(test[i]) for i in range(len(test))}


def test_load_rmd17_split_index_selects_a_different_split_file_pair(tmp_path):
    _write_rmd17_fixture(tmp_path)
    train, _, test = rmd17.load_rmd17(str(tmp_path), "aspirin", split_index=2)
    assert _frame_of(train[0]) == 1 and _frame_of(test[0]) == 0


def test_load_rmd17_converts_kcal_per_mol_to_ev_and_kcal_per_mol_angstrom_to_ev_per_angstrom(tmp_path):
    _write_rmd17_fixture(tmp_path)
    train, _, _ = rmd17.load_rmd17(str(tmp_path), "aspirin")
    arrays = _synthetic_trajectory(RMD17_N_FRAMES, RMD17_N_ATOMS)
    item = train[3]  # frame 6
    assert _frame_of(item) == 6
    _close(item["energy"].item(), arrays["E"][6] * KCAL_PER_MOL_EV, RTOL_F64_ENERGY)
    _close(item["forces"].numpy(), arrays["F"][6] * KCAL_PER_MOL_EV, RTOL_F32_ONE_CONVERSION)
    # Hand check: E[6] = 1.5 kcal/mol -> 0.0650 eV; F[6][0, 0] = -6.0 kcal/mol/A -> -0.2602 eV/A.
    assert item["energy"].item() == pytest.approx(1.5 * 0.0433641, rel=1e-6, abs=0.0)
    assert item["forces"][0, 0].item() == pytest.approx(-6.0 * 0.0433641, rel=1e-6, abs=0.0)


def test_load_rmd17_item_dtypes_shapes_and_positions_are_not_converted(tmp_path):
    _write_rmd17_fixture(tmp_path)
    _, _, test = rmd17.load_rmd17(str(tmp_path), "aspirin")
    item = test[0]  # frame 1
    arrays = _synthetic_trajectory(RMD17_N_FRAMES, RMD17_N_ATOMS)
    assert set(item) == {"x", "atomic_numbers", "energy", "forces"}
    assert item["x"].dtype == torch.float32 and item["x"].shape == (RMD17_N_ATOMS, 3)
    assert torch.equal(item["x"], torch.from_numpy(arrays["R"][1]).float())
    assert item["atomic_numbers"].dtype == torch.long and item["atomic_numbers"].tolist() == RMD17_ATOMIC_NUMBERS
    assert item["energy"].shape == () and item["energy"].dtype == torch.float64
    assert item["forces"].shape == (RMD17_N_ATOMS, 3) and item["forces"].dtype == torch.float32


def test_load_rmd17_is_deterministic(tmp_path):
    _write_rmd17_fixture(tmp_path)
    first = rmd17.load_rmd17(str(tmp_path), "aspirin")
    second = rmd17.load_rmd17(str(tmp_path), "aspirin")
    for a, b in zip(first, second):
        assert [_frame_of(a[i]) for i in range(len(a))] == [_frame_of(b[i]) for i in range(len(b))]


def test_load_rmd17_unknown_molecule_raises_value_error_before_touching_disk(tmp_path):
    with pytest.raises(ValueError, match="unknown rMD17 molecule"):
        rmd17.load_rmd17(str(tmp_path), "not_a_molecule")


def test_load_rmd17_missing_archive_raises_not_implemented_with_path(tmp_path):
    with pytest.raises(NotImplementedError, match="rMD17 archive not found") as excinfo:
        rmd17.load_rmd17(str(tmp_path), "aspirin")
    assert "rmd17_aspirin.npz" in str(excinfo.value)


def test_load_rmd17_missing_split_files_raise_not_implemented(tmp_path):
    (tmp_path / "npz_data").mkdir()
    arrays = _synthetic_trajectory(10, RMD17_N_ATOMS)
    np.savez(tmp_path / "npz_data" / "rmd17_aspirin.npz", R=arrays["R"], z=arrays["z"], E=arrays["E"], F=arrays["F"])
    with pytest.raises(NotImplementedError, match="fixed split files not found"):
        rmd17.load_rmd17(str(tmp_path), "aspirin")


def test_load_rmd17_out_of_range_split_index_raises_value_error(tmp_path):
    _write_rmd17_fixture(tmp_path)
    with pytest.raises(ValueError, match="split_index"):
        rmd17.load_rmd17(str(tmp_path), "aspirin", split_index=5)


def test_load_rmd17_reads_the_published_rmd17_npz_key_names(tmp_path):
    # rMD17 figshare archive's npz files documented as keyed nuclear_charges / coords / energies / forces (not the sGDML R/z/E/F).
    # TODO: test if this fails with KeyError: 'R' --> rmd17.load_rmd17 only works on synthetic data; confirm with np.load(real_file).files.
    (tmp_path / "npz_data").mkdir()
    arrays = _synthetic_trajectory(RMD17_N_FRAMES, RMD17_N_ATOMS)
    np.savez(
        tmp_path / "npz_data" / "rmd17_aspirin.npz",
        nuclear_charges=arrays["z"], coords=arrays["R"], energies=arrays["E"], forces=arrays["F"],
    )
    even, odd = list(range(0, RMD17_N_FRAMES, 2)), list(range(1, RMD17_N_FRAMES, 2))
    _write_split_files(tmp_path, 0, train=even, test=odd)
    train, _, _ = rmd17.load_rmd17(str(tmp_path), "aspirin")
    assert len(train) == 1000


MD22_N_ATOMS = 2


def _write_md22_fixture(root: Path, molecule: str, n_frames: int) -> dict[str, np.ndarray]:
    root.mkdir(parents=True, exist_ok=True)
    arrays = _synthetic_trajectory(n_frames, MD22_N_ATOMS)
    arrays["E"] = arrays["E"].reshape(-1, 1)  # sGDML stores energies as (N, 1)
    np.savez(root / f"md22_{molecule}.npz", R=arrays["R"], z=arrays["z"], E=arrays["E"], F=arrays["F"])
    return arrays


def _frames(dataset) -> list[int]:
    return [_frame_of(dataset[i]) for i in range(len(dataset))]


def test_load_md22_split_sizes_sum_to_n_with_paper_train_size(tmp_path):
    _write_md22_fixture(tmp_path, "buckyball-catcher", 1000)
    train, val, test = md22.load_md22(str(tmp_path), "buckyball-catcher")
    assert (len(train), len(val), len(test)) == (600, 200, 200)


def test_load_md22_odd_remainder_goes_to_test(tmp_path):
    _write_md22_fixture(tmp_path, "buckyball-catcher", 1001)
    train, val, test = md22.load_md22(str(tmp_path), "buckyball-catcher")
    assert (len(train), len(val), len(test)) == (600, 200, 201)


def test_load_md22_split_is_the_seeded_permutation_and_partitions_all_frames(tmp_path):
    _write_md22_fixture(tmp_path, "buckyball-catcher", 1000)
    train, val, test = md22.load_md22(str(tmp_path), "buckyball-catcher")
    perm = _expected_permutation(1000, seed=0)
    assert _frames(train) == perm[:600]
    assert _frames(val) == perm[600:800]
    assert _frames(test) == perm[800:]
    assert sorted(_frames(train) + _frames(val) + _frames(test)) == list(range(1000))


def test_load_md22_deterministic_and_seed_sensitive(tmp_path):
    _write_md22_fixture(tmp_path, "buckyball-catcher", 1000)
    first = [_frames(d) for d in md22.load_md22(str(tmp_path), "buckyball-catcher")]
    second = [_frames(d) for d in md22.load_md22(str(tmp_path), "buckyball-catcher")]
    other = [_frames(d) for d in md22.load_md22(str(tmp_path), "buckyball-catcher", seed=1)]
    assert first == second
    assert first != other


def test_load_md22_fewer_frames_than_train_size_puts_everything_in_train(tmp_path):
    _write_md22_fixture(tmp_path, "buckyball-catcher", 500)
    train, val, test = md22.load_md22(str(tmp_path), "buckyball-catcher")
    assert (len(train), len(val), len(test)) == (500, 0, 0)


def test_load_md22_train_sizes_match_the_published_per_molecule_values(tmp_path):
    # Chmiela et al., Sci. Adv. 9, eadf0873 (2023), Table 1 training-set sizes.
    published = {
        "Ac-Ala3-NHMe": 6000, "DHA": 8000, "stachyose": 8000, "AT-AT": 3000,
        "AT-AT-CG-CG": 2000, "buckyball-catcher": 600, "double-walled_nanotube": 800,
    }
    assert set(md22.MD22_MOLECULES) == set(published)
    for molecule, expected_train in published.items():
        root = tmp_path / molecule
        n_frames = expected_train + 11
        _write_md22_fixture(root, molecule, n_frames)
        train, val, test = md22.load_md22(str(root), molecule)
        assert len(train) == expected_train
        assert len(train) + len(val) + len(test) == n_frames
        assert (len(val), len(test)) == (5, 6)


def test_load_md22_converts_units_and_flattens_energy_shape(tmp_path):
    arrays = _write_md22_fixture(tmp_path, "buckyball-catcher", 1000)
    train, _, _ = md22.load_md22(str(tmp_path), "buckyball-catcher")
    item = train[0]
    frame = _frame_of(item)
    assert item["energy"].shape == ()
    _close(item["energy"].item(), arrays["E"][frame, 0] * KCAL_PER_MOL_EV, RTOL_F64_ENERGY)
    _close(item["forces"].numpy(), arrays["F"][frame] * KCAL_PER_MOL_EV, RTOL_F32_ONE_CONVERSION)
    assert torch.equal(item["x"], torch.from_numpy(arrays["R"][frame]).float())
    assert item["atomic_numbers"].dtype == torch.long and item["atomic_numbers"].tolist() == [8, 1]
    assert set(item) == {"x", "atomic_numbers", "energy", "forces"}


def test_load_md22_unknown_molecule_raises_value_error(tmp_path):
    with pytest.raises(ValueError, match="unknown MD22 molecule"):
        md22.load_md22(str(tmp_path), "caffeine")


def test_load_md22_missing_archive_raises_not_implemented_with_filename(tmp_path):
    with pytest.raises(NotImplementedError, match="MD22 archive not found") as excinfo:
        md22.load_md22(str(tmp_path), "DHA")
    assert "md22_DHA.npz" in str(excinfo.value)


ANI_ENERGY_KEY = "wb97x_dz.energy"
ANI_FORCES_KEY = "wb97x_dz.forces"


def _energy_hartree(k: int) -> float:
    return -1.0 - 0.125 * k


def _ani_conformers(first_k: int, n_atoms: int, slots: list[str]):
    # slots: "ok" | "nan_energy" | "nan_forces" | "nan_coords"; only "ok" slots consume a sample number k.
    coords, energies, forces = [], [], []
    k = first_k
    for slot in slots:
        sample_k = k if slot == "ok" else -1
        c = sample_k + np.arange(n_atoms * 3, dtype=np.float32).reshape(n_atoms, 3) * 0.25
        e = _energy_hartree(sample_k)
        f = np.full((n_atoms, 3), 2.0 * _energy_hartree(sample_k), dtype=np.float32)
        if slot == "nan_energy":
            e = float("nan")
        elif slot == "nan_forces":
            f[0, 0] = np.nan
        elif slot == "nan_coords":
            c[0, 0] = np.nan
        coords.append(c), energies.append(e), forces.append(f)
        k += slot == "ok"
    return np.stack(coords), np.array(energies, dtype=np.float64), np.stack(forces), k


def _write_ani1x_fixture(path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as fh:
        coords, energies, forces, next_k = _ani_conformers(0, 3, ["ok", "nan_energy", "ok", "nan_forces", "ok", "nan_coords", "ok"])
        group_a = fh.create_group("A")
        group_a.create_dataset("atomic_numbers", data=np.array([8, 1, 1], dtype=np.uint8))
        group_a.create_dataset("coordinates", data=coords)
        group_a.create_dataset(ANI_ENERGY_KEY, data=energies)
        group_a.create_dataset(ANI_FORCES_KEY, data=forces)

        coords, energies, forces, next_k = _ani_conformers(next_k, 2, ["ok"] * 6)
        group_b = fh.create_group("B")
        group_b.create_dataset("atomic_numbers", data=np.array([6, 1], dtype=np.uint8))
        group_b.create_dataset("coordinates", data=coords)
        group_b.create_dataset(ANI_ENERGY_KEY, data=energies)
        group_b.create_dataset(ANI_FORCES_KEY, data=forces)

        # Group C has energies but no forces dataset; group D has no energy dataset at all.
        coords, energies, _, _ = _ani_conformers(100, 2, ["ok", "ok"])
        group_c = fh.create_group("C")
        group_c.create_dataset("atomic_numbers", data=np.array([1, 1], dtype=np.uint8))
        group_c.create_dataset("coordinates", data=coords)
        group_c.create_dataset(ANI_ENERGY_KEY, data=energies)
        group_d = fh.create_group("D")
        group_d.create_dataset("atomic_numbers", data=np.array([1, 1], dtype=np.uint8))
        group_d.create_dataset("coordinates", data=coords)
    assert next_k == 10


def _ani1x_module():
    pytest.importorskip("h5py")
    return importlib.import_module("baselines.datasets.ani1x")


def _sample_k(item: dict) -> int:
    return int(round(float(item["x"][0, 0])))


def test_load_ani1x_skips_nonfinite_and_incomplete_groups_and_sums_to_ten(tmp_path):
    ani1x = _ani1x_module()
    _write_ani1x_fixture(tmp_path / "ani.h5")
    train, val, test = ani1x.load_ani1x(str(tmp_path / "ani.h5"))
    # 4 valid in A + 6 in B = 10 conformers. The split is by formula group, and with 2 groups round(2*0.8) = 2 train groups, round(2*0.1) = 0 val, 0 test, so both groups (all 10 conformers) go to train.
    assert (len(train), len(val), len(test)) == (10, 0, 0)


ANI_GROUP_COUNT = 10
ANI_CONFORMERS_PER_GROUP = 3


def _write_ani1x_many_group_fixture(path: Path) -> None:
    # Group g holds conformers with sample numbers 3g..3g+2, so _sample_k(item) // 3 recovers the group.
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as fh:
        for g in range(ANI_GROUP_COUNT):
            coords, energies, forces, _ = _ani_conformers(g * ANI_CONFORMERS_PER_GROUP, 2, ["ok"] * ANI_CONFORMERS_PER_GROUP)
            grp = fh.create_group(f"G{g}")
            grp.create_dataset("atomic_numbers", data=np.array([1, 1], dtype=np.uint8))
            grp.create_dataset("coordinates", data=coords)
            grp.create_dataset(ANI_ENERGY_KEY, data=energies)
            grp.create_dataset(ANI_FORCES_KEY, data=forces)


def _groups_per_split(splits) -> list[list[int]]:
    return [[_sample_k(d[i]) // ANI_CONFORMERS_PER_GROUP for i in range(len(d))] for d in splits]


def test_load_ani1x_split_keeps_every_group_in_exactly_one_split(tmp_path):
    ani1x = _ani1x_module()
    _write_ani1x_many_group_fixture(tmp_path / "ani.h5")
    groups = _groups_per_split(ani1x.load_ani1x(str(tmp_path / "ani.h5")))
    sets = [set(g) for g in groups]
    # No formula group may appear in two splits (conformers of one molecule must never straddle train/test).
    assert sets[0] & sets[1] == set() and sets[0] & sets[2] == set() and sets[1] & sets[2] == set()
    assert sets[0] | sets[1] | sets[2] == set(range(ANI_GROUP_COUNT))
    # Every group is present with all of its conformers, not a subset.
    for g in range(ANI_GROUP_COUNT):
        assert sum(g_list.count(g) for g_list in groups) == ANI_CONFORMERS_PER_GROUP


def test_load_ani1x_group_split_matches_seeded_shuffle_of_sorted_group_names(tmp_path):
    ani1x = _ani1x_module()
    _write_ani1x_many_group_fixture(tmp_path / "ani.h5")
    train, val, test = ani1x.load_ani1x(str(tmp_path / "ani.h5"))
    # order = G7 G8 G1 G5 G3 G4 G2 G0 | G9 | G6 -> round(10*0.8) = 8 train, round(10*0.1) = 1 val, 1 test.
    train_groups, val_groups, test_groups = (set(g) for g in _groups_per_split((train, val, test)))
    assert train_groups == {7, 8, 1, 5, 3, 4, 2, 0}
    assert val_groups == {9}
    assert test_groups == {6}
    assert (len(train), len(val), len(test)) == (8 * ANI_CONFORMERS_PER_GROUP, ANI_CONFORMERS_PER_GROUP, ANI_CONFORMERS_PER_GROUP)


def test_load_ani1x_deterministic_and_seed_sensitive(tmp_path):
    ani1x = _ani1x_module()
    _write_ani1x_many_group_fixture(tmp_path / "ani.h5")
    path = str(tmp_path / "ani.h5")
    first = _groups_per_split(ani1x.load_ani1x(path))
    second = _groups_per_split(ani1x.load_ani1x(path))
    other = _groups_per_split(ani1x.load_ani1x(path, seed=1))
    assert first == second
    assert {g for g in other[1]} == {1} and {g for g in other[2]} == {2}  # seed 1: val group G1, test group G2 (hand-derived)
    assert first != other


def test_load_ani1x_energy_is_hartree_to_ev_and_forces_are_hartree_per_angstrom_to_ev_per_angstrom(tmp_path):
    ani1x = _ani1x_module()
    _write_ani1x_fixture(tmp_path / "ani.h5")
    datasets = ani1x.load_ani1x(str(tmp_path / "ani.h5"))
    checked = 0
    for dataset in datasets:
        for i in range(len(dataset)):
            item = dataset[i]
            k = _sample_k(item)
            # ANI forces are Hartree/Angstrom, so only the energy factor applies --> no Bohr factor can apply.
            _close(item["energy"].item(), _energy_hartree(k) * HARTREE_EV, RTOL_F64_ENERGY)
            _close(item["forces"].numpy(), np.full(item["forces"].shape, 2.0 * _energy_hartree(k) * HARTREE_EV), RTOL_F32_TWO_STEP)
            assert item["energy"].dtype == torch.float64 and item["forces"].dtype == torch.float32
            assert item["energy"].shape == ()
            checked += 1
    assert checked == 10


def test_load_ani1x_positions_and_atomic_numbers_pass_through(tmp_path):
    ani1x = _ani1x_module()
    _write_ani1x_fixture(tmp_path / "ani.h5")
    for dataset in ani1x.load_ani1x(str(tmp_path / "ani.h5")):
        for i in range(len(dataset)):
            item = dataset[i]
            k = _sample_k(item)
            n_atoms = item["atomic_numbers"].shape[0]
            assert item["atomic_numbers"].dtype == torch.long
            assert item["atomic_numbers"].tolist() == ([8, 1, 1] if k < 4 else [6, 1])
            expected = k + np.arange(n_atoms * 3, dtype=np.float32).reshape(n_atoms, 3) * 0.25
            assert torch.equal(item["x"], torch.from_numpy(expected))
            assert item["x"].dtype == torch.float32


def test_load_ani1x_without_forces_key_yields_energy_only_samples(tmp_path):
    ani1x = _ani1x_module()
    _write_ani1x_fixture(tmp_path / "ani.h5")
    train, val, test = ani1x.load_ani1x(str(tmp_path / "ani.h5"), forces_key=None)
    # NaN forces no longer disqualify a conformer: A keeps 5 (only NaN energy / NaN coords drop), B has 6, C has 2 = 13.
    assert len(train) + len(val) + len(test) == 13
    # 3 groups (A:5, B:6, C:2 conformers): round(3*0.8) = 2 train groups, round(3*0.1) = 0 val, 1 test. Seed 0 shuffles the sorted names A,B,C to A,C,B, so train = A + C = 7 and test = B = 6.
    assert (len(train), len(val), len(test)) == (7, 0, 6)
    assert all("forces" not in d[i] for d in (train, val, test) for i in range(len(d)))


def test_load_ani1x_energy_key_absent_from_every_group_gives_empty_splits(tmp_path):
    ani1x = _ani1x_module()
    _write_ani1x_fixture(tmp_path / "ani.h5")
    splits = ani1x.load_ani1x(str(tmp_path / "ani.h5"), energy_key="ccsd(t)_cbs.energy", forces_key=None)
    assert [len(d) for d in splits] == [0, 0, 0]


def test_load_ani1x_missing_file_raises_file_not_found(tmp_path):
    ani1x = _ani1x_module()
    with pytest.raises(FileNotFoundError, match="not found"):
        ani1x.load_ani1x(str(tmp_path / "missing.h5"))


def test_ani1x_default_keys_and_fractions():
    ani1x = _ani1x_module()
    assert ani1x.ANI1X_DEFAULT_ENERGY_KEY == "wb97x_dz.energy"
    assert ani1x.ANI1X_DEFAULT_FORCES_KEY == "wb97x_dz.forces"
    assert ani1x.ANI1X_TRAIN_FRAC == 0.8 and ani1x.ANI1X_VAL_FRAC == 0.1


def test_units_check_reexports_the_config_units_objects():
    from baselines.config import units
    from baselines.datasets import units_check

    assert units_check.to_ev_angstrom is units.to_ev_angstrom
    assert units_check.HARTREE_TO_EV == units.HARTREE_TO_EV
    assert units_check.BOHR_TO_ANGSTROM == units.BOHR_TO_ANGSTROM
    assert units_check.KCAL_PER_MOL_TO_EV == units.KCAL_PER_MOL_TO_EV
    assert set(units_check.__all__) == {"to_ev_angstrom", "Number", "HARTREE_TO_EV", "BOHR_TO_ANGSTROM", "KCAL_PER_MOL_TO_EV"}


def test_units_check_constants_pass_its_own_sanity_bounds():
    from baselines.datasets import units_check

    assert 27.0 < units_check.HARTREE_TO_EV < 27.3
    assert 0.52 < units_check.BOHR_TO_ANGSTROM < 0.53
    assert 0.04 < units_check.KCAL_PER_MOL_TO_EV < 0.05
    assert units_check.to_ev_angstrom(1.0, 1.0, "ev_angstrom") == (1.0, 1.0)


@pytest.mark.parametrize("constant, bad_value", [
    ("HARTREE_TO_EV", 28.0), ("HARTREE_TO_EV", 26.0), ("BOHR_TO_ANGSTROM", 0.6), ("BOHR_TO_ANGSTROM", 1.0),
    ("KCAL_PER_MOL_TO_EV", 0.06), ("KCAL_PER_MOL_TO_EV", 4.184),
])
def test_units_check_import_time_assertion_rejects_a_corrupted_constant(monkeypatch, constant, bad_value):
    from baselines.config import units
    from baselines.datasets import units_check

    monkeypatch.setattr(units, constant, bad_value)
    try:
        with pytest.raises(AssertionError, match=constant):
            importlib.reload(units_check)
    finally:
        monkeypatch.undo()
        importlib.reload(units_check)
    assert getattr(units_check, constant) == getattr(units, constant)
