from __future__ import annotations

# Classical force-field baselines (GAFF2 and OpenFF "Sage") run through OpenMM. Vacuum, no cutoff, flexible bonds. 

import hashlib
import json
import os
from collections.abc import Sequence

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from baselines.config.units import EV_TO_KJ_PER_MOL

FORCEFIELDS = {"gaff2": "gaff-2.11", "openff": "openff-2.1.0"}
NM_PER_ANGSTROM = 0.1
# GAFF2 is parameterized for these organic elements.
SUPPORTED_ATOMIC_NUMBERS = frozenset({1, 6, 7, 8, 9, 15, 16, 17, 35, 53})
MAX_GEOMETRY_ATOMS = 200


class TopologyError(RuntimeError):
    # The molecular graph could not be obtained for this molecule.
    pass


class ParameterizationError(RuntimeError):
    # The graph is known but the force field has no parameters / charges for it.
    pass


def _rdkit():
    from rdkit import Chem

    return Chem


def bonds_from_geometry(atomic_numbers: Sequence[int], positions: np.ndarray) -> set[tuple[int, int]]:
    # Connectivity only (covalent-radius rule); used both to build the graph and to check later frames still have the same one.
    from rdkit.Chem import rdDetermineBonds

    mol = _xyz_mol(atomic_numbers, positions)
    rdDetermineBonds.DetermineConnectivity(mol)
    return {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in mol.GetBonds()}


def _xyz_mol(atomic_numbers, positions):
    Chem = _rdkit()
    block = "\n".join([str(len(atomic_numbers)), ""] + [
        f"{Chem.GetPeriodicTable().GetElementSymbol(int(z))} {x:.8f} {y:.8f} {w:.8f}" for z, (x, y, w) in zip(atomic_numbers, positions)])
    mol = Chem.MolFromXYZBlock(block)
    if mol is None:
        raise TopologyError("RDKit could not read the coordinates")
    return mol


def rdmol_from_geometry(atomic_numbers: Sequence[int], positions: np.ndarray, charge: int = 0):
    # Bond orders from geometry; for neutral molecules whose frames carry no graph (rMD17, MD22). Atom order is the input order.
    from rdkit.Chem import rdDetermineBonds

    if len(atomic_numbers) > MAX_GEOMETRY_ATOMS:
        raise TopologyError(f"{len(atomic_numbers)} atoms exceeds the {MAX_GEOMETRY_ATOMS}-atom limit for geometry-based bond perception")
    mol = _xyz_mol(atomic_numbers, positions)
    try:
        rdDetermineBonds.DetermineBonds(mol, charge=charge)
    except Exception as exc:  # RDKit raises ValueError / RuntimeError subclasses depending on the failure
        raise TopologyError(f"bond perception failed: {exc}") from exc
    return mol


def rdmol_from_mapped_smiles(smiles: str, n_atoms: int):
    # SPICE SMILES carry atom-map numbers 1..n equal to (dataset atom index + 1); return a molecule whose atom i is dataset atom i.
    Chem = _rdkit()
    params = Chem.SmilesParserParams()
    params.removeHs = False  # the default strips the mapped [H:n] atoms, which would renumber everything after them
    mol = Chem.MolFromSmiles(smiles, params)
    if mol is None:
        raise TopologyError(f"unparseable SMILES {smiles!r}")
    maps = [a.GetAtomMapNum() for a in mol.GetAtoms()]
    if sorted(maps) != list(range(1, n_atoms + 1)):
        raise TopologyError(f"atom maps of {smiles!r} are not 1..{n_atoms}")
    order = [m - 1 for m in maps]  # order[old_index] = new_index
    mol = Chem.RenumberAtoms(mol, [order.index(i) for i in range(n_atoms)])
    for a in mol.GetAtoms():
        a.SetAtomMapNum(0)
    return mol


def _fragment_mol(rdmol, idx: list[int]):
    # The sub-molecule on atoms idx (their order kept), with RDKit's own bond orders and formal charges.
    Chem = _rdkit()
    rw = Chem.RWMol(rdmol)
    keep = set(idx)
    for i in sorted((a.GetIdx() for a in rdmol.GetAtoms() if a.GetIdx() not in keep), reverse=True):
        rw.RemoveAtom(i)
    return rw.GetMol()


def to_openff(rdmol, positions: np.ndarray):
    # positions in Angstrom; the conformer is what AM1-BCC is computed on.
    from openff.toolkit import Molecule
    Chem = _rdkit()
    mol = Chem.Mol(rdmol)
    mol.RemoveAllConformers()
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, p in enumerate(np.asarray(positions, dtype=np.float64)):
        conf.SetAtomPosition(i, p.tolist())
    mol.AddConformer(conf, assignId=True)
    Chem.AssignStereochemistryFrom3D(mol)
    try:
        off = Molecule.from_rdkit(mol, allow_undefined_stereo=True, hydrogens_are_explicit=True)
    except Exception as exc:
        raise TopologyError(f"OpenFF could not read the molecule: {exc}") from exc
    return off


def molecule_key(off_mol) -> str:
    return hashlib.sha256(off_mol.to_smiles(mapped=True).encode()).hexdigest()[:20]


def assign_charges(off_mol, cache_dir: str | None = None) -> bool:
    # AM1-BCC via AmberTools on the molecule's single stored conformer, cached on disk by molecular graph. Returns True on a cache hit.
    from openff.units import unit

    path = os.path.join(cache_dir, f"{molecule_key(off_mol)}.json") if cache_dir else None
    if path and os.path.exists(path):
        with open(path) as f:
            charges = json.load(f)
        off_mol.partial_charges = np.asarray(charges, dtype=np.float64) * unit.elementary_charge
        return True
    try:
        off_mol.assign_partial_charges("am1bcc")
    except Exception as exc:
        raise ParameterizationError(f"AM1-BCC failed: {str(exc).splitlines()[0] if str(exc) else type(exc).__name__}") from exc
    if path:
        os.makedirs(cache_dir, exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump([float(q) for q in off_mol.partial_charges.m_as(unit.elementary_charge)], f)
        os.replace(tmp, path)  # atomic, so parallel jobs never read a half-written file
    return False


def split_fragments(rdmol) -> list[list[int]]:
    # Covalently bonded components of the graph, as lists of atom indices (ordered by first atom). Non-covalent partners (the two strands of a
    # base pair, a host and its guest) are separate fragments: they interact only through the force field's nonbonded terms.
    Chem = _rdkit()
    return [list(f) for f in Chem.GetMolFrags(rdmol)]


def build_openmm_system(off_mols, forcefield: str):
    # off_mols: one OpenFF molecule or a list of them; the system's particles are the molecules' atoms concatenated in list order.
    from openff.toolkit import Topology
    from openmmforcefields.generators import SystemGenerator

    if forcefield not in FORCEFIELDS:
        raise ValueError(f"forcefield must be one of {sorted(FORCEFIELDS)}, got {forcefield!r}")
    off_mols = [off_mols] if not isinstance(off_mols, (list, tuple)) else list(off_mols)
    unique = []  # the generator needs each distinct molecule once; it matches repeated copies in the topology by graph
    for m in off_mols:
        if not any(m.is_isomorphic_with(u) for u in unique):
            unique.append(m)
    generator = SystemGenerator(
        small_molecule_forcefield=FORCEFIELDS[forcefield], molecules=unique, cache=None,
        # Flexible bonds and no center-of-mass removal: forces on every atom must be the plain gradient of the energy.
        forcefield_kwargs={"constraints": None, "rigidWater": False, "removeCMMotion": False},
    )
    try:
        return generator.create_system(Topology.from_molecules(off_mols).to_openmm())
    except Exception as exc:
        raise ParameterizationError(f"{forcefield} has no parameters: {str(exc).splitlines()[0]}") from exc


class OpenMMCalculator(Calculator):
    # Energy in eV and forces in eV/Angstrom for a fixed OpenMM system (positions are the only input).
    implemented_properties = ["energy", "forces"]

    def __init__(self, system, platform: str = "Reference", order: Sequence[int] | None = None, **kwargs):
        # order[k] = index in the ASE frame of the system's k-th particle (None: the same order).
        super().__init__(**kwargs)
        self.order = None if order is None else np.asarray(order, dtype=int)
        from openmm import Context, Platform, VerletIntegrator

        # "Reference" is double precision; the single-precision CPU platform would add ~1e-4 relative noise to energies.
        self._context = Context(system, VerletIntegrator(1e-3), Platform.getPlatformByName(platform))
        self.n_atoms = system.getNumParticles()

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        if len(atoms) != self.n_atoms:
            raise ValueError(f"system has {self.n_atoms} atoms, frame has {len(atoms)}")
        positions = atoms.get_positions()
        self._context.setPositions((positions if self.order is None else positions[self.order]) * NM_PER_ANGSTROM)
        state = self._context.getState(getEnergy=True, getForces=True)
        from openmm import unit

        energy_kj = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        forces_kj_nm = np.asarray(state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer))
        forces = forces_kj_nm / EV_TO_KJ_PER_MOL * NM_PER_ANGSTROM  # kJ/mol/nm -> eV/Angstrom: divide by the energy factor, then by 10 Angstrom per nm.
        if self.order is not None:
            back = np.empty_like(forces)
            back[self.order] = forces
            forces = back
        self.results = {"energy": energy_kj / EV_TO_KJ_PER_MOL, "forces": forces}


class ClassicalCalculator(Calculator):
    # One OpenMM system per molecule --> A frame's molecule is identified by its `smiles` or by its element sequence.
    implemented_properties = ["energy", "forces"]

    def __init__(self, forcefield: str = "gaff2", charge_cache: str | None = None, platform: str = "Reference", **kwargs):
        super().__init__(**kwargs)
        if forcefield not in FORCEFIELDS:
            raise ValueError(f"forcefield must be one of {sorted(FORCEFIELDS)}, got {forcefield!r}")
        self.forcefield, self.charge_cache, self.platform = forcefield, charge_cache, platform
        self._calcs: dict[str, OpenMMCalculator] = {}
        self._failed: dict[str, str] = {}
        self.charge_cache_hits = 0
        self.graph_bonds: dict[str, set[tuple[int, int]]] = {}

    @staticmethod
    def key_for(atoms) -> str:
        smiles = atoms.info.get("smiles")
        if smiles is not None:
            return f"smiles:{smiles}"
        return "geometry:" + ",".join(str(int(z)) for z in atoms.get_atomic_numbers())

    def _build(self, key: str, atoms) -> OpenMMCalculator:
        numbers, positions = atoms.get_atomic_numbers(), atoms.get_positions()
        unsupported = sorted({int(z) for z in numbers} - SUPPORTED_ATOMIC_NUMBERS)
        if unsupported:
            raise ParameterizationError(f"elements {unsupported} are outside the GAFF2 organic set")
        if key.startswith("smiles:"):
            rdmol = rdmol_from_mapped_smiles(key[len("smiles:"):], len(atoms))
        else:
            rdmol = rdmol_from_geometry(numbers, positions)
        self.graph_bonds[key] = {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in rdmol.GetBonds()}
        if [a.GetAtomicNum() for a in rdmol.GetAtoms()] != [int(z) for z in numbers]:
            raise TopologyError("graph atom order does not match the frame")
        # AM1-BCC is defined for one molecule: charge each covalent fragment on its own conformer, then build one system from all fragments.
        fragments = split_fragments(rdmol)
        offs = []
        for idx in fragments:
            sub = _fragment_mol(rdmol, idx)
            off = to_openff(sub, positions[idx])
            try:
                self.charge_cache_hits += int(assign_charges(off, self.charge_cache))
            except ParameterizationError as exc:  # say which fragment, so a failure on a multi-fragment system is attributable
                raise ParameterizationError(f"fragment {len(offs) + 1} of {len(fragments)} ({off.hill_formula}, {len(idx)} atoms, net charge "
                                            f"{off.total_charge.m:+.0f}): {exc}") from exc
            offs.append(off)
        order = [i for idx in fragments for i in idx]
        return OpenMMCalculator(build_openmm_system(offs, self.forcefield), self.platform, order=None if order == sorted(order) else order)

    def prepare(self, atoms) -> str | None:
        # Builds (or recalls) the system for this frame's molecule. Returns None on success, else the reason it cannot be parameterized.
        key = self.key_for(atoms)
        if key in self._failed:
            return self._failed[key]
        if key not in self._calcs:
            try:
                self._calcs[key] = self._build(key, atoms)
            except (TopologyError, ParameterizationError) as exc:
                self._failed[key] = str(exc)
                return self._failed[key]
        return None

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        reason = self.prepare(atoms)
        if reason is not None:
            raise ParameterizationError(reason)
        inner = self._calcs[self.key_for(atoms)]
        inner.reset()
        inner.calculate(atoms, ["energy", "forces"], all_changes)
        self.results = dict(inner.results)

    @property
    def failures(self) -> dict[str, str]:
        return dict(self._failed)
