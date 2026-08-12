from __future__ import annotations

import glob
import math
import os
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from rdkit import Chem

from utils.mappings import ATOM_TYPE_MAP, atom_types_to_masses
from utils.calc_props import calc_rdkit_mol_props_smiles, _pad_dim1
from utils.pw_bases import compute_molecular_box

BOHR_TO_ANGSTROM = 0.529177210903
HARTREE_TO_EV = 27.211386245988
FORCE_AU_TO_EV_PER_ANGSTROM = HARTREE_TO_EV / BOHR_TO_ANGSTROM

SPICE_SUBSET_GROUPS: Dict[str, List[str]] = {
    "dipeptides": ["SPICE Dipeptides Single Points Dataset v1.3"],
    "solvated_amino_acids": ["SPICE Solvated Amino Acids Single Points Dataset v1.1"],
    "des_monomers": ["SPICE DES Monomers Single Points Dataset v1.1"],
    "des370k": [
        "SPICE DES370K Single Points Dataset v1.0",
        "SPICE DES370K Single Points Dataset Supplement v1.1",
    ],
    "pubchem": (
        [f"SPICE PubChem Set {i} Single Points Dataset v1.3" for i in range(1, 7)]
        + [f"SPICE PubChem Set {i} Single Points Dataset v1.0" for i in range(7, 11)]
    ),
    "pubchem_boron_silicon": ["SPICE PubChem Boron Silicon v1.0"],
    "solvated_pubchem": ["SPICE Solvated PubChem Set 1 v1.0"],
    "ion_pairs": ["SPICE Ion Pairs Single Points Dataset v1.2"],
    "water_clusters": ["SPICE Water Clusters v1.0"],
    "amino_acid_ligand": ["SPICE Amino Acid Ligand v1.0"],
}

_PERIODIC_TABLE = Chem.GetPeriodicTable()

_REQUIRED_DATASETS = (
    "conformations", "formation_energy", "dft_total_gradient", "atomic_numbers", "smiles", "mbis_charges",
)


def _resolve_subsets(subsets: Optional[List[str]]) -> Optional[set]:
    if subsets is None:
        return None
    resolved: set = set()
    for s in subsets:
        resolved.update(SPICE_SUBSET_GROUPS.get(s, [s]))
    return resolved


def _expand_paths(hdf5_paths: Union[str, List[str]]) -> List[str]:
    if isinstance(hdf5_paths, str):
        hdf5_paths = [hdf5_paths]
    paths: List[str] = []
    for p in hdf5_paths:
        if os.path.isdir(p):
            paths.extend(sorted(glob.glob(os.path.join(p, "*.hdf5"))))
        else:
            paths.extend(sorted(glob.glob(p)))
    if not paths:
        raise FileNotFoundError(f"No .hdf5 files found for hdf5_paths={hdf5_paths!r}")
    return paths


def _h5_str(dataset) -> str:
    val = dataset[0]
    return val.decode() if isinstance(val, bytes) else str(val)


def _parse_atom_map(smiles: str) -> Optional[Tuple["Chem.Mol", Dict[int, int], bool]]:
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None
    try:
        mol.UpdatePropertyCache(strict=False)
        Chem.GetSymmSSSR(mol)
    except Exception:
        return None

    map_to_array_idx = {
        atom.GetIdx(): atom.GetAtomMapNum() - 1
        for atom in mol.GetAtoms() if atom.GetAtomMapNum() > 0
    }
    reliable = len(map_to_array_idx) == mol.GetNumAtoms()
    if not reliable:
        map_to_array_idx = {i: i for i in range(mol.GetNumAtoms())}

    return mol, map_to_array_idx, reliable


def _split_solute_solvent(smiles: str, n_atoms: int) -> torch.Tensor:
    is_solvent = torch.zeros(n_atoms, dtype=torch.bool)

    parsed = _parse_atom_map(smiles)
    if parsed is None:
        return is_solvent
    mol, map_to_array_idx, _ = parsed

    try:
        frags = Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)
    except Exception:
        return is_solvent

    for frag_atom_idx in frags:
        if len(frag_atom_idx) != 3:
            continue
        symbols = sorted(mol.GetAtomWithIdx(i).GetSymbol() for i in frag_atom_idx)
        if symbols != ["H", "H", "O"]:
            continue
        for i in frag_atom_idx:
            arr_idx = map_to_array_idx.get(i)
            if arr_idx is not None and arr_idx < n_atoms:
                is_solvent[arr_idx] = True

    return is_solvent


_BOND_TYPES: Tuple["Chem.BondType", ...] = (
    Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC,
)
_NO_BOND_IDX     = len(_BOND_TYPES)      # 4
_SAME_RING_IDX   = _NO_BOND_IDX + 1      # 5
_TOPO_DIST_IDX   = _SAME_RING_IDX + 1    # 6
_UNRELIABLE_IDX  = _TOPO_DIST_IDX + 1    # 7
_EDGE_FEAT_DIM   = _UNRELIABLE_IDX + 1   # 8
_TOPO_DIST_CAP   = 8.0


def build_rdkit_edge_features(smiles: str, n_atoms: int) -> torch.Tensor:
    feats = torch.zeros(n_atoms, n_atoms, _EDGE_FEAT_DIM)

    parsed = _parse_atom_map(smiles)
    if parsed is None:
        feats[..., _UNRELIABLE_IDX] = 1.0
        return feats
    mol, map_to_array_idx, reliable = parsed
    if not reliable:
        feats[..., _UNRELIABLE_IDX] = 1.0

    try:
        dist_matrix = Chem.GetDistanceMatrix(mol)
    except Exception:
        feats[..., _UNRELIABLE_IDX] = 1.0
        return feats

    ring_membership: Dict[int, set] = {}
    for ring_id, ring in enumerate(mol.GetRingInfo().AtomRings()):
        for a in ring:
            ring_membership.setdefault(a, set()).add(ring_id)

    array_to_mol_idx = {v: k for k, v in map_to_array_idx.items()}
    n_mol_atoms = mol.GetNumAtoms()

    for arr_i in range(n_atoms):
        mol_i = array_to_mol_idx.get(arr_i)
        if mol_i is None or mol_i >= n_mol_atoms:
            continue
        for arr_j in range(n_atoms):
            if arr_i == arr_j:
                continue
            mol_j = array_to_mol_idx.get(arr_j)
            if mol_j is None or mol_j >= n_mol_atoms:
                continue

            bond = mol.GetBondBetweenAtoms(mol_i, mol_j)
            if bond is None:
                feats[arr_i, arr_j, _NO_BOND_IDX] = 1.0
            else:
                bt = bond.GetBondType()
                for b_idx, cand in enumerate(_BOND_TYPES):
                    if bt == cand:
                        feats[arr_i, arr_j, b_idx] = 1.0
                        break

            if ring_membership.get(mol_i, set()) & ring_membership.get(mol_j, set()):
                feats[arr_i, arr_j, _SAME_RING_IDX] = 1.0

            d = float(dist_matrix[mol_i, mol_j])
            if math.isfinite(d):
                feats[arr_i, arr_j, _TOPO_DIST_IDX] = min(max(d, 0.0), _TOPO_DIST_CAP) / _TOPO_DIST_CAP

    return feats


class SPICEDataset(Dataset):
    def __init__(
        self,
        hdf5_paths: Union[str, List[str]],
        subsets: Optional[List[str]] = None,
        exclude_solvated: bool = False,
        box_padding: float = 12.0,
        edge_cutoff: float = 5.0,
    ):
        self._paths = _expand_paths(hdf5_paths)
        allowed = _resolve_subsets(subsets)
        self._box_padding = box_padding
        self._edge_cutoff = edge_cutoff

        self._index: List[Tuple[str, str, int]] = []
        self._energies: List[float] = []
        self._force_sumsq: List[float] = []
        self._n_atoms: List[int] = []
        self._mol_props_cache: Dict[str, torch.Tensor] = {}
        self._split_cache: Dict[str, torch.Tensor] = {}
        self._edge_feat_cache: Dict[str, torch.Tensor] = {}
        self._handles: Dict[str, h5py.File] = {}

        for path in self._paths:
            with h5py.File(path, "r") as fh:
                for group_name in fh.keys():
                    grp = fh[group_name]
                    if any(key not in grp for key in _REQUIRED_DATASETS):
                        continue
                    if allowed is not None and _h5_str(grp["subset"]) not in allowed:
                        continue

                    conf_net_charge = np.asarray(grp["mbis_charges"])[..., 0].sum(axis=1)
                    valid_charge = np.isfinite(conf_net_charge)
                    if not valid_charge.any():
                        continue
                    molecule_charge = round(float(np.median(conf_net_charge[valid_charge])))
                    if molecule_charge != 0:
                        continue

                    if exclude_solvated:
                        smiles = _h5_str(grp["smiles"])
                        n_atoms = grp["atomic_numbers"].shape[0]
                        if self._solvent_mask(smiles, n_atoms).any():
                            continue

                    conformations = np.asarray(grp["conformations"])
                    energies = np.asarray(grp["formation_energy"])
                    gradients = np.asarray(grp["dft_total_gradient"])
                    n_conf = conformations.shape[0]
                    n_atoms_group = conformations.shape[1]
                    for conf_idx in range(n_conf):
                        if not (
                            valid_charge[conf_idx]
                            and np.isfinite(conformations[conf_idx]).all()
                            and np.isfinite(energies[conf_idx])
                            and np.isfinite(gradients[conf_idx]).all()
                        ):
                            continue
                        self._index.append((path, group_name, conf_idx))
                        self._energies.append(
                            float(energies[conf_idx]) * HARTREE_TO_EV
                        )
                        self._force_sumsq.append(
                            float(np.square(gradients[conf_idx] * FORCE_AU_TO_EV_PER_ANGSTROM).sum())
                        )
                        self._n_atoms.append(n_atoms_group)

    def __len__(self) -> int:
        return len(self._index)

    def energy_force_stats(self, indices: Optional[List[int]] = None) -> Tuple[float, float, float]:
        idx = range(len(self._index)) if indices is None else indices
        energies = np.array([self._energies[i] for i in idx])
        force_sumsq = np.array([self._force_sumsq[i] for i in idx])
        n_atoms = np.array([self._n_atoms[i] for i in idx])
        energy_mean = float(energies.mean())
        energy_std = float(energies.std())
        force_rms = float(np.sqrt((force_sumsq / (n_atoms * 3)).mean()))
        return energy_mean, energy_std, force_rms

    def identity_for(self, idx: int) -> Tuple[str, str, int]:
        path, group_name, conf_idx = self._index[idx]
        return (os.path.basename(path), group_name, conf_idx)

    def indices_for_identities(self, identities) -> List[int]:
        wanted = set(tuple(i) for i in identities)
        return [
            i for i, (path, group_name, conf_idx) in enumerate(self._index)
            if (os.path.basename(path), group_name, conf_idx) in wanted
        ]

    def _handle(self, path: str) -> h5py.File:
        fh = self._handles.get(path)
        if fh is None or not fh.id.valid:
            fh = h5py.File(path, "r", swmr=True)
            self._handles[path] = fh
        return fh

    def _mol_props(self, smiles: str) -> torch.Tensor:
        cached = self._mol_props_cache.get(smiles)
        if cached is None:
            cached = torch.tensor(calc_rdkit_mol_props_smiles(smiles), dtype=torch.float32)
            self._mol_props_cache[smiles] = cached
        return cached

    def _solvent_mask(self, smiles: str, n_atoms: int) -> torch.Tensor:
        cached = self._split_cache.get(smiles)
        if cached is None:
            cached = _split_solute_solvent(smiles, n_atoms)
            self._split_cache[smiles] = cached
        return cached

    def _edge_feats(self, smiles: str, n_atoms: int) -> torch.Tensor:
        cached = self._edge_feat_cache.get(smiles)
        if cached is None:
            cached = build_rdkit_edge_features(smiles, n_atoms)
            self._edge_feat_cache[smiles] = cached
        return cached

    def __getitem__(self, idx: int) -> Dict[str, object]:
        path, group_name, conf_idx = self._index[idx]
        grp = self._handle(path)[group_name]

        atomic_numbers = np.asarray(grp["atomic_numbers"])
        n_atoms = atomic_numbers.shape[0]

        smiles = _h5_str(grp["smiles"])
        subset = _h5_str(grp["subset"])

        types = torch.tensor(
            [ATOM_TYPE_MAP.get(_PERIODIC_TABLE.GetElementSymbol(int(z)), 0) for z in atomic_numbers],
            dtype=torch.long,
        )

        x = torch.from_numpy(np.asarray(grp["conformations"][conf_idx])).float() * BOHR_TO_ANGSTROM
        energy = float(grp["formation_energy"][conf_idx]) * HARTREE_TO_EV
        gradient = torch.from_numpy(np.asarray(grp["dft_total_gradient"][conf_idx])).float()
        forces = -gradient * FORCE_AU_TO_EV_PER_ANGSTROM

        V = compute_molecular_box(x, self._box_padding, self._box_padding, self._edge_cutoff)

        return {
            "idx": idx,
            "x": x,
            "types": types,
            "is_solvent": self._solvent_mask(smiles, n_atoms),
            "energy": energy,
            "forces": forces,
            "mol_props": self._mol_props(smiles),
            "edge_feats": self._edge_feats(smiles, n_atoms),
            "smiles": smiles,
            "subset": subset,
            "V": V,
        }


def _pad1(t: torch.Tensor, n_max: int) -> torch.Tensor:
    return _pad_dim1(t.unsqueeze(0), n_max).squeeze(0)


def _mask_for(n_real: int, n_max: int) -> torch.Tensor:
    return (torch.arange(n_max) < n_real).float()


def _pad_edge_feats(t: torch.Tensor, n_max: int) -> torch.Tensor:
    n = t.shape[0]
    if n == n_max:
        return t
    out = torch.zeros(n_max, n_max, t.shape[-1], dtype=t.dtype)
    out[:n, :n] = t
    return out


def spice_collate_fn(batch: List[Dict[str, object]]) -> Dict[str, object]:
    B = len(batch)

    solute_x, solute_types, solute_forces, solute_n = [], [], [], []
    solvent_x, solvent_types, solvent_forces, solvent_n = [], [], [], []
    solute_edge_feats = []
    energies, mol_props, smiles_list, subset_list, box_list, idx_list = [], [], [], [], [], []

    for item in batch:
        is_solvent = item["is_solvent"]
        is_solute = ~is_solvent

        idx_list.append(item["idx"])

        solute_x.append(item["x"][is_solute])
        solute_types.append(item["types"][is_solute])
        solute_forces.append(item["forces"][is_solute])
        solute_n.append(int(is_solute.sum().item()))

        solvent_x.append(item["x"][is_solvent])
        solvent_types.append(item["types"][is_solvent])
        solvent_forces.append(item["forces"][is_solvent])
        solvent_n.append(int(is_solvent.sum().item()))

        solute_edge_feats.append(item["edge_feats"][is_solute][:, is_solute])

        energies.append(item["energy"])
        mol_props.append(item["mol_props"])
        smiles_list.append(item["smiles"])
        subset_list.append(item["subset"])
        box_list.append(item["V"])

    N1_max = max(max(solute_n), 1)
    N2_max = max(max(solvent_n), 1)

    all_solute_types = torch.stack([_pad1(t, N1_max) for t in solute_types], dim=0)
    all_solute_mask = torch.stack([_mask_for(n, N1_max) for n in solute_n], dim=0)

    all_solvent_types = torch.stack([_pad1(t, N2_max) for t in solvent_types], dim=0)
    all_solvent_mask = torch.stack([_mask_for(n, N2_max) for n in solvent_n], dim=0)

    V = torch.stack(box_list, dim=0)

    return {
        "all_solute_x": torch.stack([_pad1(t, N1_max) for t in solute_x], dim=0),
        "all_solute_types": all_solute_types,
        "all_solute_masses": atom_types_to_masses(all_solute_types) * all_solute_mask,
        "all_solute_mask": all_solute_mask,
        "all_solute_edge_feats": torch.stack([_pad_edge_feats(t, N1_max) for t in solute_edge_feats], dim=0),
        "all_solvent_x": torch.stack([_pad1(t, N2_max) for t in solvent_x], dim=0),
        "all_solvent_types": all_solvent_types,
        "all_solvent_masses": atom_types_to_masses(all_solvent_types) * all_solvent_mask,
        "all_solvent_mask": all_solvent_mask,
        "V": V,
        "energy": torch.tensor(energies, dtype=torch.float32),
        "solute_forces": torch.stack([_pad1(t, N1_max) for t in solute_forces], dim=0),
        "solvent_forces": torch.stack([_pad1(t, N2_max) for t in solvent_forces], dim=0),
        "mol_props": torch.stack(mol_props, dim=0),
        "smiles": smiles_list,
        "subset": subset_list,
        "idx": idx_list,
    }
