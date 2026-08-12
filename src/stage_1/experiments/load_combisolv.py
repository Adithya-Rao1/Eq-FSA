from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional

import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch.utils.data import Dataset

from utils.mappings import ATOM_TYPE_MAP, atom_types_to_masses
from utils.calc_props import calc_rdkit_mol_props_smiles, _MOL_FEAT_DIM
from stage_1.spice_dataset import build_rdkit_edge_features
from utils.pw_bases import compute_molecular_box
from stage_1.experiments.load_freesolv import load_freesolv_dataset
from stage_1.experiments.load_solvtum import load_solvtum_dataset

MORGAN_RADIUS = 2
MORGAN_NBITS = 2048
SOLVENT_DESCRIPTOR_DIM = MORGAN_NBITS + _MOL_FEAT_DIM

DEFAULT_BOX_PADDING = 12.0
DEFAULT_EDGE_CUTOFF = 5.0
ISOLATION_CUTOFF_ANGSTROM = 5.0
DEFAULT_EMBED_ATTEMPTS = 5


@dataclasses.dataclass
class CombiSolvPair:
    solute_smiles: str
    solvent_smiles: str
    exp_dG_kcal_per_mol: float
    source: str  # "FreeSolv" | "SolvTUM"
    solvent_name: Optional[str] = None  # SolvTUM only
    compound_id: Optional[str] = None   # FreeSolv only


def _freesolv_to_pairs(verbose: bool) -> List[CombiSolvPair]:
    return [
        CombiSolvPair(
            solute_smiles=r.smiles,
            solvent_smiles=r.solvent_smiles,
            exp_dG_kcal_per_mol=r.exp_dG,
            source="FreeSolv",
            compound_id=r.compound_id,
        )
        for r in load_freesolv_dataset(verbose=verbose)
    ]


def _solvtum_to_pairs(verbose: bool) -> List[CombiSolvPair]:
    return [
        CombiSolvPair(
            solute_smiles=r.solute_smiles,
            solvent_smiles=r.solvent_smiles,
            exp_dG_kcal_per_mol=r.exp_dG_kcal_per_mol,
            source="SolvTUM",
            solvent_name=r.solvent_name,
        )
        for r in load_solvtum_dataset(verbose=verbose)
    ]


def load_combined_pairs(verbose: bool = True) -> List[CombiSolvPair]:
    pairs = _freesolv_to_pairs(verbose) + _solvtum_to_pairs(verbose)
    if verbose:
        n_fs = sum(1 for p in pairs if p.source == "FreeSolv")
        n_st = sum(1 for p in pairs if p.source == "SolvTUM")
        print(f"[load_combisolv] combined: FreeSolv={n_fs} + Solv@TUM={n_st} = {len(pairs)} pairs")
    return pairs


def _embed_molecule_conformer(
    smiles: str,
    max_attempts: int = DEFAULT_EMBED_ATTEMPTS,
    isolation_cutoff: float = ISOLATION_CUTOFF_ANGSTROM,
) -> Optional[torch.Tensor]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)

    for attempt in range(max_attempts):
        params = AllChem.ETKDGv3()
        params.randomSeed = attempt
        params.useRandomCoords = attempt > 0
        conf_id = AllChem.EmbedMolecule(mol, params)
        if conf_id < 0:
            continue

        try:
            if AllChem.MMFFHasAllMoleculeParams(mol):
                AllChem.MMFFOptimizeMolecule(mol, confId=conf_id)
            else:
                AllChem.UFFOptimizeMolecule(mol, confId=conf_id)
        except Exception:
            pass  

        conf = mol.GetConformer(conf_id)
        pos = torch.tensor(conf.GetPositions(), dtype=torch.float32)
        n = pos.shape[0]
        if n < 2:
            return pos 

        dists = torch.cdist(pos, pos)
        dists.fill_diagonal_(float("inf"))
        nearest = dists.min(dim=-1).values
        if bool((nearest <= isolation_cutoff).all()):
            return pos

    return None


def _atom_types_from_smiles(smiles: str) -> torch.Tensor:
    mol_h = Chem.AddHs(Chem.MolFromSmiles(smiles))
    return torch.tensor(
        [ATOM_TYPE_MAP.get(atom.GetSymbol(), 0) for atom in mol_h.GetAtoms()],
        dtype=torch.long,
    )


class Stage2SolvationDataset(Dataset):
    def __init__(
        self,
        pairs: Optional[List[CombiSolvPair]] = None,
        box_padding: float = DEFAULT_BOX_PADDING,
        edge_cutoff: float = DEFAULT_EDGE_CUTOFF,
        verbose: bool = True,
    ):
        self.box_padding = box_padding
        self.edge_cutoff = edge_cutoff

        raw_pairs = pairs if pairs is not None else load_combined_pairs(verbose=verbose)

        self._conformer_cache: Dict[str, Optional[torch.Tensor]] = {}
        self._types_cache: Dict[str, torch.Tensor] = {}
        self._mol_props_cache: Dict[str, torch.Tensor] = {}
        self._edge_feat_cache: Dict[str, torch.Tensor] = {}
        self._solvent_descriptor_cache: Dict[str, Optional[torch.Tensor]] = {}

        self.pairs: List[CombiSolvPair] = []
        n_embed_failed = 0
        for pair in raw_pairs:
            if self._conformer(pair.solute_smiles) is None:
                n_embed_failed += 1
                continue
            if self._conformer(pair.solvent_smiles) is None:
                n_embed_failed += 1
                continue
            if self._solvent_descriptor(pair.solvent_smiles) is None:
                n_embed_failed += 1
                continue
            self.pairs.append(pair)

        if verbose:
            print(f"[Stage2SolvationDataset] {len(raw_pairs)} pairs -> "
                  f"{n_embed_failed} dropped (conformer embedding/descriptor failure) -> "
                  f"{len(self.pairs)} usable")

    def _conformer(self, smiles: str) -> Optional[torch.Tensor]:
        if smiles not in self._conformer_cache:
            self._conformer_cache[smiles] = _embed_molecule_conformer(smiles)
        return self._conformer_cache[smiles]

    def _types(self, smiles: str) -> torch.Tensor:
        if smiles not in self._types_cache:
            self._types_cache[smiles] = _atom_types_from_smiles(smiles)
        return self._types_cache[smiles]

    def _mol_props(self, smiles: str) -> torch.Tensor:
        if smiles not in self._mol_props_cache:
            self._mol_props_cache[smiles] = torch.tensor(
                calc_rdkit_mol_props_smiles(smiles), dtype=torch.float32
            )
        return self._mol_props_cache[smiles]

    def _edge_feats(self, smiles: str, n_atoms: int) -> torch.Tensor:
        key = f"{smiles}:{n_atoms}"
        if key not in self._edge_feat_cache:
            self._edge_feat_cache[key] = build_rdkit_edge_features(smiles, n_atoms)
        return self._edge_feat_cache[key]

    def _solvent_descriptor(self, smiles: str) -> Optional[torch.Tensor]:
        if smiles not in self._solvent_descriptor_cache:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                self._solvent_descriptor_cache[smiles] = None
            else:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=MORGAN_NBITS)
                fp_tensor = torch.tensor(list(fp), dtype=torch.float32)
                props_tensor = torch.tensor(calc_rdkit_mol_props_smiles(smiles), dtype=torch.float32)
                self._solvent_descriptor_cache[smiles] = torch.cat([fp_tensor, props_tensor], dim=0)
        return self._solvent_descriptor_cache[smiles]

    def _entity(self, smiles: str, with_edge_feats: bool) -> Dict[str, torch.Tensor]:
        x = self._conformer(smiles)
        types = self._types(smiles)
        n_atoms = x.shape[0]
        out = {
            "x": x,
            "types": types,
            "mask": torch.ones(n_atoms, dtype=torch.bool),
            "masses": atom_types_to_masses(types),
            "mol_props": self._mol_props(smiles),
        }
        if with_edge_feats:
            out["edge_feats"] = self._edge_feats(smiles, n_atoms)
        return out

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        pair = self.pairs[idx]

        solute = self._entity(pair.solute_smiles, with_edge_feats=True)
        solvent = self._entity(pair.solvent_smiles, with_edge_feats=False)

        V = compute_molecular_box(solute["x"], self.box_padding, self.box_padding, self.edge_cutoff)

        return {
            "idx": idx,
            "solute_x": solute["x"],
            "solute_types": solute["types"],
            "solute_mask": solute["mask"],
            "solute_masses": solute["masses"],
            "solute_mol_props": solute["mol_props"],
            "solute_edge_feats": solute["edge_feats"],
            "solvent_x": solvent["x"],
            "solvent_types": solvent["types"],
            "solvent_mask": solvent["mask"],
            "solvent_masses": solvent["masses"],
            "solvent_mol_props": solvent["mol_props"],
            "solvent_descriptor": self._solvent_descriptor(pair.solvent_smiles),
            "V": V,
            "target_dG": pair.exp_dG_kcal_per_mol,
            "solute_smiles": pair.solute_smiles,
            "solvent_smiles": pair.solvent_smiles,
            "source": pair.source,
        }


def _pad_1d(t: torch.Tensor, n_max: int) -> torch.Tensor:
    if t.shape[0] == n_max:
        return t
    pad_shape = (n_max - t.shape[0],) + tuple(t.shape[1:])
    return torch.cat([t, torch.zeros(pad_shape, dtype=t.dtype)], dim=0)


def _pad_edge_feats(t: torch.Tensor, n_max: int) -> torch.Tensor:
    n = t.shape[0]
    if n == n_max:
        return t
    out = torch.zeros(n_max, n_max, t.shape[-1], dtype=t.dtype)
    out[:n, :n] = t
    return out


def stage2_collate_fn(batch: List[Dict[str, object]]) -> Dict[str, object]:
    n_max_solute = max(item["solute_x"].shape[0] for item in batch)
    n_max_solvent = max(item["solvent_x"].shape[0] for item in batch)

    return {
        "idx": [item["idx"] for item in batch],
        "solute_x": torch.stack([_pad_1d(item["solute_x"], n_max_solute) for item in batch]),
        "solute_types": torch.stack([_pad_1d(item["solute_types"], n_max_solute) for item in batch]),
        "solute_mask": torch.stack([_pad_1d(item["solute_mask"], n_max_solute) for item in batch]),
        "solute_masses": torch.stack([_pad_1d(item["solute_masses"], n_max_solute) for item in batch]),
        "solute_mol_props": torch.stack([item["solute_mol_props"] for item in batch]),
        "solute_edge_feats": torch.stack(
            [_pad_edge_feats(item["solute_edge_feats"], n_max_solute) for item in batch]
        ),
        "solvent_x": torch.stack([_pad_1d(item["solvent_x"], n_max_solvent) for item in batch]),
        "solvent_types": torch.stack([_pad_1d(item["solvent_types"], n_max_solvent) for item in batch]),
        "solvent_mask": torch.stack([_pad_1d(item["solvent_mask"], n_max_solvent) for item in batch]),
        "solvent_masses": torch.stack([_pad_1d(item["solvent_masses"], n_max_solvent) for item in batch]),
        "solvent_mol_props": torch.stack([item["solvent_mol_props"] for item in batch]),
        "solvent_descriptor": torch.stack([item["solvent_descriptor"] for item in batch]),
        "V": torch.stack([item["V"] for item in batch]),
        "target_dG": torch.tensor([item["target_dG"] for item in batch], dtype=torch.float32),
        "solute_smiles": [item["solute_smiles"] for item in batch],
        "solvent_smiles": [item["solvent_smiles"] for item in batch],
        "source": [item["source"] for item in batch],
    }


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit", type=int, default=40, help="Only build the dataset from the first N pairs (fast smoke test)")
    args = p.parse_args()

    all_pairs = load_combined_pairs(verbose=True)
    subset = all_pairs[: args.limit] if args.limit else all_pairs
    dataset = Stage2SolvationDataset(pairs=subset, verbose=True)

    print(f"\nDataset size: {len(dataset)}")
    if len(dataset) > 0:
        sample = dataset[0]
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")
            else:
                print(f"  {k}: {v}")

        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=min(4, len(dataset)), collate_fn=stage2_collate_fn)
        batch = next(iter(loader))
        print("\nCollated batch shapes:")
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")