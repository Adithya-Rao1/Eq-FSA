from __future__ import annotations

import dataclasses
import os
import re
import sys

from rdkit import Chem

from utils.mappings import ATOM_TYPE_MAP

for _optional_dep in ("pandas",):
    if _optional_dep not in sys.modules:
        try:
            __import__(_optional_dep)
        except Exception:
            sys.modules[_optional_dep] = None

from stage_1.solvatum.solvatum.ui import calculate_g_solv

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SDF_PATH = os.path.normpath(
    os.path.join(_THIS_DIR, "..", "solvatum", "solvatum", "data", "solvatum.sdf")
)

_LOGK_PREFIX = "logK ("
_LOGK_SUFFIX = ")"
_T_STANDARD_K = 298.15
_STEREO_PREFIX_RE = re.compile(r"^\(\\[Xx][Bb]1\)-\s*")


def _normalize_solvent_name(raw: str) -> str:
    name = _STEREO_PREFIX_RE.sub("", raw.strip())
    return " ".join(name.upper().split())

_SOLVENT_SMILES_FALLBACK: dict[str, str] = {
    "1-ETHYL-2-PYRROLIDINONE": "CCN1CCCC1=O",
    "1-HEXADECENE": "CCCCCCCCCCCCCCC=C",
    "1-METHYL-2-PIPERIDINONE": "CN1CCCCC1=O",
    "1,5-DIMETHYL-2-PYRROLIDINONE": "CC1CCC(=O)N1C",
    "5,8,11,14-TETRAOXAOCTADECANE": "CCCCOCCOCCOCCOCCCC",
    "DIBENZYL ETHER": "c1ccc(COCc2ccccc2)cc1",
    "1,2-DICHLOROETHANE": "ClCCCl",
    "DICHLOROETHANE": "ClCCCl",
    "DIETHYLDIGLYCOL": "CCOCCOCCOCC",
    "DIETHYLENE GLYCOL DIBUTYL ETHER": "CCCCOCCOCCOCCCC",
    "ETHYLBENZOATE": "CCOC(=O)c1ccccc1",
    "ETHYLENE GLYCOL": "OCCO",
    "HEPTYLACETATE": "CCCCCCCOC(C)=O",
    "HEPTYLACETAT": "CCCCCCCOC(C)=O",
    "N-ETHYLACETAMIDE": "CCNC(C)=O",
    "N-ETHYLFORMAMIDE": "CCNC=O",
    "N,N-DIBUTYLFORMAMIDE": "CCCCN(CCCC)C=O",
    "N,N-DIBUTYLFORMAMID": "CCCCN(CCCC)C=O",
    "N,N-DIETHYLACETAMIDE": "CCN(CC)C(C)=O",
    "PENTAN-3-OL": "CCC(O)CC",
    "SULFOLANE": "O=S1(=O)CCCC1",
    "TETRADECANE": "CCCCCCCCCCCCCC",
    "TETRAETHYLENE GLYCOL DIMETHYL ETHER": "COCCOCCOCCOCCOC",
}


@dataclasses.dataclass
class SolvTumRecord:
    solute_name: str
    solute_smiles: str
    solvent_name: str
    solvent_smiles: str | None
    solvent_smiles_source: str  # "solute_record" | "fallback_table" | "unresolved"
    logK: float
    exp_dG_kcal_per_mol: float


def _iter_sdf_mols(path: str):
    supplier = Chem.SDMolSupplier(path, sanitize=False)
    for mol in supplier:
        if mol is not None:
            yield mol


def _build_solute_smiles_by_name(path: str) -> dict[str, str]:
    by_name: dict[str, str] = {}
    for mol in _iter_sdf_mols(path):
        if mol.HasProp("SMILES") and mol.HasProp("Name"):
            by_name[_normalize_solvent_name(mol.GetProp("Name"))] = mol.GetProp("SMILES")
    return by_name


def parse_solvtum_sdf(path: str = DEFAULT_SDF_PATH) -> list[SolvTumRecord]:
    solute_smiles_by_name = _build_solute_smiles_by_name(path)

    records: list[SolvTumRecord] = []
    for mol in _iter_sdf_mols(path):
        if not mol.HasProp("SMILES") or not mol.HasProp("Name"):
            continue
        solute_smiles = mol.GetProp("SMILES")
        solute_name = mol.GetProp("Name")

        for key in mol.GetPropNames():
            if not (key.startswith(_LOGK_PREFIX) and key.endswith(_LOGK_SUFFIX)):
                continue
            raw_solvent_name = key[len(_LOGK_PREFIX):-len(_LOGK_SUFFIX)]
            solvent_name = _normalize_solvent_name(raw_solvent_name)
            logK = float(mol.GetProp(key))

            solvent_smiles = solute_smiles_by_name.get(solvent_name)
            source = "solute_record"
            if solvent_smiles is None:
                solvent_smiles = _SOLVENT_SMILES_FALLBACK.get(solvent_name)
                source = "fallback_table" if solvent_smiles is not None else "unresolved"

            records.append(SolvTumRecord(
                solute_name=solute_name,
                solute_smiles=solute_smiles,
                solvent_name=solvent_name,
                solvent_smiles=solvent_smiles,
                solvent_smiles_source=source,
                logK=logK,
                exp_dG_kcal_per_mol=calculate_g_solv(logK, unit="kcal/mol", temp=_T_STANDARD_K),
            ))
    return records


def _atoms_in_vocab(mol) -> bool:
    return all(atom.GetSymbol() in ATOM_TYPE_MAP for atom in mol.GetAtoms())


def load_solvtum_dataset(path: str = DEFAULT_SDF_PATH, verbose: bool = True) -> list[SolvTumRecord]:
    raw = parse_solvtum_sdf(path)
    n_raw = len(raw)

    unresolved_solvents: set[str] = set()
    n_unresolved_solvent = 0
    n_bad_solute_smiles = 0
    n_charged = 0
    n_bad_atoms_solute = 0
    n_bad_atoms_solvent = 0
    kept: list[SolvTumRecord] = []

    for rec in raw:
        if rec.solvent_smiles_source == "unresolved":
            unresolved_solvents.add(rec.solvent_name)
            n_unresolved_solvent += 1
            continue

        solute_mol = Chem.MolFromSmiles(rec.solute_smiles)
        if solute_mol is None:
            n_bad_solute_smiles += 1
            continue
        if Chem.GetFormalCharge(solute_mol) != 0:
            n_charged += 1
            continue
        if not _atoms_in_vocab(solute_mol):
            n_bad_atoms_solute += 1
            continue

        solvent_mol = Chem.MolFromSmiles(rec.solvent_smiles)
        if solvent_mol is None or not _atoms_in_vocab(solvent_mol):
            n_bad_atoms_solvent += 1
            continue

        kept.append(rec)

    if verbose:
        print(
            f"[load_solvtum] raw={n_raw} -> unresolved_solvent_smiles={n_unresolved_solvent} "
            f"unparsable_solute_smiles={n_bad_solute_smiles} charged_solute={n_charged} "
            f"solute_out_of_vocab={n_bad_atoms_solute} solvent_out_of_vocab={n_bad_atoms_solvent} "
            f"-> kept={len(kept)}"
        )
        if unresolved_solvents:
            print(f"[load_solvtum] {len(unresolved_solvents)} solvent name(s) with NO SMILES from "
                  f"either the database's own solute records or the fallback table -- these pairs "
                  f"were dropped, not fabricated: {sorted(unresolved_solvents)}")

    return kept


if __name__ == "__main__":
    all_raw = parse_solvtum_sdf()
    unique_solvents = sorted({r.solvent_name for r in all_raw})
    by_source: dict[str, list[str]] = {"solute_record": [], "fallback_table": [], "unresolved": []}
    seen = set()
    for r in all_raw:
        if r.solvent_name not in seen:
            seen.add(r.solvent_name)
            by_source[r.solvent_smiles_source].append(r.solvent_name)

    print(f"Raw pairs: {len(all_raw)}, unique solutes: {len({r.solute_smiles for r in all_raw})}, "
          f"unique solvents: {len(unique_solvents)}")
    print(f"Solvent SMILES source breakdown: "
          f"from own solute record={len(by_source['solute_record'])}, "
          f"from fallback table={len(by_source['fallback_table'])}, "
          f"unresolved={len(by_source['unresolved'])}")
    if by_source["unresolved"]:
        print(f"  UNRESOLVED: {by_source['unresolved']}")
    if by_source["fallback_table"]:
        print(f"  from fallback table (spot-check these): {by_source['fallback_table']}")

    recs = load_solvtum_dataset(verbose=True)
    print(f"\nLoaded {len(recs)} Solv@TUM records. First 3:")
    for r in recs[:3]:
        print(" ", r)