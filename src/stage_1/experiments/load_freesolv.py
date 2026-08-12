from __future__ import annotations

import dataclasses
import os

from rdkit import Chem

from utils.mappings import ATOM_TYPE_MAP

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATABASE_TXT = os.path.normpath(
    os.path.join(_THIS_DIR, "..", "FreeSolv", "database.txt")
)
WATER_SMILES = "O"


@dataclasses.dataclass
class FreeSolvRecord:
    compound_id: str
    smiles: str
    iupac_name: str
    solvent_smiles: str
    exp_dG: float
    exp_unc: float
    gaff_calc_dG: float
    calc_unc: float
    exp_ref: str
    calc_ref: str
    notes: str


def parse_freesolv_database(path: str = DEFAULT_DATABASE_TXT) -> list[FreeSolvRecord]:
    records: list[FreeSolvRecord] = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            fields = [x.strip() for x in line.split(";")]
            if len(fields) < 10:
                continue
            compound_id, smiles, iupac_name, exp_value, exp_unc, calc_value, calc_unc, exp_ref, calc_ref, notes = fields[:10]
            records.append(FreeSolvRecord(
                compound_id=compound_id,
                smiles=smiles,
                iupac_name=iupac_name,
                solvent_smiles=WATER_SMILES,
                exp_dG=float(exp_value),
                exp_unc=float(exp_unc),
                gaff_calc_dG=float(calc_value),
                calc_unc=float(calc_unc),
                exp_ref=exp_ref,
                calc_ref=calc_ref,
                notes=notes,
            ))
    return records


def _atoms_in_vocab(mol) -> bool:
    return all(atom.GetSymbol() in ATOM_TYPE_MAP for atom in mol.GetAtoms())


def load_freesolv_dataset(path: str = DEFAULT_DATABASE_TXT, verbose: bool = True) -> list[FreeSolvRecord]:
    raw = parse_freesolv_database(path)
    n_raw = len(raw)

    kept: list[FreeSolvRecord] = []
    n_bad_smiles = 0
    n_charged = 0
    n_bad_atoms = 0
    for rec in raw:
        mol = Chem.MolFromSmiles(rec.smiles)
        if mol is None:
            n_bad_smiles += 1
            continue
        if Chem.GetFormalCharge(mol) != 0:
            n_charged += 1
            continue
        if not _atoms_in_vocab(mol):
            n_bad_atoms += 1
            continue
        kept.append(rec)

    if verbose:
        print(
            f"[load_freesolv] raw={n_raw} -> unparsable_smiles={n_bad_smiles} "
            f"charged={n_charged} out_of_vocab_atoms={n_bad_atoms} -> kept={len(kept)}"
        )
    return kept


if __name__ == "__main__":
    recs = load_freesolv_dataset()
    print(f"\nLoaded {len(recs)} FreeSolv records. First 3:")
    for r in recs[:3]:
        print(" ", r)