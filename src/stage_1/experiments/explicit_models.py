from __future__ import annotations

from typing import List

from stage_1.experiments.baseline_reference_common import (
    LiteratureReference,
    LiteratureReferenceLoader,
)
from stage_1.experiments.load_freesolv import load_freesolv_dataset


class GAFFReferenceLoader(LiteratureReferenceLoader):
    method = "GAFF"

    def from_freesolv(self, verbose: bool = True) -> List[LiteratureReference]:
        records = load_freesolv_dataset(verbose=verbose)
        return [
            LiteratureReference(
                solute_smiles=r.smiles,
                solvent_smiles=r.solvent_smiles,
                method=self.method,
                value_kcal_per_mol=r.gaff_calc_dG,
                source=r.calc_ref,
            )
            for r in records
        ]


if __name__ == "__main__":
    from stage_1.experiments.load_freesolv import parse_freesolv_database

    loader = GAFFReferenceLoader()
    refs = loader.from_freesolv(verbose=True)
    print(f"\nLoaded {len(refs)} GAFF references from FreeSolv. First 3:")
    for r in refs[:3]:
        print(" ", r)
        
    raw_by_id = {r.compound_id: r for r in parse_freesolv_database()}
    ref_by_smiles = {r.solute_smiles: r.value_kcal_per_mol for r in refs}
    n_checked = 0
    for compound_id, raw in list(raw_by_id.items())[:20]:
        if raw.smiles in ref_by_smiles:
            assert ref_by_smiles[raw.smiles] == raw.gaff_calc_dG, (
                f"{compound_id}: GAFFReferenceLoader value {ref_by_smiles[raw.smiles]} != "
                f"raw database.txt value {raw.gaff_calc_dG}"
            )
            n_checked += 1
    print(f"\nSpot-checked {n_checked} records against raw database.txt -- exact match confirmed.")