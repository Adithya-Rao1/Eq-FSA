from __future__ import annotations

from stage_1.experiments.baseline_reference_common import (
    LiteratureReferenceLoader,
)


class COSMORSReferenceLoader(LiteratureReferenceLoader):
    method = "COSMO-RS"


class SCCSReferenceLoader(LiteratureReferenceLoader):
    method = "SCCS"


if __name__ == "__main__":
    import csv
    import os
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "cosmors_example.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["solute_smiles", "solvent_smiles", "method", "value_kcal_per_mol", "source"]
            )
            writer.writeheader()
            writer.writerow({
                "solute_smiles": "CCO", "solvent_smiles": "O", "method": "COSMO-RS",
                "value_kcal_per_mol": "-4.91", "source": "example_doi_placeholder",
            })
            writer.writerow({
                "solute_smiles": "c1ccccc1", "solvent_smiles": "O", "method": "COSMO-RS",
                "value_kcal_per_mol": "-0.65", "source": "example_doi_placeholder",
            })

        loader = COSMORSReferenceLoader()
        refs = loader.load(csv_path)
        assert len(refs) == 2
        assert refs[0].solute_smiles == "CCO" and refs[0].value_kcal_per_mol == -4.91
        assert refs[0].method == "COSMO-RS"
        print(f"Loaded {len(refs)} synthetic COSMO-RS references (smoke test, not real data):")
        for r in refs:
            print(" ", r)

    print(f"\n{COSMORSReferenceLoader.method} and {SCCSReferenceLoader.method} loaders ready; "
          f"no bundled data. Call .load(<path>) with a real CSV in the schema documented in "
          f"baseline_reference_common.py once sourced.")
    print("SMOKE TEST PASSED")