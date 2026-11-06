from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np

from baselines.adapters.dataset_loading import load_splits
from baselines.export.extxyz import ENERGY_KEY, FORCES_KEY, write_extxyz

SPLIT_NAMES = ("train", "val", "test")


def _subsample(n: int, max_n: int | None, seed: int) -> list[int]:
    if max_n is None or max_n >= n:
        return list(range(n))
    # Sorted so the exported file keeps the dataset's own order
    return sorted(np.random.default_rng(seed).permutation(n)[:max_n].tolist())


def _frames(dataset, indices: list[int], seen: dict | None = None):
    # seen (optional) accumulates the element set and the distinct compositions of what is actually written.
    for i in indices:
        item = dataset[i]
        forces = item.get("forces")
        if seen is not None:
            numbers = sorted(item["atomic_numbers"].tolist())
            seen["elements"].update(numbers)
            seen["compositions"].add(tuple(numbers))
        smiles = item.get("smiles")
        yield {
            # Datasets with a molecular graph (SPICE) carry an atom-mapped SMILES; the classical force-field baseline needs it per frame.
            "tags": {"smiles": smiles} if isinstance(smiles, str) else None,
            "atomic_numbers": item["atomic_numbers"].tolist(),
            "positions": item["x"].double().numpy(),
            "energy": float(item["energy"]),
            "forces": forces.double().numpy() if forces is not None else None,
        }


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def export_splits(
    dataset: str, split: str, data_root: str, out_dir: str,
    max_train: int | None = None, max_val: int | None = None, max_test: int | None = None, subset_seed: int = 0,
) -> dict:
    splits = load_splits(dataset, split, data_root)
    limits = dict(zip(SPLIT_NAMES, (max_train, max_val, max_test)))
    record: dict = {
        "dataset": dataset, "split": split, "data_root": data_root, "units": {"energy": "eV", "length": "Angstrom", "forces": "eV/Angstrom"},
        "keys": {"energy": ENERGY_KEY, "forces": FORCES_KEY}, "subset_seed": subset_seed, "splits": {},
    }
    seen: dict = {"elements": set(), "compositions": set()}
    for name, ds in zip(SPLIT_NAMES, splits):
        kept = _subsample(len(ds), limits[name], subset_seed)
        path = os.path.join(out_dir, f"{name}.xyz")
        written = write_extxyz(path, _frames(ds, kept, seen))
        has_forces = bool(written) and "forces" in ds[kept[0]]
        record["splits"][name] = {
            "file": f"{name}.xyz", "n_source": len(ds), "n": written, "max_n": limits[name], "has_forces": has_forces,
            "indices": kept if written != len(ds) else "all", "sha256": _sha256(path),
        }
    # Wrappers need the element list up front 
    from ase.data import chemical_symbols

    record["atomic_numbers"] = sorted(seen["elements"])
    record["species"] = [chemical_symbols[z] for z in record["atomic_numbers"]]
    record["fixed_composition"] = len(seen["compositions"]) <= 1
    with open(os.path.join(out_dir, "split.json"), "w") as f:
        json.dump(record, f, indent=1)
    return record


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export a dataset split to extended XYZ (eV, Angstrom) plus split.json.")
    p.add_argument("--dataset", required=True, help="qm9 | ani1x | spice | rmd17:<molecule> | md22:<molecule>")
    p.add_argument("--split", required=True, help="split name understood by adapters/dataset_loading.py")
    p.add_argument("--data-root", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-train", type=int, default=None, help="seeded subsample cap (applied after the molecule/group-level split)")
    p.add_argument("--max-val", type=int, default=None)
    p.add_argument("--max-test", type=int, default=None)
    p.add_argument("--subset-seed", type=int, default=0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    record = export_splits(args.dataset, args.split, args.data_root, args.out_dir, args.max_train, args.max_val, args.max_test, args.subset_seed)
    for name in SPLIT_NAMES:
        info = record["splits"][name]
        print(f"[export] {name}: {info['n']} of {info['n_source']} -> {os.path.join(args.out_dir, info['file'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
