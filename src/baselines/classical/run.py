from __future__ import annotations

# Needs the mlip-classical image (OpenMM + OpenFF + AmberTools); antechamber / sqm must be on PATH for AM1-BCC.

import argparse
import json
import os
import sys

import numpy as np

from baselines.classical.forcefield import FORCEFIELDS, ClassicalCalculator, bonds_from_geometry
from baselines.evaluation.evaluate import evaluate_calculator
from baselines.export.extxyz import read_extxyz

N_VERIFY_FRAMES = 10
MAX_REPORTED_FAILURES = 20


def topology_drift(calc: ClassicalCalculator, frames) -> dict:
    # The graph of a geometry-built molecule comes from one frame; count how many sampled frames would give a different bond set.
    checked = mismatched = 0
    by_key: dict[str, list] = {}
    for a in frames:
        key = calc.key_for(a)
        if key.startswith("geometry:") and key in calc.graph_bonds:
            by_key.setdefault(key, []).append(a)
    for key, group in by_key.items():
        for i in np.linspace(0, len(group) - 1, min(N_VERIFY_FRAMES, len(group))).astype(int):
            checked += 1
            mismatched += int(bonds_from_geometry(group[i].get_atomic_numbers(), group[i].get_positions()) != calc.graph_bonds[key])
    return {"topology_frames_checked": checked, "topology_frames_mismatched": mismatched}


def evaluate_classical(
    test_xyz: str, forcefield: str, out_path: str | None = None, charge_cache: str | None = None, platform: str = "Reference",
    max_structures: int | None = None, allow_geometry_topology: bool = True,
) -> dict:
    frames = read_extxyz(test_xyz)
    if max_structures is not None:
        frames = frames[:max_structures]
    has_smiles = [("smiles" in a.info) for a in frames]
    if not all(has_smiles) and any(has_smiles):
        raise ValueError("test file mixes frames with and without a smiles tag")
    if not any(has_smiles):
        if not allow_geometry_topology:
            raise ValueError("no smiles tag and geometry topology not allowed for this dataset")
        if len({tuple(a.get_atomic_numbers()) for a in frames}) > 1:
            raise ValueError("frames have different compositions and no smiles tag; a bond graph cannot be inferred reliably (e.g. ANI-1x)")
    calc = ClassicalCalculator(forcefield, charge_cache, platform)
    kept, skipped = [], {}
    for a in frames:
        reason = calc.prepare(a)
        if reason is None:
            kept.append(a)
        else:
            skipped[calc.key_for(a)] = reason
    if not kept:
        raise RuntimeError(f"no molecule could be parameterized; first reasons: {list(skipped.values())[:3]}")
    metrics = evaluate_calculator(calc, kept, relative=True, group_key="smiles" if all(has_smiles) else None)
    metrics.update({
        "model": f"classical_{forcefield}", "forcefield": FORCEFIELDS[forcefield],
        "n_structures_total": len(frames), "n_structures_skipped": len(frames) - len(kept),
        "n_molecules_parameterized": len(calc._calcs), "n_molecules_skipped": len(skipped),
        "skipped_reasons": dict(list(skipped.items())[:MAX_REPORTED_FAILURES]), "charge_cache_hits": calc.charge_cache_hits,
        "test_file": test_xyz,
    })
    metrics.update(topology_drift(calc, kept))
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=1)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Score GAFF2 / OpenFF on an exported test set (energy relative to a per-molecule offset, and forces).")
    p.add_argument("--data-dir", required=True, help="export directory holding test.xyz and split.json")
    p.add_argument("--forcefield", choices=sorted(FORCEFIELDS), required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--charge-cache", default=None, help="directory of cached AM1-BCC charges (shared between jobs)")
    p.add_argument("--platform", default="Reference", help="OpenMM platform; Reference is double precision")
    p.add_argument("--max-structures", type=int, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with open(os.path.join(args.data_dir, "split.json")) as f:
        record = json.load(f)
    if record["dataset"].split(":")[0] in {"ani1x", "qm9"}:
        raise SystemExit(f"{record['dataset']} has no bond topology, so no classical baseline is defined for it")
    metrics = evaluate_classical(
        os.path.join(args.data_dir, record["splits"]["test"]["file"]), args.forcefield, args.out, args.charge_cache, args.platform, args.max_structures)
    print(json.dumps({k: v for k, v in metrics.items() if k != "skipped_reasons"}, indent=1), file=sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
