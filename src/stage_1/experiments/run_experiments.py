from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Optional

import torch

from stage_1.experiments.load_combisolv import (
    CombiSolvPair, Stage2SolvationDataset, SOLVENT_DESCRIPTOR_DIM, load_combined_pairs,
)
from stage_1.experiments.baseline_reference_common import (
    LiteratureReferenceLoader, compute_metrics, plot_parity, plot_method_comparison, RegressionMetrics,
)
from stage_1.experiments.explicit_models import GAFFReferenceLoader
from stage_1.experiments.implicit_models import COSMORSReferenceLoader, SCCSReferenceLoader
from stage_1.experiments.deep_learning_models import MACEBaseline, FieldSchNetBaseline

EQFSA_NOTE = ("")


def _split_pairs(pairs: List[CombiSolvPair], test_fraction: float, seed: int) -> tuple[List[CombiSolvPair], List[CombiSolvPair]]:
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(pairs), generator=g).tolist()
    n_test = max(int(len(pairs) * test_fraction), 1)
    test_idx = set(idx[:n_test])
    train = [p for i, p in enumerate(pairs) if i not in test_idx]
    test = [p for i, p in enumerate(pairs) if i in test_idx]
    return train, test


def _run_reference_baseline(
    loader: LiteratureReferenceLoader,
    refs,
    test_pairs: List[CombiSolvPair],
    verbose: bool = True,
) -> Dict[tuple, float]:
    return loader.match_to_dataset(refs, test_pairs, verbose=verbose)


def _run_dl_baseline(baseline, train_dataset: Stage2SolvationDataset, test_pairs: List[CombiSolvPair],
                      n_epochs: int, batch_size: int, verbose: bool = True) -> Dict[tuple, float]:
    baseline.fine_tune(train_dataset, n_epochs=n_epochs, batch_size=batch_size, verbose=verbose)

    test_dataset = Stage2SolvationDataset(pairs=test_pairs, verbose=verbose)
    preds: Dict[tuple, float] = {}
    for item in test_dataset:
        key = (item["solute_smiles"], item["solvent_smiles"])
        preds[key] = baseline.predict(item["solute_smiles"], item["solute_x"], item["solvent_descriptor"])
    return preds


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit", type=int, default=None, help="Only use the first N combined pairs (omit for the full dataset)")
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=str, default="stage2_baseline_results")
    p.add_argument("--skip-dl", action="store_true", help="Skip MACE/FieldSchNet fine-tuning (reference baselines only, fast)")
    p.add_argument("--skip-fieldschnet", action="store_true", help="Skip FieldSchNet specifically (needs schnetpack==0.3)")
    p.add_argument("--dl-epochs", type=int, default=10)
    p.add_argument("--dl-batch-size", type=int, default=16)
    p.add_argument("--cosmors-csv", type=str, default=None, help="External CSV in baseline_reference_common's schema; omit to skip COSMO-RS")
    p.add_argument("--sccs-csv", type=str, default=None, help="External CSV in baseline_reference_common's schema; omit to skip SCCS")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    all_pairs = load_combined_pairs(verbose=True)
    if args.limit:
        all_pairs = all_pairs[: args.limit]
    train_pairs, test_pairs = _split_pairs(all_pairs, args.test_fraction, args.seed)
    print(f"[run_experiments] {len(all_pairs)} pairs -> train={len(train_pairs)} test={len(test_pairs)}")

    predictions: Dict[str, Dict[tuple, float]] = {}

    gaff_refs = GAFFReferenceLoader().from_freesolv(verbose=True)
    predictions["GAFF"] = _run_reference_baseline(GAFFReferenceLoader(), gaff_refs, test_pairs)

    if args.cosmors_csv:
        cosmors_refs = COSMORSReferenceLoader().load(args.cosmors_csv)
        predictions["COSMO-RS"] = _run_reference_baseline(COSMORSReferenceLoader(), cosmors_refs, test_pairs)
    else:
        print("[run_experiments] --cosmors-csv not given -- skipping COSMO-RS (no bundled data, see implicit_models.py)")

    if args.sccs_csv:
        sccs_refs = SCCSReferenceLoader().load(args.sccs_csv)
        predictions["SCCS"] = _run_reference_baseline(SCCSReferenceLoader(), sccs_refs, test_pairs)
    else:
        print("[run_experiments] --sccs-csv not given -- skipping SCCS (no bundled data, see implicit_models.py)")

    if not args.skip_dl:
        train_dataset = Stage2SolvationDataset(pairs=train_pairs, verbose=True)
        if len(train_dataset) < 2:
            print("[run_experiments] Not enough usable training pairs for DL fine-tuning -- skipping MACE/FieldSchNet")
        else:
            print("\n[run_experiments] Fine-tuning MACEBaseline...")
            mace_baseline = MACEBaseline(solvent_descriptor_dim=SOLVENT_DESCRIPTOR_DIM, model="small", device="cpu")
            predictions["MACE"] = _run_dl_baseline(
                mace_baseline, train_dataset, test_pairs, args.dl_epochs, args.dl_batch_size,
            )

            if not args.skip_fieldschnet:
                print("\n[run_experiments] Fine-tuning FieldSchNetBaseline...")
                model_path = os.path.normpath(os.path.join(
                    os.path.dirname(__file__), "..", "field_schnet", "example", "ethanol_vacuum.model",
                ))
                fs_baseline = FieldSchNetBaseline(
                    solvent_descriptor_dim=SOLVENT_DESCRIPTOR_DIM, model_path=model_path, device="cpu",
                )
                predictions["FieldSchNet"] = _run_dl_baseline(
                    fs_baseline, train_dataset, test_pairs, args.dl_epochs, args.dl_batch_size,
                )
    else:
        print("[run_experiments] --skip-dl: not fine-tuning MACE/FieldSchNet")

    print(f"\n[run_experiments] {EQFSA_NOTE}")

    methods = list(predictions.keys())
    csv_path = os.path.join(args.outdir, "results.csv")
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["solute_smiles", "solvent_smiles", "exp_dG", "eqfsa_pred"]
        for m in methods:
            fieldnames += [f"{m}_pred", f"{m}_abs_error"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for pair in test_pairs:
            key = (pair.solute_smiles, pair.solvent_smiles)
            row = {
                "solute_smiles": pair.solute_smiles,
                "solvent_smiles": pair.solvent_smiles,
                "exp_dG": pair.exp_dG_kcal_per_mol,
                "eqfsa_pred": "",  
            }
            for m in methods:
                pred = predictions[m].get(key)
                row[f"{m}_pred"] = pred if pred is not None else ""
                row[f"{m}_abs_error"] = abs(pred - pair.exp_dG_kcal_per_mol) if pred is not None else ""
            writer.writerow(row)
    print(f"\n[run_experiments] Wrote per-pair results: {csv_path}")

    metrics_by_method: Dict[str, RegressionMetrics] = {}
    for m in methods:
        y_true, y_pred = [], []
        for pair in test_pairs:
            key = (pair.solute_smiles, pair.solvent_smiles)
            pred = predictions[m].get(key)
            if pred is not None:
                y_true.append(pair.exp_dG_kcal_per_mol)
                y_pred.append(pred)
        if not y_true:
            print(f"[run_experiments] {m}: no matched predictions on the test set -- skipping metrics")
            continue
        metrics = plot_parity(y_true, y_pred, m, os.path.join(args.outdir, f"parity_{m}.png"))
        metrics_by_method[m] = metrics
        print(f"[run_experiments] {m}: n={metrics.n} RMSE={metrics.rmse_kcal_per_mol:.3f} "
              f"MAE={metrics.mae_kcal_per_mol:.3f} R^2={metrics.r2:.3f} "
              f"(coverage: {metrics.n}/{len(test_pairs)} test pairs)")

    if metrics_by_method:
        plot_method_comparison(metrics_by_method, os.path.join(args.outdir, "method_comparison.png"))
        print(f"[run_experiments] Wrote parity plots and method_comparison.png to {args.outdir}")
    else:
        print("[run_experiments] No baseline produced any matched predictions -- nothing to plot")


if __name__ == "__main__":
    main()