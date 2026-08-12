"""
Expected external CSV schema (for any reference not already derivable from data already in this
repo, e.g. SCCS/COSMO-RS or a future Solv@TUM GAFF source): columns
`solute_smiles,solvent_smiles,method,value_kcal_per_mol,source` -- one row per (solute, solvent,
method) triple. `source` is a free-text citation key or DOI, not parsed further here.
"""
from __future__ import annotations

import csv
import dataclasses
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclasses.dataclass
class LiteratureReference:
    solute_smiles: str
    solvent_smiles: str
    method: str  # "GAFF" | "SCCS" | "COSMO-RS"
    value_kcal_per_mol: float
    source: str


_CSV_FIELDS = ["solute_smiles", "solvent_smiles", "method", "value_kcal_per_mol", "source"]


class LiteratureReferenceLoader:
    method: str = "UNSET"

    def load(self, path: str) -> List[LiteratureReference]:
        refs: List[LiteratureReference] = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            missing = set(_CSV_FIELDS) - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{path}: missing required column(s) {sorted(missing)}; "
                                  f"expected {_CSV_FIELDS}")
            for row in reader:
                refs.append(LiteratureReference(
                    solute_smiles=row["solute_smiles"],
                    solvent_smiles=row["solvent_smiles"],
                    method=row["method"],
                    value_kcal_per_mol=float(row["value_kcal_per_mol"]),
                    source=row["source"],
                ))
        return refs

    @staticmethod
    def match_to_dataset(
        refs: List[LiteratureReference],
        pairs,  # List[CombiSolvPair]-like: objects with .solute_smiles / .solvent_smiles
        verbose: bool = True,
    ) -> Dict[Tuple[str, str], float]:
        ref_lookup = {(r.solute_smiles, r.solvent_smiles): r.value_kcal_per_mol for r in refs}
        dataset_keys = {(p.solute_smiles, p.solvent_smiles) for p in pairs}

        matched = {k: v for k, v in ref_lookup.items() if k in dataset_keys}
        if verbose:
            print(f"[LiteratureReferenceLoader] {len(refs)} references, {len(dataset_keys)} dataset "
                  f"pairs -> {len(matched)} matched ({100 * len(matched) / max(len(dataset_keys), 1):.1f}% "
                  f"of dataset pairs covered)")
        return matched


@dataclasses.dataclass
class RegressionMetrics:
    n: int
    rmse_kcal_per_mol: float
    mae_kcal_per_mol: float
    pearson_r: float
    r2: float

def compute_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> RegressionMetrics:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: y_true={y_true.shape} y_pred={y_pred.shape}")
    n = y_true.shape[0]
    if n == 0:
        return RegressionMetrics(n=0, rmse_kcal_per_mol=float("nan"), mae_kcal_per_mol=float("nan"),
                                  pearson_r=float("nan"), r2=float("nan"))

    diff = y_pred - y_true
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))

    if n < 2 or y_true.std() == 0 or y_pred.std() == 0:
        pearson_r = float("nan")
    else:
        pearson_r = float(np.corrcoef(y_true, y_pred)[0, 1])
    r2 = pearson_r ** 2 if np.isfinite(pearson_r) else float("nan")

    return RegressionMetrics(n=n, rmse_kcal_per_mol=rmse, mae_kcal_per_mol=mae,
                              pearson_r=pearson_r, r2=r2)


def plot_parity(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    method_name: str,
    path: str,
    unit: str = "kcal/mol",
):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    metrics = compute_metrics(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=18, alpha=0.6, edgecolors="none")

    lo = float(min(y_true.min(), y_pred.min())) if len(y_true) else -1.0
    hi = float(max(y_true.max(), y_pred.max())) if len(y_true) else 1.0
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="gray", linewidth=1, linestyle="--")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)

    ax.set_xlabel(f"Experimental dG ({unit})")
    ax.set_ylabel(f"Predicted dG ({unit})")
    ax.set_title(
        f"{method_name}  (n={metrics.n})\n"
        f"RMSE={metrics.rmse_kcal_per_mol:.3f}  MAE={metrics.mae_kcal_per_mol:.3f}  "
        f"R²={metrics.r2:.3f}"
    )
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return metrics


def plot_method_comparison(results: Dict[str, RegressionMetrics], path: str):
    methods = list(results.keys())
    rmses = [results[m].rmse_kcal_per_mol for m in methods]
    maes = [results[m].mae_kcal_per_mol for m in methods]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(methods)), 5))
    ax.bar(x - width / 2, rmses, width, label="RMSE")
    ax.bar(x + width / 2, maes, width, label="MAE")
    for i, m in enumerate(methods):
        ax.annotate(f"R²={results[m].r2:.2f}", (x[i], max(rmses[i], maes[i])),
                    textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylabel("kcal/mol")
    ax.set_title("Baseline comparison")
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    y_true = rng.normal(-3.0, 2.0, 200)
    y_pred_good = y_true + rng.normal(0, 0.5, 200)
    y_pred_bad = rng.normal(-3.0, 2.0, 200)

    good = compute_metrics(y_true, y_pred_good)
    bad = compute_metrics(y_true, y_pred_bad)
    print("good (noisy but correlated):", good)
    print("bad (uncorrelated):", bad)
    assert good.r2 > 0.7, "expected strong correlation for the noisy-but-correlated case"
    assert bad.r2 < 0.1, "expected near-zero correlation for the random case"
    print("SMOKE TEST PASSED")