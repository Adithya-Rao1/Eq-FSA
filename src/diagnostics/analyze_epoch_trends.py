from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diagnostics.analyze_run import load_csv, _floats

_GRAD_BUCKETS = (
    "sr_psi", "sr_pairread", "sr_radial", "sr_selfnorm", "sr_selflin",
    "master", "fusion",
    "other_seedradial", "other_molpropemb", "other_f0ffn", "other_molffn",
    "other_seedprojout", "other_seedprojnorm", "other_seedproj", "other_inputembed",
    "other_layernorm", "other_gate", "other_readout", "other_final", "other",
)
_WEIGHT_BUCKETS = tuple(f"weight_{b}" for b in _GRAD_BUCKETS)


def bin_by_epoch(cols: dict, keys: list[str]) -> dict[str, dict[int, float]]:
    epochs = _floats(cols, "epoch")
    out: dict[str, dict[int, float]] = {k: {} for k in keys}
    for key in keys:
        values = _floats(cols, key)
        by_epoch: dict[int, list[float]] = {}
        for e, v in zip(epochs, values):
            if e is None or v is None:
                continue
            by_epoch.setdefault(int(e), []).append(v)
        for e, vals in by_epoch.items():
            out[key][e] = float(np.median(vals))
    return out


def trend_slope(epoch_values: dict[int, float]) -> dict:
    epochs = sorted(epoch_values.keys())
    if len(epochs) < 3:
        return {"slope": None, "pearson_r": None, "n_epochs": len(epochs)}
    xs = np.array(epochs, dtype=float)
    ys = np.array([epoch_values[e] for e in epochs], dtype=float)
    slope, _intercept = np.polyfit(xs, ys, 1)
    r = float(np.corrcoef(xs, ys)[0, 1]) if ys.std() > 0 else float("nan")
    return {
        "slope": float(slope), "pearson_r": r, "n_epochs": len(epochs),
        "first_value": float(ys[0]), "last_value": float(ys[-1]),
    }


def grad_weight_ratio_by_epoch(binned: dict[str, dict[int, float]]) -> dict[str, dict[int, float]]:
    ratios: dict[str, dict[int, float]] = {}
    for bucket in _GRAD_BUCKETS:
        grad_series = binned.get(bucket, {})
        weight_series = binned.get(f"weight_{bucket}", {})
        ratios[bucket] = {
            e: grad_series[e] / weight_series[e]
            for e in grad_series
            if e in weight_series and weight_series[e] > 1e-12
        }
    return ratios


def epoch_correlation(a: dict[int, float], b: dict[int, float]) -> dict:
    common = sorted(set(a) & set(b))
    if len(common) < 3:
        return {"pearson_r": None, "n_epochs": len(common)}
    xs = np.array([a[e] for e in common])
    ys = np.array([b[e] for e in common])
    if xs.std() == 0 or ys.std() == 0:
        return {"pearson_r": None, "n_epochs": len(common)}
    return {"pearson_r": float(np.corrcoef(xs, ys)[0, 1]), "n_epochs": len(common)}


def _plot_series(binned: dict[str, dict[int, float]], keys: list[str], title: str, ylabel: str, path: str, logy: bool = True):
    plt.figure(figsize=(10, 5))
    any_plotted = False
    for key in keys:
        series = binned.get(key, {})
        if not series:
            continue
        epochs = sorted(series.keys())
        plt.plot(epochs, [series[e] for e in epochs], marker="o", markersize=4, label=key)
        any_plotted = True
    if not any_plotted:
        plt.close()
        return
    if logy:
        plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def analyze(csv_path: str, outdir: str) -> dict:
    os.makedirs(outdir, exist_ok=True)
    cols = load_csv(csv_path)
    if not cols.get("epoch"):
        print(f"No train_batch rows with an epoch column found in {csv_path}")
        return {"error": "no train_batch rows with an epoch column found"}

    keys = ["grad_norm", *_GRAD_BUCKETS, *_WEIGHT_BUCKETS, "alpha", "Z_min", "g_a_norm_max"]
    binned = bin_by_epoch(cols, keys)

    trends = {key: trend_slope(binned[key]) for key in keys if binned[key]}

    ratios = grad_weight_ratio_by_epoch(binned)
    ratio_trends = {bucket: trend_slope(ratios[bucket]) for bucket in ratios if ratios[bucket]}

    correlations = {}
    for a_key, b_key, label in [
        ("weight_master", "alpha", "weight_master_vs_alpha"),
        ("weight_fusion", "alpha", "weight_fusion_vs_alpha"),
        ("weight_master", "grad_norm", "weight_master_vs_grad_norm"),
        ("weight_fusion", "grad_norm", "weight_fusion_vs_grad_norm"),
    ]:
        if binned.get(a_key) and binned.get(b_key):
            correlations[label] = epoch_correlation(binned[a_key], binned[b_key])

    _plot_series(binned, ["grad_norm", *_GRAD_BUCKETS], "Grad-norm baseline (per-epoch median) vs. epoch",
                 "grad L2 norm", os.path.join(outdir, "epoch_grad_norm_baseline.png"))
    _plot_series(binned, list(_WEIGHT_BUCKETS), "Weight-norm buckets (per-epoch median) vs. epoch",
                 "weight L2 norm", os.path.join(outdir, "epoch_weight_norm.png"))
    _plot_series(binned, ["alpha", "Z_min", "g_a_norm_max"], "New candidate diagnostics vs. epoch",
                 "value", os.path.join(outdir, "epoch_candidate_diagnostics.png"), logy=False)
    _plot_series(ratios, list(_GRAD_BUCKETS), "Grad/weight ratio per bucket vs. epoch",
                 "grad_norm / weight_norm", os.path.join(outdir, "epoch_grad_weight_ratio.png"))

    print("\n=== Epoch-level trend slopes (value units per epoch) ===")
    for key, t in trends.items():
        if t["slope"] is not None:
            print(f"  {key:14s} slope={t['slope']:+.4g}/epoch  r={t['pearson_r']:+.3f}  "
                  f"first={t['first_value']:.4g} -> last={t['last_value']:.4g}  (n={t['n_epochs']} epochs)")
        else:
            print(f"  {key:14s} insufficient epochs ({t['n_epochs']})")

    print("\n=== Grad/weight ratio trend per bucket ===")
    for bucket, t in ratio_trends.items():
        if t["slope"] is not None:
            print(f"  {bucket:8s} slope={t['slope']:+.4g}/epoch  first={t['first_value']:.4g} -> last={t['last_value']:.4g}")

    print("\n=== Epoch-level correlations (does weight-norm growth track alpha/grad_norm growth?) ===")
    for name, c in correlations.items():
        print(f"  {name}: r={c['pearson_r']}  (n={c['n_epochs']} epochs)")

    n_epoch_rows = len([e for e in _floats(cols, "epoch") if e is not None])
    return {
        "csv": csv_path, "trends": trends, "ratio_trends": ratio_trends,
        "correlations": correlations, "n_train_batch_rows": n_epoch_rows,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=str, required=True, help="Path to the real run's loss/grad-norm CSV")
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--json-out", type=str, default=None, help="Defaults to <outdir>/epoch_trends.json")
    args = p.parse_args()

    result = analyze(args.csv, args.outdir)

    json_out = args.json_out or os.path.join(args.outdir, "epoch_trends.json")
    os.makedirs(os.path.dirname(json_out) or ".", exist_ok=True)
    with open(json_out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n[analyze_epoch_trends] wrote {json_out}")


if __name__ == "__main__":
    sys.exit(main())
