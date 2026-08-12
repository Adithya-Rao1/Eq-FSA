from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_csv(path: str) -> dict[str, list]:
    cols: dict[str, list] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("phase") != "train_batch":
                continue
            for k, v in row.items():
                cols.setdefault(k, []).append(v)
    return cols


def _floats(cols: dict, key: str) -> list[float | None]:
    out = []
    for v in cols.get(key, []):
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            out.append(None)
    return out


def _plot(x: list[float], series: dict[str, list], title: str, ylabel: str, path: str):
    plt.figure(figsize=(10, 5))
    for label, y in series.items():
        xy = [(xi, yi) for xi, yi in zip(x, y) if yi is not None]
        if not xy:
            continue
        xs, ys = zip(*xy)
        plt.plot(xs, ys, label=label, linewidth=1)
    plt.xlabel("global_step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_grad_norm_buckets(cols: dict, outdir: str):
    x = _floats(cols, "global_step")
    series = {b: _floats(cols, b) for b in ("sr", "master", "fusion", "other")}
    series["total"] = _floats(cols, "grad_norm")
    _plot(x, series, "Gradient norm per module bucket", "grad L2 norm", os.path.join(outdir, "grad_norm_buckets.png"))


def plot_weight_norm_buckets(cols: dict, outdir: str):
    x = _floats(cols, "global_step")
    series = {b: _floats(cols, f"weight_{b}") for b in ("sr", "master", "fusion", "other")}
    _plot(x, series, "Weight norm per module bucket", "weight L2 norm", os.path.join(outdir, "weight_norm_buckets.png"))


def plot_loss_components(cols: dict, outdir: str):
    x = _floats(cols, "global_step")
    series = {
        "loss": _floats(cols, "loss"),
        "energy_term": _floats(cols, "energy_term"),
        "force_term": _floats(cols, "force_term"),
    }
    _plot(x, series, "Decomposed loss", "loss value", os.path.join(outdir, "loss_components.png"))


def plot_alpha_damping(cols: dict, outdir: str):
    x = _floats(cols, "global_step")
    series = {
        "alpha": _floats(cols, "alpha"), "damping_sum_min": _floats(cols, "damping_sum_min"),
        "Z_min": _floats(cols, "Z_min"),
    }
    _plot(x, series, "MasterNodeAttention Ewald-damping scalars", "value", os.path.join(outdir, "alpha_damping.png"))


def plot_g_a_norm(cols: dict, outdir: str):
    x = _floats(cols, "global_step")
    series = {"g_a_norm_max": _floats(cols, "g_a_norm_max")}
    _plot(x, series, "Master-node residual stream magnitude (max over layers)", "||g_a_out||", os.path.join(outdir, "g_a_norm.png"))


def plot_risk_scalars(cols: dict, outdir: str):
    x = _floats(cols, "global_step")
    series = {
        "ln_clamp_hits": _floats(cols, "ln_clamp_hits"),
        "safe_inv_retries": _floats(cols, "safe_inv_retries"),
        "eqln_deg_gt0_near_degenerate": _floats(cols, "eqln_deg_gt0_near_degenerate"),
        "max_force_mag": _floats(cols, "max_force_mag"),
        "max_abs_U": _floats(cols, "max_abs_U"),
    }
    _plot(x, series, "Risk scalars / event counters", "value", os.path.join(outdir, "risk_scalars.png"))


def plot_seed_radial_collapse(cols: dict, outdir: str):
    x = _floats(cols, "global_step")
    series = {
        "min_abs_radial": _floats(cols, "min_abs_radial"),
        "grad_norm": _floats(cols, "grad_norm"),
    }
    _plot(x, series, "seed_radial_net min|output| vs. grad_norm", "value", os.path.join(outdir, "seed_radial_collapse.png"))


def lagged_correlation(x: list[float | None], y: list[float | None], max_lag: int) -> dict[int, float]:
    xs = np.array([v if v is not None else np.nan for v in x], dtype=float)
    ys = np.array([v if v is not None else np.nan for v in y], dtype=float)
    n = min(len(xs), len(ys))
    xs, ys = xs[:n], ys[:n]

    out: dict[int, tuple[float, int]] = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            xv, yv = xs[: n - lag] if lag > 0 else xs, ys[lag:]
        else:
            xv, yv = xs[-lag:], ys[: n + lag]
        valid = np.isfinite(xv) & np.isfinite(yv)
        n_valid = int(valid.sum())
        if n_valid < 5 or xv[valid].std() == 0 or yv[valid].std() == 0:
            out[lag] = (float("nan"), n_valid)
            continue
        r = float(np.corrcoef(xv[valid], yv[valid])[0, 1])
        out[lag] = (r, n_valid)
    return out


def spike_indicator(values: list[float | None], percentile: float = 99.0, min_threshold: float | None = None) -> list[float | None]:
    finite = [v for v in values if v is not None and np.isfinite(v)]
    if not finite:
        return [None] * len(values)
    threshold = min_threshold if min_threshold is not None else float(np.percentile(finite, percentile))
    return [None if v is None or not np.isfinite(v) else float(v > threshold) for v in values]


def plot_lag_correlation(lag_result: dict[int, tuple[float, int]], title: str, path: str):
    lags = sorted(lag_result.keys())
    rs = [lag_result[l][0] for l in lags]
    plt.figure(figsize=(10, 4))
    plt.plot(lags, rs, marker="o", markersize=3, linewidth=1)
    plt.axhline(0.0, color="gray", linewidth=0.5)
    plt.axvline(0.0, color="gray", linewidth=0.5, linestyle="--")
    plt.xlabel("lag (rows; positive = x leads y)")
    plt.ylabel("Pearson r")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _best_lag(lag_result: dict[int, tuple[float, int]], min_samples: int = 30) -> tuple[int, float, int] | None:
    candidates = [(lag, r, n) for lag, (r, n) in lag_result.items() if np.isfinite(r) and n >= min_samples]
    if not candidates:
        return None
    return max(candidates, key=lambda t: abs(t[1]))


def analyze_correlations(cols: dict, outdir: str, max_lag: int = 20):
    alpha = _floats(cols, "alpha")
    damping = _floats(cols, "damping_sum_min")
    Z_min = _floats(cols, "Z_min")
    g_a_norm_max = _floats(cols, "g_a_norm_max")
    grad_norm = _floats(cols, "grad_norm")
    loss = _floats(cols, "loss")
    buckets = {b: _floats(cols, b) for b in ("sr", "master", "fusion", "other")}

    spikes = spike_indicator(grad_norm, percentile=99.0)
    spikes_loss = spike_indicator(loss, min_threshold=10.0)

    results: dict[str, dict[int, tuple[float, int]]] = {}
    pairs = [
        ("alpha_vs_gradnorm_spike", alpha, spikes),
        ("alpha_vs_loss_spike", alpha, spikes_loss),
        ("damping_sum_min_vs_gradnorm_spike", damping, spikes),
        ("alpha_vs_master_bucket", alpha, buckets["master"]),
        ("alpha_vs_fusion_bucket", alpha, buckets["fusion"]),
        ("alpha_vs_sr_bucket", alpha, buckets["sr"]),
        ("Z_min_vs_gradnorm_spike", Z_min, spikes),
        ("Z_min_vs_master_bucket", Z_min, buckets["master"]),
        ("Z_min_vs_fusion_bucket", Z_min, buckets["fusion"]),
        ("g_a_norm_vs_gradnorm_spike", g_a_norm_max, spikes),
        ("g_a_norm_vs_master_bucket", g_a_norm_max, buckets["master"]),
        ("g_a_norm_vs_fusion_bucket", g_a_norm_max, buckets["fusion"]),
    ]

    print("\n=== Lagged correlation analysis (Phase 0.3 / Open Question 6) ===")
    for name, x, y in pairs:
        lag_result = lagged_correlation(x, y, max_lag)
        results[name] = lag_result
        plot_lag_correlation(lag_result, name.replace("_", " "), os.path.join(outdir, f"lagcorr_{name}.png"))
        zero_r, zero_n = lag_result.get(0, (float("nan"), 0))
        best = _best_lag(lag_result)
        if best is None:
            print(f"  {name}: insufficient data at every lag (need >=30 valid pairs)")
        else:
            best_lag, best_r, best_n = best
            print(f"  {name}: r(lag=0)={zero_r:.4f} (n={zero_n})  |  strongest |r| at lag={best_lag}: "
                  f"r={best_r:.4f} (n={best_n})")

    return results


def load_provenance(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def basic_run_stats(cols: dict) -> dict:
    global_steps = [v for v in _floats(cols, "global_step") if v is not None]
    loss = [v for v in _floats(cols, "loss") if v is not None]
    grad_norm = [v for v in _floats(cols, "grad_norm") if v is not None]
    energy_term = [v for v in _floats(cols, "energy_term") if v is not None]
    force_term = [v for v in _floats(cols, "force_term") if v is not None]
    epochs = [v for v in _floats(cols, "epoch") if v is not None]
    return {
        "n_train_batch_rows": len(global_steps),
        "final_global_step": int(global_steps[-1]) if global_steps else None,
        "final_epoch": int(epochs[-1]) if epochs else None,
        "final_loss": loss[-1] if loss else None,
        "final_grad_norm": grad_norm[-1] if grad_norm else None,
        "max_grad_norm": max(grad_norm) if grad_norm else None,
        "mean_force_to_energy_term_ratio_last_100": (
            (sum(force_term[-100:]) / len(force_term[-100:])) / max(sum(energy_term[-100:]) / len(energy_term[-100:]), 1e-12)
            if len(force_term) >= 1 and len(energy_term) >= 1 else None
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=str, required=True, help="Path to the loss/grad-norm CSV")
    parser.add_argument("--provenance", type=str, default=None, help="Path to provenance.jsonl (optional)")
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--max-lag", type=int, default=20, help="Max +/- lag (in logged rows) for the correlation analysis")
    parser.add_argument("--skip-correlation", action="store_true", help="Skip the lagged-correlation analysis, plots only")
    parser.add_argument("--json-out", type=str, default=None, help="Also write basic run stats + lagged-correlation results as JSON (for the session report generator)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    cols = load_csv(args.csv)
    if not cols.get("global_step"):
        print(f"No train_batch rows found in {args.csv}")
        return

    plot_grad_norm_buckets(cols, args.outdir)
    plot_weight_norm_buckets(cols, args.outdir)
    plot_loss_components(cols, args.outdir)
    plot_alpha_damping(cols, args.outdir)
    plot_risk_scalars(cols, args.outdir)
    plot_seed_radial_collapse(cols, args.outdir)
    plot_g_a_norm(cols, args.outdir)
    print(f"Wrote plots to {args.outdir}")

    stats = basic_run_stats(cols)
    corr_results = analyze_correlations(cols, args.outdir, max_lag=args.max_lag) if not args.skip_correlation else {}

    if args.provenance and os.path.exists(args.provenance):
        rows = load_provenance(args.provenance)
        print(f"Loaded {len(rows)} provenance rows from {args.provenance}")

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump({"csv": args.csv, "run_stats": stats, "lagged_correlations": corr_results}, f, indent=2, default=str)
        print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()
