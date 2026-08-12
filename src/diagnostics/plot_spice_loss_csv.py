from __future__ import annotations

import argparse
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diagnostics.analyze_run import load_csv, _floats

_SURFACE = "#fcfcfb"
_INK_PRIMARY = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_INK_MUTED = "#898781"
_GRIDLINE = "#e1e0d9"
_AXIS = "#c3c2b7"
_ACCENT = "#2a78d6"

_EXCLUDE_COLUMNS = {"timestamp", "phase", "global_step"}  


def _pretty_title(col: str) -> str:
    return col.replace("_", " ").strip()


def _should_use_log_scale(values: list) -> bool:
    positive = [v for v in values if v is not None and v > 0]
    if len(positive) < 2:
        return False
    lo, hi = min(positive), max(positive)
    return lo > 0 and (hi / lo) > 100


def _epoch_boundaries(epochs: list, steps: list) -> list[tuple[int, float]]:
    boundaries = []
    seen = set()
    for e, s in zip(epochs, steps):
        if e is None or s is None:
            continue
        e_int = int(e)
        if e_int not in seen:
            seen.add(e_int)
            boundaries.append((e_int, s))
    return sorted(boundaries, key=lambda t: t[1])


def _style_axes(ax):
    ax.set_facecolor(_SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(_AXIS)
    ax.tick_params(colors=_INK_MUTED, labelsize=8)
    ax.grid(True, color=_GRIDLINE, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def _draw_column(ax, x: list, y: list, col: str, epoch_bounds: list[tuple[int, float]]):
    _style_axes(ax)
    ax.set_title(_pretty_title(col), color=_INK_PRIMARY, fontsize=10, loc="left")

    xy = [(xi, yi) for xi, yi in zip(x, y) if xi is not None and yi is not None]
    if not xy:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", color=_INK_MUTED, transform=ax.transAxes)
        return

    xs, ys = zip(*xy)

    for epoch_idx, step in epoch_bounds:
        if epoch_idx > 0 and xs[0] <= step <= xs[-1]:
            ax.axvline(step, color=_GRIDLINE, linewidth=0.8, linestyle="--", zorder=0)

    marker = "o" if len(xs) < 150 else None
    ax.plot(xs, ys, color=_ACCENT, linewidth=1.3, marker=marker, markersize=3, alpha=0.95)

    if _should_use_log_scale(list(ys)):
        ax.set_yscale("log")

    ax.set_xlabel("global step", color=_INK_SECONDARY, fontsize=8)


def _numeric_columns(cols: dict) -> list[str]:
    numeric_columns = []
    for col in cols.keys():
        if col in _EXCLUDE_COLUMNS:
            continue
        values = _floats(cols, col)
        if not values or all(v is None for v in values):
            continue
        numeric_columns.append(col)
    return numeric_columns


def plot_all_columns(cols: dict, outdir: str) -> list[str]:
    os.makedirs(outdir, exist_ok=True)
    x = _floats(cols, "global_step")
    epochs = _floats(cols, "epoch")
    epoch_bounds = _epoch_boundaries(epochs, x)
    numeric_columns = _numeric_columns(cols)

    written = []
    for col in numeric_columns:
        y = _floats(cols, col)
        fig, ax = plt.subplots(figsize=(9, 4), facecolor=_SURFACE)
        _draw_column(ax, x, y, col, epoch_bounds)
        fig.tight_layout()
        path = os.path.join(outdir, f"{col}.png")
        fig.savefig(path, dpi=150, facecolor=_SURFACE)
        plt.close(fig)
        written.append(path)
        print(f"[plot_spice_loss_csv] wrote {path}")

    overview_path = _plot_overview(cols, numeric_columns, x, epoch_bounds, outdir)
    if overview_path:
        written.append(overview_path)

    return written


def _plot_overview(cols: dict, numeric_columns: list[str], x: list, epoch_bounds, outdir: str) -> str | None:
    n = len(numeric_columns)
    if n == 0:
        return None
    n_cols = 4
    n_rows = math.ceil(n / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.0 * n_rows), facecolor=_SURFACE)
    axes = axes.flatten() if n > 1 else [axes]

    for i, col in enumerate(numeric_columns):
        y = _floats(cols, col)
        _draw_column(axes[i], x, y, col, epoch_bounds)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("SPICE training run -- all diagnostics columns vs. global step", color=_INK_PRIMARY, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = os.path.join(outdir, "overview.png")
    fig.savefig(path, dpi=150, facecolor=_SURFACE)
    plt.close(fig)
    print(f"[plot_spice_loss_csv] wrote {path}")
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--outdir", type=str, required=True)
    args = p.parse_args()

    cols = load_csv(args.csv)
    if not cols.get("global_step"):
        print(f"No train_batch rows found in {args.csv}")
        return

    written = plot_all_columns(cols, args.outdir)
    print(f"\n[plot_spice_loss_csv] wrote {len(written)} files to {args.outdir}")


if __name__ == "__main__":
    sys.exit(main())
