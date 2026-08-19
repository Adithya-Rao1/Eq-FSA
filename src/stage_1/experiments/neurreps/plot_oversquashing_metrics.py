from __future__ import annotations

import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SURFACE = "#fcfcfb"
_INK_PRIMARY = "#0b0b0b"
_INK_MUTED = "#898781"
_GRIDLINE = "#e1e0d9"
_AXIS = "#c3c2b7"

_VARIANT_STYLE = {
    "egnn": {"color": "#c0392b", "marker": "o", "label": "EGNN"},
    "eqfsa_full": {"color": "#2a78d6", "marker": "s", "label": "Eq-FSA (full)"},
    "eqfsa_invariant_only": {"color": "#7c8b60", "marker": "^", "label": "Eq-FSA (invariant-only)"},
}

_METRICS = [
    ("val_error", "Classification error (1 - accuracy)"),
    ("jacobian_sensitivity", "Source->target Jacobian sensitivity"),
    ("mi_lower_bound", "Mutual-information lower bound (nats)"),
]


def _style_axes(ax):
    ax.set_facecolor(_SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(_AXIS)
    ax.tick_params(colors=_INK_MUTED, labelsize=8)
    ax.grid(True, color=_GRIDLINE, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def _load_rows(csv_path: str) -> list[dict]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def plot_oversquashing_metrics(csv_path: str, outdir: str) -> None:
    rows = _load_rows(csv_path)
    topologies = sorted({row["topology"] for row in rows})

    for metric_key, metric_title in _METRICS:
        fig, axes = plt.subplots(
            1, len(topologies), figsize=(5.2 * len(topologies), 4.2), squeeze=False,
        )
        for ax, topology in zip(axes[0], topologies):
            _style_axes(ax)
            ax.set_title(topology, color=_INK_PRIMARY, fontsize=10, loc="left")
            ax.set_xlabel("realized r (source->target graph distance)", color=_INK_MUTED, fontsize=8)
            ax.set_ylabel(metric_title, color=_INK_MUTED, fontsize=8)

            by_variant = defaultdict(list)
            for row in rows:
                if row["topology"] != topology:
                    continue
                by_variant[row["variant"]].append((int(row["r"]), float(row[metric_key])))

            for variant, points in by_variant.items():
                points.sort(key=lambda p: p[0])
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                style = _VARIANT_STYLE.get(variant, {"color": "#888888", "marker": "x", "label": variant})
                ax.plot(
                    xs, ys, color=style["color"], marker=style["marker"], markersize=5,
                    linewidth=1.6, label=style["label"],
                )
            ax.legend(fontsize=7, frameon=False)

        fig.patch.set_facecolor(_SURFACE)
        fig.tight_layout()
        out_path = os.path.join(outdir, f"{metric_key}_vs_r.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    plot_oversquashing_metrics(args.csv_path, args.outdir)
