from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_CKPT_RE = re.compile(r"checkpoint_epoch_(\d+)\.pt$")
_LAYER_RE = re.compile(r"energy_model\.layers\.(\d+)\.")


def _bucket_name(name: str) -> str:
    if "sr_attn" in name or "radial_V_sr" in name or "W_V_self" in name:
        return "sr"
    if "master_attn" in name or "W_master_self" in name or "master_pool" in name or "master_expand" in name:
        return "master"
    if "master_fusion" in name:
        return "fusion"
    return "other"


def find_checkpoints(ckpt_dir: str) -> list[tuple[int, str]]:
    paths = glob.glob(os.path.join(ckpt_dir, "checkpoint_epoch_*.pt"))
    out = []
    for p in paths:
        m = _CKPT_RE.search(os.path.basename(p))
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def _spectral_norm(w: torch.Tensor) -> float:
    if w.dim() != 2:
        return float("nan")
    return float(torch.linalg.matrix_norm(w.float(), ord=2))


def extract_epoch_stats(state_dict: dict) -> dict:
    layer_idxs = sorted({int(m.group(1)) for k in state_dict for m in [_LAYER_RE.search(k)] if m})
    stats: dict = {"n_layers": len(layer_idxs), "layers": {}}

    for i in layer_idxs:
        prefix = f"energy_model.layers.{i}."
        layer_stats: dict = {}

        w_master_self = state_dict.get(prefix + "W_master_self.weight.weight")
        if w_master_self is not None:
            layer_stats["W_master_self_frobenius"] = float(w_master_self.float().norm())
            layer_stats["W_master_self_spectral"] = _spectral_norm(w_master_self)

        alpha_raw = state_dict.get(prefix + "master_attn.alpha")
        if alpha_raw is not None:
            alpha_raw_f = float(alpha_raw)
            layer_stats["alpha_raw"] = alpha_raw_f
            layer_stats["alpha_post_softplus"] = float(
                torch.nn.functional.softplus(torch.tensor(alpha_raw_f)) + 1e-4
            )

        for gate_name in ("w_q_norm", "w_k_norm"):
            gamma = state_dict.get(prefix + f"master_attn.{gate_name}.gamma")
            beta = state_dict.get(prefix + f"master_attn.{gate_name}.beta")
            if gamma is not None:
                layer_stats[f"{gate_name}_gamma_norm"] = float(gamma.float().norm())
            if beta is not None:
                layer_stats[f"{gate_name}_beta_norm"] = float(beta.float().norm())

        for qkv in ("W_Q", "W_K", "W_V0"):
            w = state_dict.get(prefix + f"master_attn.{qkv}.weight.weight")
            if w is not None:
                layer_stats[f"master_attn_{qkv}_spectral"] = _spectral_norm(w)

        stats["layers"][i] = layer_stats

    fusion_stats: dict = {}
    for mod_name in ("W_Q", "readout_K", "readout_V"):
        for deg in (0, 1, 2):
            w = state_dict.get(f"energy_model.master_fusion.{mod_name}.{deg}.weight.weight")
            if w is not None:
                fusion_stats[f"{mod_name}_{deg}_spectral"] = _spectral_norm(w)
    stats["master_fusion"] = fusion_stats

    bucket_sums = {"sr": 0.0, "master": 0.0, "fusion": 0.0, "other": 0.0}
    for name, p in state_dict.items():
        if not torch.is_tensor(p) or not torch.is_floating_point(p):
            continue
        bucket_sums[_bucket_name(name)] += float((p.float() ** 2).sum())
    stats["weight_norm_buckets"] = {k: v ** 0.5 for k, v in bucket_sums.items()}

    return stats


def build_trend(ckpt_dir: str) -> dict:
    checkpoints = find_checkpoints(ckpt_dir)
    if not checkpoints:
        return {"ckpt_dir": ckpt_dir, "epochs": [], "error": "no checkpoint_epoch_*.pt files found"}

    per_epoch = {}
    for epoch, path in checkpoints:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        per_epoch[epoch] = extract_epoch_stats(state_dict)
        print(f"[checkpoint_weight_trend] epoch {epoch}: {path}")

    return {"ckpt_dir": ckpt_dir, "epochs": sorted(per_epoch.keys()), "per_epoch": per_epoch}


def _series(trend: dict, extractor) -> tuple[list[int], list[float]]:
    epochs = trend["epochs"]
    xs, ys = [], []
    for e in epochs:
        v = extractor(trend["per_epoch"][e])
        if v is not None:
            xs.append(e)
            ys.append(v)
    return xs, ys


def plot_trend(trend: dict, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    n_layers = max((s["n_layers"] for s in trend["per_epoch"].values()), default=0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    for i in range(n_layers):
        xs, ys = _series(trend, lambda s, i=i: s["layers"].get(i, {}).get("W_master_self_spectral"))
        if xs:
            ax.plot(xs, ys, marker="o", markersize=3, label=f"layer{i}")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("W_master_self spectral norm")
    ax.set_title("Candidate #1: W_master_self spectral norm vs. epoch")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for i in range(n_layers):
        xs, ys = _series(trend, lambda s, i=i: s["layers"].get(i, {}).get("alpha_post_softplus"))
        if xs:
            ax.plot(xs, ys, marker="o", markersize=3, label=f"layer{i}")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("alpha (post-softplus)")
    ax.set_title("alpha vs. epoch")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    for i in range(n_layers):
        xs, ys = _series(trend, lambda s, i=i: s["layers"].get(i, {}).get("w_q_norm_gamma_norm"))
        if xs:
            ax.plot(xs, ys, marker="o", markersize=3, label=f"layer{i} w_q_norm.gamma")
        xs, ys = _series(trend, lambda s, i=i: s["layers"].get(i, {}).get("w_k_norm_gamma_norm"))
        if xs:
            ax.plot(xs, ys, marker="s", markersize=3, linestyle="--", label=f"layer{i} w_k_norm.gamma")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("gamma norm")
    ax.set_title("Candidate #2 context: w_q_norm/w_k_norm gamma vs. epoch")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    for bucket in ("sr", "master", "fusion", "other"):
        xs, ys = _series(trend, lambda s, bucket=bucket: s["weight_norm_buckets"].get(bucket))
        if xs:
            ax.plot(xs, ys, marker="o", markersize=3, label=bucket)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("weight norm (bucket aggregate)")
    ax.set_title("Weight-norm buckets vs. epoch (cross-check vs. CSV's weight_* columns)")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(outdir, "checkpoint_weight_trend.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[checkpoint_weight_trend] wrote {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt-dir", type=str, required=True, help="Directory of checkpoint_epoch_*.pt files from the real training run")
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--json-out", type=str, default=None, help="Defaults to <outdir>/checkpoint_weight_trend.json")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    trend = build_trend(args.ckpt_dir)

    if trend.get("error"):
        print(f"[checkpoint_weight_trend] {trend['error']}")
    else:
        plot_trend(trend, args.outdir)

    json_out = args.json_out or os.path.join(args.outdir, "checkpoint_weight_trend.json")
    with open(json_out, "w") as f:
        json.dump(trend, f, indent=2, default=str)
    print(f"[checkpoint_weight_trend] wrote {json_out}")


if __name__ == "__main__":
    sys.exit(main())
