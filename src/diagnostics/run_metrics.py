from __future__ import annotations

import csv
import os
from datetime import datetime


def load_train_batch_rows(csv_path: str) -> list[dict]:
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("phase") == "train_batch":
                rows.append(row)
    return rows


def _floats(rows: list[dict], key: str) -> list[float]:
    out = []
    for r in rows:
        v = r.get(key)
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            continue
    return out


def _mean(vals: list[float]) -> float | None:
    return sum(vals) / len(vals) if vals else None


def extract_run_metrics(csv_path: str, explode_threshold: float = 50.0) -> dict:
    if not os.path.exists(csv_path):
        return {"error": f"csv not found: {csv_path}"}
    rows = load_train_batch_rows(csv_path)
    if not rows:
        return {"error": "no train_batch rows logged", "n_train_batch_rows": 0}

    timestamps = [r.get("timestamp") for r in rows]
    loss = _floats(rows, "loss")
    grad_norm = _floats(rows, "grad_norm")
    energy_term = _floats(rows, "energy_term")
    force_term = _floats(rows, "force_term")
    global_steps = _floats(rows, "global_step")

    seconds_per_batch = None
    if len(global_steps) >= 2 and timestamps[0] and timestamps[-1]:
        step_span = global_steps[-1] - global_steps[0]
        if step_span > 0:
            try:
                t0 = datetime.fromisoformat(timestamps[0])
                t1 = datetime.fromisoformat(timestamps[-1])
                seconds_per_batch = (t1 - t0).total_seconds() / step_span
            except ValueError:
                pass

    force_energy_ratio = None
    mean_force_term, mean_energy_term = _mean(force_term), _mean(energy_term)
    if mean_force_term is not None and mean_energy_term is not None and mean_energy_term > 1e-12:
        force_energy_ratio = mean_force_term / mean_energy_term

    Z_min = _floats(rows, "Z_min")
    g_a_norm_max = _floats(rows, "g_a_norm_max")

    return {
        "n_train_batch_rows": len(rows),
        "first_global_step": int(global_steps[0]) if global_steps else None,
        "last_global_step": int(global_steps[-1]) if global_steps else None,
        "seconds_per_batch": seconds_per_batch,
        "first_loss": loss[0] if loss else None,
        "last_loss": loss[-1] if loss else None,
        "mean_loss": _mean(loss),
        "max_grad_norm": max(grad_norm) if grad_norm else None,
        "mean_grad_norm": _mean(grad_norm),
        "n_grad_norm_over_threshold": sum(1 for g in grad_norm if g > explode_threshold),
        "n_grad_norm_rows": len(grad_norm),
        "mean_energy_term": mean_energy_term,
        "mean_force_term": mean_force_term,
        "force_to_energy_ratio": force_energy_ratio,
        "first_Z_min": Z_min[0] if Z_min else None,
        "min_Z_min": min(Z_min) if Z_min else None,
        "first_g_a_norm_max": g_a_norm_max[0] if g_a_norm_max else None,
        "max_g_a_norm_max": max(g_a_norm_max) if g_a_norm_max else None,
    }