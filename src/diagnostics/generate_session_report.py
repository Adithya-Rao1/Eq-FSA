from __future__ import annotations

import argparse
import json
import os
import sys


def _load_json(path: str) -> dict | None:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def _gpu_util_line(stage_meta: dict) -> str | None:
    gpu = stage_meta.get("gpu_utilization")
    if not gpu or gpu.get("error"):
        return None
    return (f"- GPU utilization: mean={gpu.get('mean_utilization_pct'):.1f}%  "
            f"median={gpu.get('median_utilization_pct'):.1f}%  "
            f"idle_fraction(<10%)={gpu.get('fraction_samples_idle'):.1%}  "
            f"max_mem={gpu.get('max_memory_used_mb'):.0f}MB  (n={gpu.get('n_samples')} samples)")


def _status_of(meta: dict) -> str:
    if not meta:
        return "MISSING"
    if "status" in meta and meta["status"] in ("SKIPPED", "SKIPPED_OUT_OF_TIME", "DRY_RUN", "FAILED"):
        return meta["status"]
    if meta.get("returncode") == 0:
        return "OK"
    if meta.get("timed_out"):
        return "TIMEOUT"
    if "returncode" in meta:
        return f"FAILED(rc={meta.get('returncode')})"
    return "UNKNOWN"


def _phase01_section(results_root: str) -> dict:
    path = os.path.join(results_root, "phase0_1", "phase01_summary.json")
    summary = _load_json(path)
    if summary is None:
        return {"available": False, "path": path}
    return {"available": True, "path": path, "stages": summary}


def _phase2_section(results_root: str) -> dict:
    path = os.path.join(results_root, "phase2", "phase2_battery_summary.json")
    summary = _load_json(path)
    if summary is None:
        return {"available": False, "path": path}
    return {"available": True, "path": path, **summary}


def generate_report(results_root: str, out_json: str | None = None, out_md: str | None = None) -> dict:
    out_json = out_json or os.path.join(results_root, "session_report.json")
    out_md = out_md or os.path.join(results_root, "session_report.md")

    session = {
        "results_root": results_root,
        "phase0_1": _phase01_section(results_root),
        "phase2": _phase2_section(results_root),
    }

    with open(out_json, "w") as f:
        json.dump(session, f, indent=2, default=str)

    md = _render_markdown(session)
    with open(out_md, "w") as f:
        f.write(md)

    print(f"[generate_session_report] Wrote {out_json}")
    print(f"[generate_session_report] Wrote {out_md}")
    return session


def _render_markdown(session: dict) -> str:
    lines = ["# SPICE stage-1 diagnostics -- overnight session report", ""]

    lines += _render_phase01_markdown(session["phase0_1"])
    lines += [""]
    lines += _render_phase2_markdown(session["phase2"])

    return "\n".join(lines)


def _render_phase01_markdown(section: dict) -> list[str]:
    lines = ["## Phase 0/1"]
    if not section.get("available"):
        lines.append(f"\n_Not run or summary missing ({section.get('path')})._")
        return lines

    stages = section["stages"]

    s = stages.get("0.4_explosion_inventory", {})
    lines.append(f"\n### 0.4 -- Explosion-dump inventory: {_status_of(s)}")
    if s.get("n_dumps") is not None:
        lines.append(f"- {s['n_dumps']} dump(s) found. Full inventory: `{s.get('json')}`")

    s = stages.get("1.3_replay_batch", {})
    lines.append(f"\n### 1.3 -- Replay top explosion dump: {_status_of(s)}")
    if s.get("stdout_log"):
        lines.append(f"- Qualitative report (no structured output): `{s['stdout_log']}`")
    elif s.get("reason"):
        lines.append(f"- {s['reason']}")
    gpu_line = _gpu_util_line(s)
    if gpu_line:
        lines.append(gpu_line)

    s = stages.get("1.4_branch_ablation", {})
    lines.append(f"\n### 1.4 -- Master-node branch ablation: {_status_of(s)}")
    result = s.get("result")
    if result:
        base, abl = result.get("baseline", {}), result.get("ablated", {})
        lines.append(f"- energy MAE: baseline={base.get('energy_mae_eV')} eV, ablated={abl.get('energy_mae_eV')} eV "
                      f"(rel_change={result.get('energy_mae_rel_change'):+.2%})" if result.get("energy_mae_rel_change") is not None else "- (missing energy MAE numbers)")
        lines.append(f"- force MAE: baseline={base.get('force_mae_eV_per_A')} eV/A, ablated={abl.get('force_mae_eV_per_A')} eV/A "
                      f"(rel_change={result.get('force_mae_rel_change'):+.2%})" if result.get("force_mae_rel_change") is not None else "- (missing force MAE numbers)")
    elif s.get("stdout_log"):
        lines.append(f"- No JSON result captured; see `{s['stdout_log']}`")
    gpu_line = _gpu_util_line(s)
    if gpu_line:
        lines.append(gpu_line)

    s = stages.get("1.5_alpha_sweep", {})
    lines.append(f"\n### 1.5 -- Alpha sensitivity sweep: {_status_of(s)}")
    result = s.get("result")
    if result:
        lines.append(f"- learned per-layer alpha (baseline): {result.get('learned_alphas')}")
        for label, m in (result.get("results") or {}).items():
            lines.append(f"  - {label}: energy_MAE={m.get('energy_mae_eV')} eV, force_MAE={m.get('force_mae_eV_per_A')} eV/A")
    elif s.get("stdout_log"):
        lines.append(f"- No JSON result captured; see `{s['stdout_log']}`")
    gpu_line = _gpu_util_line(s)
    if gpu_line:
        lines.append(gpu_line)

    s = stages.get("1.6_eqln_degeneracy", {})
    lines.append(f"\n### 1.6 -- Real eqln-degeneracy measurement: {_status_of(s)}")
    result = s.get("result")
    if result:
        overall = result.get("overall", {})
        lines.append(f"- would_trip_fraction_of_calls={overall.get('would_trip_fraction_of_calls')}  "
                      f"real elementwise_fraction={overall.get('elementwise_fraction_of_real_atom_channels')}  "
                      f"real atom_fraction={overall.get('fraction_of_real_atoms_with_ge1_near_degenerate_channel')}")
        lines.append("- (If would_trip stays high while the real fractions are near 0, it's confirmed as the same "
                      "missing-mask padding artifact as min_abs_radial. If the real fractions are non-trivial too, "
                      "it's a genuine training-dynamics signal.)")
    elif s.get("stdout_log"):
        lines.append(f"- No JSON result captured; see `{s['stdout_log']}`")
    gpu_line = _gpu_util_line(s)
    if gpu_line:
        lines.append(gpu_line)

    s = stages.get("0.3_analyze_run", {})
    lines.append(f"\n### 0.3 -- Real lagged-correlation analysis: {_status_of(s)}")
    result = s.get("result")
    if result:
        rs = result.get("run_stats", {})
        lines.append(f"- run reached global_step={rs.get('final_global_step')}, epoch={rs.get('final_epoch')}, "
                      f"final_loss={rs.get('final_loss')}, max_grad_norm={rs.get('max_grad_norm')}")
        for name, lag_result in (result.get("lagged_correlations") or {}).items():
            zero = lag_result.get("0") or lag_result.get(0)
            if zero:
                lines.append(f"  - {name}: r(lag=0)={zero[0] if isinstance(zero, list) else zero}")
    elif s.get("reason"):
        lines.append(f"- {s['reason']}")

    s = stages.get("1a_checkpoint_weight_trend", {})
    lines.append(f"\n### Progressive-instability 1a -- checkpoint weight-trend archaeology: {_status_of(s)}")
    result = s.get("result")
    if result and not result.get("error"):
        n_epochs = len(result.get("epochs", []))
        lines.append(f"- {n_epochs} checkpoint(s) analyzed. Candidate #1 test: is `W_master_self`'s spectral "
                      f"norm growing across epochs? See plot/JSON: `{s.get('stdout_log')}` dir.")
        per_epoch = result.get("per_epoch") or {}
        epochs = sorted(int(e) for e in per_epoch.keys())
        if len(epochs) >= 2:
            first_layers = per_epoch[str(epochs[0])]["layers"]
            last_layers = per_epoch[str(epochs[-1])]["layers"]
            for layer_idx in sorted(first_layers.keys(), key=int):
                first_spec = first_layers[layer_idx].get("W_master_self_spectral")
                last_spec = last_layers.get(layer_idx, {}).get("W_master_self_spectral")
                if first_spec is not None and last_spec is not None:
                    lines.append(f"  - layer{layer_idx} W_master_self spectral norm: "
                                  f"epoch {epochs[0]}={first_spec:.4g} -> epoch {epochs[-1]}={last_spec:.4g}")
    elif result and result.get("error"):
        lines.append(f"- {result['error']}")
    elif s.get("stdout_log"):
        lines.append(f"- No JSON result captured; see `{s['stdout_log']}`")

    s = stages.get("1b_analyze_epoch_trends", {})
    lines.append(f"\n### Progressive-instability 1b -- epoch-level trend analysis: {_status_of(s)}")
    result = s.get("result")
    if result and not result.get("error"):
        for key, t in (result.get("trends") or {}).items():
            if t.get("slope") is not None:
                lines.append(f"  - {key}: slope={t['slope']:+.4g}/epoch, r={t.get('pearson_r'):+.3f}, "
                              f"{t.get('first_value'):.4g} -> {t.get('last_value'):.4g} ({t.get('n_epochs')} epochs)")
        for name, c in (result.get("correlations") or {}).items():
            if c.get("pearson_r") is not None:
                lines.append(f"  - correlation {name}: r={c['pearson_r']:.3f} (n={c.get('n_epochs')} epochs)")
    elif result and result.get("error"):
        lines.append(f"- {result['error']}")
    elif s.get("reason"):
        lines.append(f"- {s['reason']}")

    return lines


def _render_phase2_markdown(section: dict) -> list[str]:
    lines = ["## Phase 2"]
    if not section.get("available"):
        lines.append(f"\n_Not run or summary missing ({section.get('path')})._")
        return lines

    lines.append(f"\nNo wall-clock budget -- every item runs its fixed max_steps to completion. "
                  f"batches/epoch estimate: {section.get('batches_per_epoch')}")

    for item_id, item in (section.get("items") or {}).items():
        lines.append(f"\n### {item_id}")
        if "description" in item:
            lines.append(f"_{item['description']}_")
        if item.get("status") in ("SKIPPED_OUT_OF_TIME", "DRY_RUN"):
            lines.append(f"\nStatus: {item['status']}")
            continue

        variants = item.get("variants")
        if variants is None:
            lines.append(f"\nStatus: {_status_of(item)}")
            result = item.get("result")
            if result and result.get("best"):
                lines.append(f"- best config: `{result['best'].get('variant')}` -> {result['best'].get('samples_per_sec'):.1f} samples/sec")
            elif item.get("stdout_log"):
                lines.append(f"- see `{item['stdout_log']}`")
            continue

        lines.append("")
        lines.append("| variant | status | max_steps | n_rows | last_loss | mean_grad_norm | max_grad_norm | n_grad>threshold | sec/batch | force/energy ratio | min Z_min | max g_a_norm | mean GPU% | idle frac |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for vid, vmeta in variants.items():
            status = _status_of(vmeta)
            if status in ("SKIPPED_OUT_OF_TIME", "DRY_RUN"):
                lines.append(f"| {vid} | {status} | {vmeta.get('max_steps', '-')} | - | - | - | - | - | - | - | - | - | - | - |")
                continue
            metrics = vmeta.get("metrics", {}) or {}
            gpu = vmeta.get("gpu_utilization") or {}
            lines.append(
                f"| {vid} | {status} | {(vmeta.get('cli_overrides') or {}).get('max_steps', '-')} | "
                f"{metrics.get('n_train_batch_rows', '-')} | {metrics.get('last_loss', '-')} | "
                f"{metrics.get('mean_grad_norm', '-')} | {metrics.get('max_grad_norm', '-')} | "
                f"{metrics.get('n_grad_norm_over_threshold', '-')} | {metrics.get('seconds_per_batch', '-')} | "
                f"{metrics.get('force_to_energy_ratio', '-')} | {metrics.get('min_Z_min', '-')} | "
                f"{metrics.get('max_g_a_norm_max', '-')} | {gpu.get('mean_utilization_pct', '-')} | "
                f"{gpu.get('fraction_samples_idle', '-')} |"
            )

    return lines


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", type=str, required=True)
    p.add_argument("--json-out", type=str, default=None)
    p.add_argument("--md-out", type=str, default=None)
    args = p.parse_args()
    generate_report(args.results_root, args.json_out, args.md_out)


if __name__ == "__main__":
    sys.exit(main())