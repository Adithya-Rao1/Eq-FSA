from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

import torch

from diagnostics.explosion_inventory import inventory as run_explosion_inventory
from diagnostics.phase2_runner import run_diagnostics_script
from diagnostics.run_phase2_overnight import run_phase2_battery

_CKPT_RE = re.compile(r"checkpoint_epoch_(\d+)\.pt$")


def find_latest_checkpoint(ckpt_dir: str) -> str | None:
    candidates = glob.glob(os.path.join(ckpt_dir, "checkpoint_epoch_*.pt"))
    if not candidates:
        return None
    def _epoch(p):
        m = _CKPT_RE.search(os.path.basename(p))
        return int(m.group(1)) if m else -1
    return max(candidates, key=_epoch)


def check_gpu_free(force: bool) -> bool:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("[run_overnight] nvidia-smi not available -- skipping GPU-busy check.")
        return True

    busy = out.stdout.strip()
    if busy:
        print(f"[run_overnight] GPU appears busy:\n{busy}")
        if force:
            print("[run_overnight] --force set, proceeding anyway.")
            return True
        print("[run_overnight] Refusing to start -- stop the running job first, or pass --force to override.")
        return False
    return True


def run_phase01(
    base_ckpt: str, results_root: str, spice_loss_csv: str | None,
    hdf5_paths_override: list[str] | None, device: str | None,
) -> dict:
    phase01_root = os.path.join(results_root, "phase0_1")
    os.makedirs(phase01_root, exist_ok=True)
    results: dict = {}
    ckpt_dir = os.path.dirname(base_ckpt)

    print("\n=== Phase 0.4: explosion-dump inventory ===")
    rows = []
    try:
        dump_dir = os.path.join(ckpt_dir, "diag_dumps")
        rows = run_explosion_inventory(dump_dir)
        json_path = os.path.join(phase01_root, "0.4_explosion_inventory.json")
        with open(json_path, "w") as f:
            json.dump({"dump_dir": dump_dir, "n_dumps": len(rows), "dumps": rows}, f, indent=2, default=str)
        results["0.4_explosion_inventory"] = {"status": "OK", "n_dumps": len(rows), "json": json_path}
        print(f"Found {len(rows)} dump(s) -> {json_path}")
    except Exception as e:
        results["0.4_explosion_inventory"] = {"status": "FAILED", "error": repr(e)}
        print(f"[run_overnight] 0.4 failed: {e!r}")

    print("\n=== Phase 1.3: replay top explosion dump (if any matched a known spike) ===")
    top_match = next((r for r in rows if r.get("closest_known_spike")), None)
    if top_match:
        run_dir = os.path.join(phase01_root, "1.3_replay_batch")
        results["1.3_replay_batch"] = run_diagnostics_script(
            "diagnostics.replay_batch",
            ["--path", top_match["path"]], run_dir, "1.3_replay_batch",
            timeout_s=None, json_out_arg=None, profile_gpu=True,
        )
    else:
        results["1.3_replay_batch"] = {"status": "SKIPPED", "reason": "no explosion dump matched a known spike"}
        print("No matching dump found -- skipping replay.")

    def _ckpt_diag_script(module: str, run_id: str):
        run_dir = os.path.join(phase01_root, run_id)
        args = ["--ckpt", base_ckpt]
        if hdf5_paths_override:
            args += ["--hdf5-paths", *hdf5_paths_override]
        if device:
            args += ["--device", device]
        results[run_id] = run_diagnostics_script(module, args, run_dir, run_id, timeout_s=None, profile_gpu=True)

    print("\n=== Phase 1.4: master-branch ablation ===")
    _ckpt_diag_script("diagnostics.branch_ablation", "1.4_branch_ablation")

    print("\n=== Phase 1.5: alpha sensitivity sweep ===")
    _ckpt_diag_script("diagnostics.alpha_sweep", "1.5_alpha_sweep")

    print("\n=== Phase 1.6: real eqln-degeneracy measurement ===")
    _ckpt_diag_script("diagnostics.measure_eqln_degeneracy", "1.6_eqln_degeneracy")

    print("\n=== Phase 0.3: real lagged-correlation analysis ===")
    if spice_loss_csv and os.path.exists(spice_loss_csv):
        run_dir = os.path.join(phase01_root, "0.3_analyze_run")
        results["0.3_analyze_run"] = run_diagnostics_script(
            "diagnostics.analyze_run",
            ["--csv", spice_loss_csv, "--outdir", run_dir], run_dir, "0.3_analyze_run", timeout_s=None,
        )
    else:
        results["0.3_analyze_run"] = {"status": "SKIPPED", "reason": f"spice_loss_csv not found: {spice_loss_csv}"}
        print(f"No spice_loss.csv at {spice_loss_csv} -- skipping.")

    print("\n=== Progressive-instability stage 1a: checkpoint weight-trend archaeology ===")
    run_dir = os.path.join(phase01_root, "1a_checkpoint_weight_trend")
    results["1a_checkpoint_weight_trend"] = run_diagnostics_script(
        "diagnostics.checkpoint_weight_trend",
        ["--ckpt-dir", ckpt_dir, "--outdir", run_dir], run_dir, "1a_checkpoint_weight_trend", timeout_s=None,
    )

    print("\n=== Progressive-instability stage 1b: epoch-level trend analysis ===")
    if spice_loss_csv and os.path.exists(spice_loss_csv):
        run_dir = os.path.join(phase01_root, "1b_analyze_epoch_trends")
        results["1b_analyze_epoch_trends"] = run_diagnostics_script(
            "diagnostics.analyze_epoch_trends",
            ["--csv", spice_loss_csv, "--outdir", run_dir], run_dir, "1b_analyze_epoch_trends", timeout_s=None,
        )
    else:
        results["1b_analyze_epoch_trends"] = {"status": "SKIPPED", "reason": f"spice_loss_csv not found: {spice_loss_csv}"}
        print(f"No spice_loss.csv at {spice_loss_csv} -- skipping.")

    summary_path = os.path.join(phase01_root, "phase01_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[run_overnight] Phase 0/1 complete -> {summary_path}")
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt-dir", type=str, required=True, help="Directory of checkpoint_epoch_*.pt files from the just-stopped run")
    p.add_argument("--base-ckpt", type=str, default=None, help="Explicit checkpoint path; overrides auto-discovery from --ckpt-dir")
    p.add_argument("--results-root", type=str, required=True)
    p.add_argument("--spice-loss-csv", type=str, default=None, help="Real loss CSV from the stopped run, for Phase 0.3 (default: guessed as a sibling of --ckpt-dir, matching train_spice.py's own --csv-path/--ckpt-dir defaults)")
    p.add_argument("--hdf5-paths", type=str, nargs="+", default=None, help="Override the checkpoint's stored hdf5 paths")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--skip-phase01", action="store_true")
    p.add_argument("--skip-phase2", action="store_true")
    p.add_argument("--force", action="store_true", help="Proceed even if the GPU appears busy")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if not check_gpu_free(args.force):
        sys.exit(1)

    base_ckpt = args.base_ckpt or find_latest_checkpoint(args.ckpt_dir)
    if not base_ckpt or not os.path.exists(base_ckpt):
        print(f"[run_overnight] No checkpoint found (looked in {args.ckpt_dir}, --base-ckpt={args.base_ckpt}). Aborting.")
        sys.exit(1)
    print(f"[run_overnight] Using base checkpoint: {base_ckpt}")

    os.makedirs(args.results_root, exist_ok=True)
    t_start = time.monotonic()

    if args.dry_run:
        print("[DRY RUN] Would run Phase 0/1 battery here (skipped in dry-run mode -- it's fast and side-effecting).")
    elif not args.skip_phase01:
        spice_loss_csv = args.spice_loss_csv or os.path.join(os.path.dirname(args.ckpt_dir.rstrip("/")), "spice_loss.csv")
        run_phase01(base_ckpt, args.results_root, spice_loss_csv, args.hdf5_paths, args.device)

    phase01_elapsed = time.monotonic() - t_start
    print(f"\n[run_overnight] Phase 0/1 took {phase01_elapsed:.0f}s.")

    if not args.skip_phase2:
        ckpt = torch.load(base_ckpt, map_location="cpu", weights_only=False)
        base_config = dict(ckpt["config"])
        if args.hdf5_paths:
            base_config["hdf5_paths"] = args.hdf5_paths
        phase2_results_root = os.path.join(args.results_root, "phase2")
        run_phase2_battery(base_config, phase2_results_root, dry_run=args.dry_run, device=args.device)

    if not args.dry_run:
        from diagnostics.generate_session_report import generate_report
        generate_report(args.results_root)


if __name__ == "__main__":
    sys.exit(main())