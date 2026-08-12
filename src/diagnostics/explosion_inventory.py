from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime

import torch

_STEP_RE = re.compile(r"explosion_step(\d+)\.pt$")
_VALUE_RE = re.compile(r"value=([0-9.eE+\-]+)")

KNOWN_SPIKES = {5: 3.00e7, 7: 6.95e6}
REL_TOL = 0.15


def _extract_max_reason_value(reasons: dict) -> float | None:
    vals = []
    for reason in (reasons or {}).values():
        m = _VALUE_RE.search(reason)
        if m:
            try:
                vals.append(float(m.group(1)))
            except ValueError:
                pass
    return max(vals) if vals else None


def _closest_known_spike(value: float | None) -> tuple[int, float] | None:
    if value is None:
        return None
    best = None
    for epoch, known in KNOWN_SPIKES.items():
        rel_err = abs(value - known) / known
        if rel_err <= REL_TOL and (best is None or rel_err < best[1]):
            best = (epoch, rel_err)
    return best


def inventory(dump_dir: str) -> list[dict]:
    paths = sorted(
        glob.glob(os.path.join(dump_dir, "explosion_step*.pt")),
        key=lambda p: int(_STEP_RE.search(os.path.basename(p)).group(1)) if _STEP_RE.search(os.path.basename(p)) else -1,
    )
    rows = []
    for path in paths:
        stat = os.stat(path)
        header = torch.load(path, map_location="cpu", weights_only=False)
        reasons = header.get("reasons", {})
        max_val = _extract_max_reason_value(reasons)
        rows.append({
            "path": path,
            "global_step": header.get("global_step"),
            "epoch": header.get("epoch"),
            "batch_idx": header.get("batch_idx"),
            "reasons": reasons,
            "max_reason_value": max_val,
            "closest_known_spike": _closest_known_spike(max_val),
            "n_smiles": len(header.get("batch", {}).get("smiles", []) or []),
            "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
            "size_mb": stat.st_size / (1024 * 1024),
            "keys": sorted(header.keys()),
        })
    return rows


def print_report(rows: list[dict]):
    if not rows:
        print("No explosion_step*.pt dumps found. Either training never triggered the drift "
              "detector (DriftDetector.update_and_check), or --diagnostics wasn't on for this run.")
        return

    print(f"Found {len(rows)} explosion dump(s):\n")
    header = f"{'file':40s} {'step':>8s} {'epoch':>6s} {'batch':>6s} {'size(MB)':>9s} {'mtime':>20s}  max_reason_value"
    print(header)
    print("-" * len(header))
    for r in rows:
        fname = os.path.basename(r["path"])
        max_val_str = f"{r['max_reason_value']:.4g}" if r["max_reason_value"] is not None else "-"
        print(
            f"{fname:40s} {str(r['global_step']):>8s} {str(r['epoch']):>6s} {str(r['batch_idx']):>6s} "
            f"{r['size_mb']:>9.2f} {r['mtime']:>20s}  {max_val_str}"
        )
        for bucket, reason in r["reasons"].items():
            print(f"    {bucket}: {reason}")
        if r["closest_known_spike"] is not None:
            epoch, rel_err = r["closest_known_spike"]
            print(
                f"    >> matches the original review's epoch-{epoch} spike "
                f"(known={KNOWN_SPIKES[epoch]:.3g}, this dump={r['max_reason_value']:.3g}, "
                f"rel_err={rel_err:.1%}) -- prioritize this one for Phase 1.3 replay."
            )
        print(f"    replay: python -m diagnostics.replay_batch --path {r['path']}")

    unmatched_known = [
        epoch for epoch in KNOWN_SPIKES
        if not any(r["closest_known_spike"] and r["closest_known_spike"][0] == epoch for r in rows)
    ]
    if unmatched_known:
        print(
            f"\nNote: no dump matched the review's epoch {unmatched_known} spike(s) within "
            f"{REL_TOL:.0%} relative tolerance. Either the drift detector didn't trigger a dump "
            f"for that spike (e.g. it was caught by the absolute --diag-explode-threshold on a "
            f"different bucket than the ones summarized in the review, or the ring-buffer "
            f"checkpoint cadence missed it), or the max_reason_value regex parse missed the "
            f"relevant number -- check that dump's `reasons` dict by hand."
        )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--diag-dump-dir", type=str, default=None,
                    help="Directory containing explosion_step*.pt dumps. If omitted, derived as <ckpt-dir>/diag_dumps")
    p.add_argument("--ckpt-dir", type=str, default="/home/ubuntu/metis-v1-storage/spice_stage1",
                    help="Used to derive --diag-dump-dir when it isn't passed explicitly (matches train_spice.py's default)")
    p.add_argument("--json-out", type=str, default=None, help="Session report generator JSON for phase 0, 1, 2, 3 experiments")
    args = p.parse_args()

    dump_dir = args.diag_dump_dir or os.path.join(args.ckpt_dir, "diag_dumps")
    print(f"Inventorying: {dump_dir}\n")
    rows = inventory(dump_dir)
    print_report(rows)

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        serializable = [{k: v for k, v in r.items()} for r in rows]
        with open(args.json_out, "w") as f:
            json.dump({"dump_dir": dump_dir, "n_dumps": len(rows), "dumps": serializable}, f, indent=2, default=str)
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    sys.exit(main())