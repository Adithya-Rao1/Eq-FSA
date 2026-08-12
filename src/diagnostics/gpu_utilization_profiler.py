from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

_QUERY_FIELDS = ["utilization.gpu", "utilization.memory", "memory.used", "memory.total"]
_CSV_FIELDS = ["timestamp", "utilization_gpu_pct", "utilization_memory_pct", "memory_used_mb", "memory_total_mb"]


def sample_once(gpu_index: int | None = None) -> dict | None:
    query = ",".join(_QUERY_FIELDS)
    cmd = ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"]
    if gpu_index is not None:
        cmd += ["-i", str(gpu_index)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None

    line = out.stdout.strip().splitlines()[0]
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != len(_QUERY_FIELDS):
        return None
    try:
        values = {k: float(v) for k, v in zip(_QUERY_FIELDS, parts)}
    except ValueError:
        return None

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "utilization_gpu_pct": values["utilization.gpu"],
        "utilization_memory_pct": values["utilization.memory"],
        "memory_used_mb": values["memory.used"],
        "memory_total_mb": values["memory.total"],
    }


def run(csv_path: str, interval_s: float, duration_s: float | None, gpu_index: int | None) -> tuple[int, int]:
    n_samples, n_failed = 0, 0
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        f.flush()
        t_start = time.monotonic()
        while duration_s is None or (time.monotonic() - t_start) < duration_s:
            sample = sample_once(gpu_index)
            if sample is not None:
                writer.writerow(sample)
                f.flush()
                n_samples += 1
            else:
                n_failed += 1
            time.sleep(interval_s)
    return n_samples, n_failed


def summarize(csv_path: str, idle_threshold_pct: float = 10.0) -> dict:
    utils, mem_used = [], []
    try:
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    utils.append(float(row["utilization_gpu_pct"]))
                    mem_used.append(float(row["memory_used_mb"]))
                except (KeyError, ValueError, TypeError):
                    continue
    except FileNotFoundError:
        return {"error": f"csv not found: {csv_path}"}

    if not utils:
        return {"error": "no samples collected", "n_samples": 0}

    return {
        "n_samples": len(utils),
        "mean_utilization_pct": statistics.mean(utils),
        "median_utilization_pct": statistics.median(utils),
        "stdev_utilization_pct": statistics.pstdev(utils) if len(utils) > 1 else 0.0,
        "min_utilization_pct": min(utils),
        "max_utilization_pct": max(utils),
        "fraction_samples_idle": sum(1 for u in utils if u < idle_threshold_pct) / len(utils),
        "mean_memory_used_mb": statistics.mean(mem_used) if mem_used else None,
        "max_memory_used_mb": max(mem_used) if mem_used else None,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv-path", type=str, required=True)
    p.add_argument("--interval", type=float, default=1.0, help="Seconds between samples")
    p.add_argument("--duration", type=float, default=None, help="Stop after this many seconds (default: run until killed -- intended usage as a companion process)")
    p.add_argument("--gpu-index", type=int, default=None)
    p.add_argument("--json-out", type=str, default=None, help="Write a summary JSON after stopping (only reached on graceful exit, i.e. --duration set -- a companion process killed by its parent won't reach this, use summarize() directly on the CSV instead)")
    args = p.parse_args()

    n_samples, n_failed = run(args.csv_path, args.interval, args.duration, args.gpu_index)
    print(f"[gpu_utilization_profiler] collected {n_samples} samples ({n_failed} failed) -> {args.csv_path}")

    if args.json_out:
        summary = summarize(args.csv_path)
        with open(args.json_out, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"[gpu_utilization_profiler] wrote {args.json_out}")


if __name__ == "__main__":
    sys.exit(main())
