from __future__ import annotations

# After running phases (via Slurm or run_local), verify that every job left what it should under the scratch root and that no job log shows an error.
# Read-only; exits 0 only if everything expected is present and sane. Uses the same phase / filter arguments as slurm.campaign.

import argparse
import json
import math
import os
import re
import sys

from baselines.slurm import campaign
from baselines.slurm.campaign import Job, job_layout

EXPORT_FILES = ("split.json", "train.xyz", "val.xyz", "test.xyz")
LOG_ERROR_PATTERN = re.compile(r"Traceback \(most recent call last\)|^ERROR[: ]|CUDA out of memory|Killed|slurmstepd: error", re.MULTILINE)


def _finite(x: object) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def check_metrics(path: str) -> list[str]:
    try:
        with open(path) as f:
            m = json.load(f)
    except (OSError, ValueError) as e:
        return [f"unreadable metrics.json: {e}"]
    problems = []
    for split in ("val", "test"):
        block = m.get(split)
        if not isinstance(block, dict):
            problems.append(f"no '{split}' block")
        elif split == "test" and not any(_finite(v) for v in block.values()):
            problems.append("test block has no finite number")
        elif any(isinstance(v, (int, float)) and not _finite(v) for v in block.values()):
            problems.append(f"non-finite value in '{split}' block")
    return problems


def check_job(job: Job, scratch: str) -> list[str]:
    export_dir, run_dir, image = job_layout(job, scratch)
    if job.job_type == "smoke_test":
        return [] if os.path.isfile(image) else [f"image missing: {image}"]
    if job.job_type == "export":
        problems = [f"missing {name}" for name in EXPORT_FILES if not os.path.isfile(os.path.join(export_dir, name))]
        if not problems and os.path.getsize(os.path.join(export_dir, "train.xyz")) == 0:
            problems.append("train.xyz is empty")
        if not problems and os.path.getsize(os.path.join(export_dir, "val.xyz")) == 0:
            problems.append("val.xyz is empty (trainers need a validation set; use the _val split names)")
        return problems
    metrics = os.path.join(run_dir, "metrics.json")
    return check_metrics(metrics) if os.path.isfile(metrics) else ["missing metrics.json"]


def scan_logs(log_dirs: list[str]) -> list[str]:
    hits = []
    for d in log_dirs:
        if not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            for name in sorted(files):
                if name.endswith((".err", ".log", ".out")):
                    path = os.path.join(root, name)
                    with open(path, errors="replace") as f:
                        text = f.read()
                    m = LOG_ERROR_PATTERN.search(text)
                    if m:
                        hits.append(f"{path}: {m.group(0).strip()[:60]}")
    return hits


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Check that the jobs of the given phases produced complete, sane outputs under the scratch root.")
    p.add_argument("--phase", nargs="+", required=True, choices=list(campaign.PHASES))
    p.add_argument("--scratch-root", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--datasets", nargs="+", default=None, choices=list(campaign.DATASET_FAMILIES))
    p.add_argument("--only", nargs="+", default=None, help="exact dataset ids to keep, e.g. rmd17:aspirin")
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--spice-file", default=campaign.SPICE_FILENAME)
    p.add_argument("--ani1x-file", default=campaign.ANI1X_FILENAME)
    p.add_argument("--log-dir", nargs="*", default=None,
                   help="directories whose .err/.out/.log files are scanned for errors (default: <scratch>/campaign/logs and <scratch>/campaign/local)")
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    scratch = a.scratch_root.rstrip("/")
    jobs: list[Job] = []
    for phase in a.phase:
        jobs += campaign.expand_phase(phase, tuple(a.seeds), a.spice_file, a.ani1x_file, tuple(a.models) if a.models else None,
                                      tuple(a.datasets) if a.datasets else None, a.pilot)
    if a.only:
        jobs = [j for j in jobs if j.job_type == "smoke_test" or j.dataset in a.only]
    if not jobs:
        print("[check] no jobs match the filters", file=sys.stderr)
        return 2

    bad = 0
    for job in jobs:
        problems = check_job(job, scratch)
        bad += bool(problems)
        print(f"{'OK  ' if not problems else 'FAIL'} {job.phase}/{job.run_name}" + ("" if not problems else "  <- " + "; ".join(problems)))
    log_dirs = a.log_dir if a.log_dir is not None else [f"{scratch}/campaign/logs", f"{scratch}/campaign/local"]
    hits = scan_logs(log_dirs)
    for h in hits:
        print(f"LOG  {h}")
    print(f"[check] {len(jobs) - bad}/{len(jobs)} jobs complete; {len(hits)} log file(s) with error text")
    return 0 if bad == 0 and not hits else 1


if __name__ == "__main__":
    raise SystemExit(main())
