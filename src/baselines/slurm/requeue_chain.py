from __future__ import annotations

# Checkpoint-and-requeue wrapper for CityUHK Burgundy's 5-day walltime cap (per this project's
# Burgundy task brief: "longer work needs checkpoint-and-requeue"). Only ever prints/generates
# the `scontrol requeue` / `sbatch --dependency=afterany:` commands -- never executes them,
# matching this project's standing rule that remote/Slurm actions are handed off to the user to
# run themselves (repo CLAUDE.md). Slurm docs: https://slurm.schedmd.com/scontrol.html (requeue),
# https://slurm.schedmd.com/sbatch.html#OPT_dependency (afterany chaining).

import argparse
import os
import subprocess
from pathlib import Path

from baselines.config.schema import RunPaths
from baselines.slurm.launch import TIER_SPECS, render_sbatch, write_script

_DESCRIPTION = "Check a run's checkpoint/walltime state and print/generate the requeue step (never submits anything)."


def is_training_complete(run_dir: str) -> bool:
    # Design assumption, made without visibility into the sibling agents' adapter code (not
    # present in this worktree): "complete" is defined here as a checkpoint tagged "final"
    # existing, via RunPaths.checkpoint("final"). Flag this against the real adapters' actual
    # completion convention once they land -- if they signal completion differently, this
    # function (and only this function) needs to change.
    return Path(RunPaths(run_dir).checkpoint("final")).exists()


def latest_checkpoint(run_dir: str) -> str | None:
    paths = RunPaths(run_dir)
    ckpt_dir = Path(paths.checkpoint_dir)
    if not ckpt_dir.is_dir():
        return None
    checkpoints = sorted(ckpt_dir.glob("checkpoint_*.pt"), key=lambda p: p.stat().st_mtime)
    return str(checkpoints[-1]) if checkpoints else None


def slurm_time_left_seconds(job_id: str) -> int | None:
    # Local Slurm client query (squeue) -- a cluster-internal command, not an outbound-internet
    # call, so this is safe to run from inside a job on Burgundy's internal-only network.
    try:
        completed = subprocess.run(
            ["squeue", "-h", "-j", job_id, "-o", "%L"],
            capture_output=True, text=True, timeout=15, check=True,
        )
    except Exception:
        return None
    return _parse_slurm_time_left(completed.stdout.strip())


def _parse_slurm_time_left(raw: str) -> int | None:
    # Slurm's "%L" (time left) format is "[D-]HH:MM:SS" -- the day segment is only present once
    # at least one full day remains.
    if not raw:
        return None
    try:
        days = 0
        remainder = raw
        if "-" in raw:
            day_part, remainder = raw.split("-", 1)
            days = int(day_part)
        hours, minutes, seconds = (int(x) for x in remainder.split(":"))
        return days * 86400 + hours * 3600 + minutes * 60 + seconds
    except (ValueError, AttributeError):
        return None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=_DESCRIPTION)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--tier", required=True, choices=sorted(TIER_SPECS))
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--partition", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--digest-sidecar", default=None)
    p.add_argument("--mode", default=None, choices=["zero_shot", "fine_tune"], help="tier3_foundation only")
    p.add_argument("--extra-adapter-args", default="")
    p.add_argument("--requeue-threshold-seconds", type=int, default=900, help="requeue once remaining walltime drops below this")
    p.add_argument("--chain-mode", choices=["requeue", "chain"], default="requeue",
                    help="'requeue': print `scontrol requeue` for the SAME allocation. "
                         "'chain': render+write a new dependent sbatch script instead.")
    p.add_argument("--out", default=None, help="output path for the rendered dependent sbatch script (chain mode only)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if is_training_complete(args.run_dir):
        print(f"[requeue_chain] {args.run_dir} already has a final checkpoint; nothing to requeue.")
        return 0

    job_id = os.environ.get("SLURM_JOB_ID")
    remaining = slurm_time_left_seconds(job_id) if job_id else None
    if remaining is not None and remaining >= args.requeue_threshold_seconds:
        print(f"[requeue_chain] {remaining}s remain (>= threshold {args.requeue_threshold_seconds}s); nothing to do yet.")
        return 0
    if remaining is None:
        print(
            "[requeue_chain] could not determine remaining walltime (not inside a running Slurm "
            "job, or squeue unavailable) -- proceeding as though a requeue decision is wanted now."
        )

    checkpoint = latest_checkpoint(args.run_dir)
    resume_arg = f"--resume {checkpoint}" if checkpoint else ""
    extra_adapter_args = " ".join(part for part in (args.extra_adapter_args, resume_arg) if part)

    # Literal placeholder text when not actually running inside a job (e.g. this is being tested
    # standalone) -- the user fills in their real $SLURM_JOB_ID when they run the printed command.
    job_id_display = job_id or "$SLURM_JOB_ID"

    if args.chain_mode == "requeue":
        print("[requeue_chain] walltime running low. Run this to requeue the SAME job allocation:")
        print(f"  scontrol requeue {job_id_display}")
        return 0

    out_path = args.out or os.path.join(args.run_dir, "slurm", "requeue_chain_next.sbatch")
    script = render_sbatch(
        job_type="train",
        tier=args.tier,
        model=args.model,
        dataset=args.dataset,
        split=args.split,
        partition=args.partition,
        image=args.image,
        run_dir=args.run_dir,
        digest_sidecar=args.digest_sidecar,
        mode=args.mode,
        extra_adapter_args=extra_adapter_args,
        extra_sbatch_directives=f"#SBATCH --dependency=afterany:{job_id_display}",
    )
    write_script(script, out_path)
    print(f"[requeue_chain] wrote dependent sbatch script: {out_path}")
    print("This project's standing rule: remote/Slurm actions are handed off, never submitted automatically.")
    print("Run this yourself to submit it:")
    print(f"  sbatch {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
