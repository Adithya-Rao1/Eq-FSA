from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass

from baselines.datasets.splits import RMD17_MOLECULES
from baselines.slurm.launch import TIER_SPECS, render_sbatch, write_script

# Duplicated from baselines.datasets.md22 so this module imports without torch/numpy 
MD22_MOLECULES: tuple[str, ...] = (
    "Ac-Ala3-NHMe", "DHA", "stachyose", "AT-AT", "AT-AT-CG-CG", "buckyball-catcher", "double-walled_nanotube",
)

IMAGE_FOR_TIER = {
    "tier0_legacy": "mlip-legacy",
    "tier1_mpnn": "mlip-mpnn",
    "tier2_transformer": "mlip-transformer",
    "tier3_foundation": "mlip-foundation",
}

# Artifact section 02, "Trained from scratch on" column. ANI-1x is not listed for any model there, so it is not scheduled.
TRAIN_MATRIX: dict[tuple[str, str], tuple[str, ...]] = {
    ("tier0_legacy", "schnet"): ("qm9", "rmd17"),
    ("tier0_legacy", "painn"): ("qm9", "rmd17"),
    ("tier0_legacy", "ani2x"): ("rmd17", "spice"),
    ("tier0_legacy", "aimnet2"): ("spice",),
    ("tier1_mpnn", "nequip"): ("rmd17", "md22", "spice"),
    ("tier1_mpnn", "allegro"): ("rmd17", "md22", "spice"),
    ("tier1_mpnn", "mace"): ("rmd17", "md22", "spice"),
    ("tier2_transformer", "equiformer"): ("spice",),
    ("tier2_transformer", "equiformerv2"): ("spice",),
}
ZERO_SHOT_MODELS = ("mace_mp_0", "mace_off23", "orb_v2", "orb_v3", "uma")

PARTITION = {
    "smoke": "gpu_a40-test",
    "scratch_train_small": "gpu_v100s",
    "tier2_and_finetune": "gpu_a100",
    "zero_shot": "stingy",
    "md": "gpu_a100",
}
SMOKE_RESOURCES = {"cpus_per_task": 4, "mem": "40G"}
TRAIN_RESOURCES = {"cpus_per_task": 8, "mem": "90G"}

PHASES: dict[str, str] = {
    "phase0_smoke": "Week 1: smoke-test every image (imports + CUDA passthrough)",
    "phase1_calibration": "Weeks 2-3: Tier 0-1 from scratch on QM9 / rMD17 / MD22; review the calibration gate before phase1_spice",
    "phase1_spice": "Weeks 2-3: Tier 0-2 from scratch on SPICE (molecule-level 95/5)",
    "phase2_zero_shot": "Weeks 4-5: Tier 3 foundation models zero-shot on OMol25",
    "phase2_omol25_finetune": "Weeks 4-5: fine-tune MACE on an OMol25 subsample",
    "phase3_solvation": "Weeks 6-7: alchemical solvation free energy (FreeSolv / MNSol)",
    "phase4_md": "Week 8: MD vs classical force fields",
}

# Layout under --scratch-root on Burgundy: <root>/<image>.sif (+ .sha256), <root>/data/<name>, <root>/runs/<phase>/<run>.
SPICE_FILENAME = "SPICE-2.0.1.hdf5"  # assumption: adjust with --spice-file if the downloaded release is named differently
QM9_SPLIT = "gilmer_110k_10k_rest"
SPICE_SPLIT = "molecule_95_5"
RMD17_SPLIT = "rmd17_split_0"  # seeds vary init/shuffle; all use the first of rMD17's 5 published splits

BLOCKERS = {
    "tier2_transformer": "adapter only builds the model: there is no training loop (and equiformer v1 raises NotImplementedError)",
    "tier3_foundation": "adapter only builds the model: neither zero_shot nor fine_tune is implemented",
    "omol25": "omol25 is not wired into adapters/dataset_loading.py",
    "phase3_solvation": "no alchemical free-energy code exists in the repo (only the plan's note to extend stage_1/experiments)",
    "phase4_md": "needs a deployed/TorchScript potential per model, which nothing exports, and the md/ runners take different CLI "
                 "arguments than the adapter templates",
}
PARTIAL_NOTES = {
    "tier0_legacy": "trains and checkpoints only: no val/test evaluation, so no MAE/RMSE is produced",
    "tier1_mpnn": "trains and checkpoints only: no val/test evaluation, so no MAE/RMSE is produced",
}
RMD17_NOTE = "rMD17 npz key names are unverified (CLAUDE.md open issue 3)"


@dataclass(frozen=True)
class Job:
    phase: str
    job_type: str
    tier: str
    model: str
    dataset: str
    split: str
    mode: str | None
    partition: str
    seed: int
    data_subpath: str | None  # relative to <scratch>/data, also what --data-root points at under /data
    status: str  # ready | partial | blocked
    note: str

    @property
    def run_name(self) -> str:
        dataset = self.dataset.replace(":", "-")
        return f"{self.tier}_{self.model}_{dataset}_{self.split}_s{self.seed}"


def _dataset_specs(name: str, spice_file: str) -> list[tuple[str, str, str]]:
    # (dataset arg for adapters/dataset_loading.py, split name, data subpath)
    if name == "qm9":
        return [("qm9", QM9_SPLIT, "qm9")]
    if name == "rmd17":
        return [(f"rmd17:{mol}", RMD17_SPLIT, "rmd17") for mol in RMD17_MOLECULES]
    if name == "md22":
        return [(f"md22:{mol}", "md22_fixed", "md22") for mol in MD22_MOLECULES]
    if name == "spice":
        return [("spice", SPICE_SPLIT, f"spice/{spice_file}")]
    raise ValueError(f"unknown dataset {name!r}")


def _train_status(tier: str, dataset_name: str) -> tuple[str, str]:
    if tier in BLOCKERS:
        return "blocked", BLOCKERS[tier]
    note = PARTIAL_NOTES[tier]
    if dataset_name == "rmd17":
        note = f"{note}; {RMD17_NOTE}"
    return "partial", note


def expand_phase(
    phase: str,
    seeds: tuple[int, ...] = (0, 1, 2),
    spice_file: str = SPICE_FILENAME,
    models: tuple[str, ...] | None = None,
    datasets: tuple[str, ...] | None = None,
    pilot: bool = False,
) -> list[Job]:
    if phase not in PHASES:
        raise ValueError(f"unknown phase {phase!r}; expected one of {sorted(PHASES)}")
    jobs: list[Job] = []

    if phase == "phase0_smoke":
        for tier, spec in TIER_SPECS.items():
            if models is None or spec.models[0] in models:
                jobs.append(Job(phase, "smoke_test", tier, spec.models[0], "smoke", "smoke", "zero_shot" if spec.needs_mode else None,
                                PARTITION["smoke"], 0, None, "ready", "imports the adapter module and checks torch.cuda"))
        return jobs

    if phase in ("phase1_calibration", "phase1_spice"):
        wanted = ("spice",) if phase == "phase1_spice" else ("qm9", "rmd17", "md22")
        for (tier, model), model_datasets in TRAIN_MATRIX.items():
            if models is not None and model not in models:
                continue
            for dataset_name in model_datasets:
                if dataset_name not in wanted or (datasets is not None and dataset_name not in datasets):
                    continue
                status, note = _train_status(tier, dataset_name)
                partition = PARTITION["tier2_and_finetune"] if tier == "tier2_transformer" else PARTITION["scratch_train_small"]
                for dataset, split, subpath in _dataset_specs(dataset_name, spice_file):
                    for seed in seeds[:1] if pilot else seeds:
                        jobs.append(Job(phase, "train", tier, model, dataset, split, None, PARTITION["smoke"] if pilot else partition,
                                        seed, subpath, status, note))
                    if pilot:
                        break  # one dataset instance per (model, dataset family) is enough for timing
        return jobs

    if phase == "phase2_zero_shot":
        for model in ZERO_SHOT_MODELS:
            if models is None or model in models:
                for seed in seeds[:1]:  # zero-shot inference is deterministic given the checkpoint; one run per model
                    jobs.append(Job(phase, "train", "tier3_foundation", model, "omol25", "omol25_ood", "zero_shot",
                                    PARTITION["zero_shot"], seed, "omol25", "blocked", f"{BLOCKERS['tier3_foundation']}; {BLOCKERS['omol25']}"))
        return jobs

    if phase == "phase2_omol25_finetune":
        for seed in seeds:
            jobs.append(Job(phase, "train", "tier3_foundation", "mace_mp_0", "omol25", "omol25_ood", "fine_tune",
                            PARTITION["tier2_and_finetune"], seed, "omol25", "blocked", f"{BLOCKERS['tier3_foundation']}; {BLOCKERS['omol25']}"))
        return jobs

    # phase3_solvation / phase4_md have no launchable entrypoint: report the gap rather than invent jobs.
    return []


def phase_gap(phase: str) -> str | None:
    return BLOCKERS.get(phase)


def render_job(job: Job, scratch_root: str, epochs: int, extra_adapter_args: str, time: str | None, repo_dir: str | None = None) -> str:
    image = f"{scratch_root}/{IMAGE_FOR_TIER[job.tier]}.sif"
    run_dir = f"{scratch_root}/runs/{job.phase}/{job.run_name}"
    extra = [f"--seed {job.seed}"]
    if job.job_type == "train":
        if job.tier in ("tier0_legacy", "tier1_mpnn"):
            extra.append(f"--epochs {epochs}")
        if job.data_subpath:
            extra.append(f"--data-root /data/{job.data_subpath}")
    if extra_adapter_args:
        extra.append(extra_adapter_args)
    resources = SMOKE_RESOURCES if job.job_type == "smoke_test" else TRAIN_RESOURCES
    return render_sbatch(
        job_type=job.job_type, tier=job.tier, model=job.model, dataset=job.dataset, split=job.split, partition=job.partition,
        image=image, run_dir=run_dir, mode=job.mode, extra_adapter_args=" ".join(extra), time=time,
        data_dir=None if job.job_type == "smoke_test" else f"{scratch_root}/data", repo_dir=repo_dir or f"{scratch_root}/Eq-FSA", **resources,
    )


def submit_script(script_paths: list[str], chain: int) -> str:
    # Plain text for the user to read and run on Burgundy's login node; Python never executes it.
    lines = [
        "#!/bin/bash",
        "# Generated by baselines.slurm.campaign -- review, then run on the Burgundy login node. Nothing here has been submitted.",
        "set -euo pipefail",
        'cd "$(dirname "${BASH_SOURCE[0]}")"  # script paths below are relative to this file, so the folder can be copied to Burgundy',
    ]
    for path in script_paths:
        lines.append(f'PREV=$(sbatch --parsable "{path}")  # {os.path.basename(path)}')
        for _ in range(chain - 1):
            # Each extra copy starts after the previous ends (any reason); the train template makes a finished run exit at once and an interrupted one resume from its newest checkpoint.
            lines.append(f'PREV=$(sbatch --parsable --dependency=afterany:$PREV "{path}")')
    lines.append('echo "submitted; check with: squeue -u $USER"')
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Render (never submit) the sbatch scripts for one phase of the baseline timeline.")
    p.add_argument("--phase", required=True, choices=list(PHASES))
    p.add_argument("--scratch-root", required=True, help="Burgundy scratch dir holding the .sif images, data/ and runs/, e.g. $HOME/scratch")
    p.add_argument("--out-dir", required=True, help="where to write the sbatch scripts, submit script and manifest")
    p.add_argument("--epochs", type=int, default=1, help="tier0/tier1 epochs (no hyperparameters are chosen for you; 1 is for pilots)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--datasets", nargs="+", default=None, choices=["qm9", "rmd17", "md22", "spice"])
    p.add_argument("--chain", type=int, default=1, help="submit this many dependent copies of each job (5-day walltime cap)")
    p.add_argument("--time", default=None, help="override the partition's default walltime")
    p.add_argument("--repo-dir", default=None, help="Burgundy path of this repo checkout (default: <scratch-root>/Eq-FSA)")
    p.add_argument("--spice-file", default=SPICE_FILENAME)
    p.add_argument("--extra-adapter-args", default="")
    p.add_argument("--pilot", action="store_true", help="timing pilot: seed 0, one dataset instance per model, gpu_a40-test partition")
    p.add_argument("--include-blocked", action="store_true", help="also render jobs whose code does not exist yet (they will fail)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    jobs = expand_phase(args.phase, tuple(args.seeds), args.spice_file, tuple(args.models) if args.models else None,
                        tuple(args.datasets) if args.datasets else None, args.pilot)
    gap = phase_gap(args.phase)
    counts = {s: sum(j.status == s for j in jobs) for s in ("ready", "partial", "blocked")}
    print(f"[campaign] {args.phase}: {PHASES[args.phase]}")
    if gap and not jobs:
        print(f"[campaign] BLOCKED, nothing to render: {gap}")
        return 1

    renderable = [j for j in jobs if j.status != "blocked" or args.include_blocked]
    scripts = []
    for job in renderable:
        path = os.path.join(args.out_dir, args.phase, f"{job.run_name}.sbatch")
        write_script(render_job(job, args.scratch_root.rstrip("/"), args.epochs, args.extra_adapter_args, args.time, args.repo_dir), path)
        scripts.append(path)
    if scripts:
        submit_path = os.path.join(args.out_dir, f"submit_{args.phase}.sh")
        write_script(submit_script([os.path.relpath(p, args.out_dir) for p in scripts], max(1, args.chain)), submit_path)
    manifest = [asdict(j) for j in jobs]
    write_script(json.dumps(manifest, indent=1), os.path.join(args.out_dir, f"manifest_{args.phase}.json"))

    print(f"[campaign] jobs: {counts['ready']} ready, {counts['partial']} partial, {counts['blocked']} blocked; rendered {len(scripts)}")
    for status in ("partial", "blocked"):
        for note in sorted({j.note for j in jobs if j.status == status}):
            print(f"[campaign]   {status}: {note}")
    if counts["blocked"] and not args.include_blocked:
        print("[campaign] blocked jobs were skipped (pass --include-blocked to render them anyway)")
    if scripts:
        print(f"[campaign] review, then run on Burgundy:  bash {submit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
