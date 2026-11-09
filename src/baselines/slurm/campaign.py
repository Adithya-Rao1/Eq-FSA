from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass

from baselines.datasets.splits import RMD17_MOLECULES
from baselines.slurm.launch import (
    CLASSICAL_FORCEFIELDS, CLASSICAL_IMAGE, EXPORT_IMAGE, MODEL_SPECS, SMOKE_IMAGES, classical_commands, export_commands, render_sbatch,
    smoke_commands, train_commands, write_script,
)

MD22_MOLECULES: tuple[str, ...] = (
    "Ac-Ala3-NHMe", "DHA", "stachyose", "AT-AT", "AT-AT-CG-CG", "buckyball-catcher", "double-walled_nanotube",
)

DATASET_FAMILIES = ("qm9", "rmd17", "md22", "spice", "ani1x")
TRAIN_MATRIX: dict[str, tuple[str, ...]] = {model: DATASET_FAMILIES for model in MODEL_SPECS}
CLASSICAL_DATASETS = ("rmd17", "md22", "spice")

PHASES: dict[str, str] = {
    "phase0_smoke": "smoke-test every image (imports + CUDA passthrough)",
    "phase1_export": "export every dataset instance to extended XYZ + split.json (CPU)",
    "phase2_train_calibration": "train + evaluate every ML model on QM9 / rMD17 / MD22",
    "phase2_train_spice": "train + evaluate every ML model on SPICE (molecule-level 95/5)",
    "phase2_train_ani1x": "train + evaluate every ML model on the fixed seeded ANI-1x subset",
    "phase3_classical": "GAFF2 / OpenFF on the rMD17, MD22 and SPICE test sets (CPU)",
}
TRAIN_PHASE_DATASETS = {
    "phase2_train_calibration": ("qm9", "rmd17", "md22"),
    "phase2_train_spice": ("spice",),
    "phase2_train_ani1x": ("ani1x",),
}

PARTITION = {
    "smoke": "gpu_a40-test",
    "train": "gpu_v100s",
    "train_large": "gpu_a100", 
    "cpu": "stingy", 
}
GPU_RESOURCES = {"cpus_per_task": 8, "mem": "90G", "gpus": 1}
SMOKE_RESOURCES = {"cpus_per_task": 4, "mem": "40G", "gpus": 1}
CPU_RESOURCES = {"cpus_per_task": 8, "mem": "64G", "gpus": 0}

SPICE_FILENAME = "SPICE-2.0.1.hdf5"  
ANI1X_FILENAME = "ani1x-release.h5" 
QM9_SPLIT = "gilmer_110k_10k_rest"
SPICE_SPLIT = "molecule_95_5_val5" 
RMD17_SPLIT = "rmd17_split_0_val50"  
MD22_SPLIT = "md22_fixed"
ANI1X_SPLIT = "ani1x_group"
ANI1X_CAPS = (100_000, 10_000, 10_000)
SUBSET_SEED = 0
RMD17_NOTE = "rMD17 loader keys verified against malonaldehyde npz"


@dataclass(frozen=True)
class Job:
    phase: str
    job_type: str  # smoke_test | export | train | classical
    image: str
    model: str  # ML model, classical force field, or "-" for export / smoke jobs
    dataset: str  # loader argument, e.g. rmd17:aspirin; "smoke" for smoke tests
    split: str
    partition: str
    seed: int
    data_subpath: str | None  # relative to <scratch>/data
    note: str

    @property
    def export_name(self) -> str:
        name = f"{self.dataset.replace(':', '-')}_{self.split}"
        return f"{name}_n{ANI1X_CAPS[0]}" if self.dataset == "ani1x" else name

    @property
    def run_name(self) -> str:
        if self.job_type == "smoke_test":
            return f"smoke_{self.image}"
        if self.job_type == "export":
            return f"export_{self.export_name}"
        return f"{self.model}_{self.export_name}_s{self.seed}"


def _dataset_specs(family: str, spice_file: str, ani1x_file: str) -> list[tuple[str, str, str]]:
    if family == "qm9":
        return [("qm9", QM9_SPLIT, "qm9")]
    if family == "rmd17":
        return [(f"rmd17:{mol}", RMD17_SPLIT, "rmd17") for mol in RMD17_MOLECULES]
    if family == "md22":
        return [(f"md22:{mol}", MD22_SPLIT, "md22") for mol in MD22_MOLECULES]
    if family == "spice":
        return [("spice", SPICE_SPLIT, f"spice/{spice_file}")]
    if family == "ani1x":
        return [("ani1x", ANI1X_SPLIT, f"ani1x/{ani1x_file}")]
    raise ValueError(f"unknown dataset family {family!r}")


def _note(family: str) -> str:
    return RMD17_NOTE if family == "rmd17" else ""


def expand_phase(
    phase: str, seeds: tuple[int, ...] = (0, 1, 2), spice_file: str = SPICE_FILENAME, ani1x_file: str = ANI1X_FILENAME,
    models: tuple[str, ...] | None = None, datasets: tuple[str, ...] | None = None, pilot: bool = False, cpu_partition: str = PARTITION["cpu"],
    gpu_partition: str | None = None,
) -> list[Job]:
    if phase not in PHASES:
        raise ValueError(f"unknown phase {phase!r}; expected one of {sorted(PHASES)}")
    wanted = lambda family: datasets is None or family in datasets  # noqa: E731
    jobs: list[Job] = []

    # gpu_partition replaces every GPU partition below (smoke / train / train_large) for clusters whose partitions differ from Burgundy's
    gpu = lambda kind: gpu_partition or PARTITION[kind]  # noqa: E731

    if phase == "phase0_smoke":
        return [Job(phase, "smoke_test", image, "-", "smoke", "smoke", gpu("smoke"), 0, None, "imports the wrappers and checks torch.cuda")
                for image in SMOKE_IMAGES]

    if phase == "phase1_export":
        for family in DATASET_FAMILIES:
            if wanted(family):
                for dataset, split, subpath in _dataset_specs(family, spice_file, ani1x_file):
                    jobs.append(Job(phase, "export", EXPORT_IMAGE, "-", dataset, split, cpu_partition, 0, subpath, _note(family)))
        return jobs

    if phase in TRAIN_PHASE_DATASETS:
        for model, families in TRAIN_MATRIX.items():
            if models is not None and model not in models:
                continue
            partition = gpu("smoke") if pilot else gpu("train_large" if model == "equiformerv2" else "train")
            for family in families:
                if family not in TRAIN_PHASE_DATASETS[phase] or not wanted(family):
                    continue
                for dataset, split, subpath in _dataset_specs(family, spice_file, ani1x_file):
                    for seed in seeds[:1] if pilot else seeds:
                        jobs.append(Job(phase, "train", MODEL_SPECS[model].image, model, dataset, split, partition, seed, subpath, _note(family)))
                    if pilot:
                        break  # one dataset instance per (model, family) is enough for timing
        return jobs

    # phase3_classical: deterministic, so one run per (force field, dataset instance), no seeds
    for forcefield in CLASSICAL_FORCEFIELDS:
        if models is not None and forcefield not in models:
            continue
        for family in CLASSICAL_DATASETS:
            if wanted(family):
                for dataset, split, subpath in _dataset_specs(family, spice_file, ani1x_file):
                    jobs.append(Job(phase, "classical", CLASSICAL_IMAGE, forcefield, dataset, split, cpu_partition, 0, subpath, _note(family)))
    return jobs


def job_layout(job: Job, scratch_root: str) -> tuple[str, str, str]:
    """(export_dir, run_dir, image path) for a job; shared by the Slurm renderer and the local runner."""
    return (f"{scratch_root}/exports/{job.export_name}", f"{scratch_root}/runs/{job.phase}/{job.run_name}", f"{scratch_root}/{job.image}.sif")


def job_commands(job: Job, scratch_root: str, epochs: int, extra_adapter_args: str) -> list[str]:
    export_dir, run_dir, _ = job_layout(job, scratch_root)
    if job.job_type == "smoke_test":
        return smoke_commands(job.image)
    if job.job_type == "export":
        caps = ANI1X_CAPS if job.dataset == "ani1x" else (None, None, None)
        return export_commands(job.dataset, job.split, f"{scratch_root}/data/{job.data_subpath}", export_dir, *caps, SUBSET_SEED)
    if job.job_type == "classical":
        return classical_commands(job.model, export_dir, run_dir, f"{scratch_root}/classical_charges")
    return train_commands(job.model, export_dir, run_dir, job.seed, epochs, extra_adapter_args)


def render_job(job: Job, scratch_root: str, repo_dir: str, epochs: int, extra_adapter_args: str, time: str | None) -> str:
    _, run_dir, image = job_layout(job, scratch_root)
    commands = job_commands(job, scratch_root, epochs, extra_adapter_args)
    common = dict(partition=job.partition, image=image, run_dir=run_dir, scratch_root=scratch_root, repo_dir=repo_dir, time=time)
    if job.job_type == "smoke_test":
        return render_sbatch(job_name=f"eqfsa_{job.run_name}", commands=commands, smoke=True, **SMOKE_RESOURCES, **common)
    if job.job_type == "export":
        return render_sbatch(job_name=f"eqfsa_{job.run_name}", commands=commands, **CPU_RESOURCES, **common)
    if job.job_type == "classical":
        return render_sbatch(job_name=f"eqfsa_classical_{job.run_name}", commands=commands, **CPU_RESOURCES, **common)
    return render_sbatch(job_name=f"eqfsa_{job.model}_{job.phase}", commands=commands, **GPU_RESOURCES, **common)


def submit_script(script_paths: list[str], chain: int) -> str:
    lines = [
        "#!/bin/bash",
        "# Generated by baselines.slurm.campaign -- review, then run on the Burgundy login node. Nothing here has been submitted.",
        "set -euo pipefail",
        'cd "$(dirname "${BASH_SOURCE[0]}")"  # script paths below are relative to this file, so the folder can be copied to Burgundy',
        "mkdir -p logs  # Slurm does not create the directory named in --output; each job's .out/.err land here",
    ]
    for path in script_paths:
        lines.append(f'PREV=$(sbatch --parsable "{path}")  # {os.path.basename(path)}')
        for _ in range(chain - 1):
            # Each extra copy starts after the previous one ends
            lines.append(f'PREV=$(sbatch --parsable --dependency=afterany:$PREV "{path}")')
    lines.append('echo "submitted; check with: squeue -u $USER"')
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Render (never submit) the sbatch scripts for one phase of the baseline comparison.")
    p.add_argument("--phase", required=True, choices=list(PHASES))
    p.add_argument("--scratch-root", required=True, help="Burgundy scratch dir holding the .sif images, data/, exports/ and runs/, e.g. $HOME/scratch")
    p.add_argument("--out-dir", required=True, help="where to write the sbatch scripts, submit script and manifest")
    p.add_argument("--epochs", type=int, default=1, help="training epochs (no hyperparameters are chosen for you; 1 is for pilots)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--models", nargs="+", default=None, help="ML models (or gaff2 / openff for phase3_classical)")
    p.add_argument("--datasets", nargs="+", default=None, choices=list(DATASET_FAMILIES))
    p.add_argument("--chain", type=int, default=1, help="submit this many dependent copies of each job (5-day walltime cap)")
    p.add_argument("--time", default=None, help="override the partition's default walltime")
    p.add_argument("--repo-dir", default=None, help="Burgundy path of this repo checkout (default: <scratch-root>/Eq-FSA)")
    p.add_argument("--cpu-partition", default=PARTITION["cpu"], help="partition for the CPU-only export and classical jobs")
    p.add_argument("--gpu-partition", default=None, help="use this partition for every GPU job (smoke, train); also pass --time unless it is a known partition")
    p.add_argument("--spice-file", default=SPICE_FILENAME)
    p.add_argument("--ani1x-file", default=ANI1X_FILENAME)
    p.add_argument("--extra-adapter-args", default="", help="verbatim extra CLI text appended to the trainer command")
    p.add_argument("--pilot", action="store_true", help="timing pilot: seed 0, one dataset instance per model, gpu_a40-test partition")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scratch = args.scratch_root.rstrip("/")
    jobs = expand_phase(args.phase, tuple(args.seeds), args.spice_file, args.ani1x_file, tuple(args.models) if args.models else None,
                        tuple(args.datasets) if args.datasets else None, args.pilot, args.cpu_partition, args.gpu_partition)
    print(f"[campaign] {args.phase}: {PHASES[args.phase]}")
    if not jobs:
        print("[campaign] no jobs match --models / --datasets for this phase; nothing rendered")
        return 1
    scripts = []
    for job in jobs:
        path = os.path.join(args.out_dir, args.phase, f"{job.run_name}.sbatch")
        write_script(render_job(job, scratch, args.repo_dir or f"{scratch}/Eq-FSA", args.epochs, args.extra_adapter_args, args.time), path)
        scripts.append(path)
    submit_path = os.path.join(args.out_dir, f"submit_{args.phase}.sh")
    write_script(submit_script([os.path.relpath(p, args.out_dir) for p in scripts], max(1, args.chain)), submit_path)
    write_script(json.dumps([asdict(j) for j in jobs], indent=1), os.path.join(args.out_dir, f"manifest_{args.phase}.json"))
    print(f"[campaign] rendered {len(scripts)} job scripts")
    for note in sorted({j.note for j in jobs if j.note}):
        print(f"[campaign]   note: {note}")
    print(f"[campaign] review, then run on Burgundy:  bash {submit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
