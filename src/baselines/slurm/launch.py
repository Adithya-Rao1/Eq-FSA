from __future__ import annotations

# https://docs.python.org/3/library/string.html#template-strings. Every job runs through one template, slurm/templates/job.sbatch.tmpl;
# this module builds the command list for each job type.

import argparse
import os
import shlex
from string import Template
from typing import NamedTuple

TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
JOB_TEMPLATE = "job.sbatch.tmpl"


class ModelSpec(NamedTuple):
    image: str  # image name without .sif
    module: str  # native-trainer wrapper (python -m <module>)
    takes_model_flag: bool  # SchNetPack and NequIP/Allegro wrappers serve two models and need --model


MODEL_SPECS: dict[str, ModelSpec] = {
    "schnet": ModelSpec("mlip-legacy", "baselines.trainers.schnetpack_wrapper", True),
    "painn": ModelSpec("mlip-legacy", "baselines.trainers.schnetpack_wrapper", True),
    "nequip": ModelSpec("mlip-mpnn", "baselines.trainers.nequip_wrapper", True),
    "allegro": ModelSpec("mlip-mpnn", "baselines.trainers.nequip_wrapper", True),
    "mace": ModelSpec("mlip-mpnn", "baselines.trainers.mace_wrapper", False),
    "equiformerv2": ModelSpec("mlip-transformer", "baselines.trainers.fairchem_wrapper", False),
}
CLASSICAL_IMAGE = "mlip-classical"
CLASSICAL_FORCEFIELDS = ("gaff2", "openff")
EXPORT_IMAGE = "mlip-mpnn"  # torch (dataset loaders) + ase + h5py + rdkit; the export needs no GPU
SMOKE_IMAGES = ("mlip-legacy", "mlip-mpnn", "mlip-transformer", "mlip-classical")

# Known Burgundy partitions
_PARTITION_DEFAULT_TIME = {
    "debugq": "00:30:00",       
    "gpu_a40-test": "02:00:00", 
    "stingy": "05:00:00",        
    "gpu_v100s": "5-00:00:00",   
    "gpu_a100": "5-00:00:00",    
}
_SMOKE_TEST_DEFAULT_TIME = "00:15:00"
_SMOKE_IMPORTS = {
    "mlip-legacy": "import torch.utils.data.dataloader as d; d.T_co = getattr(d, 'T_co', None) or d._T_co; import torch; "
                   "print('cuda_available=', torch.cuda.is_available()); import schnetpack; import baselines.trainers.schnetpack_wrapper",
    "mlip-mpnn": "import torch; print('cuda_available=', torch.cuda.is_available()); import nequip, allegro, mace; "
                 "import baselines.trainers.nequip_wrapper, baselines.trainers.mace_wrapper, baselines.export.export_splits",
    "mlip-transformer": "import torch; print('cuda_available=', torch.cuda.is_available()); import fairchem.core; "
                        "import baselines.trainers.fairchem_wrapper",
    "mlip-classical": "import openmm, openff.toolkit, openmmforcefields, rdkit; import baselines.classical.run",
}


def q(value: object) -> str:
    return shlex.quote(str(value))


def _check_model(model: str) -> ModelSpec:
    if model not in MODEL_SPECS:
        raise ValueError(f"unknown model {model!r}; expected one of {sorted(MODEL_SPECS)}")
    return MODEL_SPECS[model]


def smoke_commands(image_name: str) -> list[str]:
    if image_name not in _SMOKE_IMPORTS:
        raise ValueError(f"no smoke test for image {image_name!r}; expected one of {sorted(_SMOKE_IMPORTS)}")
    return [f"sing python -c {q(_SMOKE_IMPORTS[image_name])}"]


def export_commands(dataset: str, split: str, data_root: str, out_dir: str, max_train: int | None, max_val: int | None, max_test: int | None,
                    subset_seed: int) -> list[str]:
    parts = ["sing", "python", "-m", "baselines.export.export_splits", "--dataset", q(dataset), "--split", q(split), "--data-root", q(data_root),
             "--out-dir", q(out_dir), "--subset-seed", str(subset_seed)]
    for flag, value in (("--max-train", max_train), ("--max-val", max_val), ("--max-test", max_test)):
        if value is not None:
            parts += [flag, str(value)]
    # An existing split.json means a complete export (it is written last), so a re-submitted job never rewrites files other runs are reading.
    return [f"if [[ -f {q(out_dir + '/split.json')} ]]; then echo 'export already complete: {out_dir}'; else", "    " + " ".join(parts), "fi"]


def _require_export(export_dir: str) -> str:
    return (f"[[ -f {q(export_dir + '/split.json')} ]] || {{ echo 'ERROR: no export at {export_dir}; run the phase1_export jobs first' >&2; exit 2; }}")


def train_commands(model: str, export_dir: str, run_dir: str, seed: int, epochs: int, extra_args: str = "") -> list[str]:
    # Train, then evaluate. The wrappers resume from their own checkpoints and exit at once when the run is finished, so re-running this
    # script (a chained copy after the walltime cap) is safe; set -e keeps a failed or unfinished training from reaching the evaluation.
    spec = _check_model(model)
    train = ["sing", "python", "-m", spec.module]
    if spec.takes_model_flag:
        train += ["--model", model]
    train += ["--data-dir", q(export_dir), "--run-dir", q(run_dir), "--seed", str(seed), "--epochs", str(epochs)]
    if extra_args:
        train.append(extra_args)
    evaluate = ["sing", "python", "-m", "baselines.evaluation.run_eval", "--model", model, "--data-dir", q(export_dir), "--run-dir", q(run_dir),
                "--out", q(run_dir + "/metrics.json"), "--seed", str(seed)]
    return [_require_export(export_dir), " ".join(train), f"[[ -f {q(run_dir + '/metrics.json')} ]] || " + " ".join(evaluate)]


def classical_commands(forcefield: str, export_dir: str, run_dir: str, charge_cache: str) -> list[str]:
    if forcefield not in CLASSICAL_FORCEFIELDS:
        raise ValueError(f"forcefield must be one of {CLASSICAL_FORCEFIELDS}, got {forcefield!r}")
    run = ["sing", "python", "-m", "baselines.classical.run", "--data-dir", q(export_dir), "--forcefield", forcefield,
           "--out", q(run_dir + "/metrics.json"), "--charge-cache", q(charge_cache)]
    return [_require_export(export_dir), f"[[ -f {q(run_dir + '/metrics.json')} ]] || " + " ".join(run)]


def render_sbatch(
    *, job_name: str, partition: str, image: str, run_dir: str, commands: list[str], scratch_root: str, repo_dir: str,
    gpus: int = 1, cpus_per_task: int = 4, mem: str = "32G", time: str | None = None, smoke: bool = False, digest_sidecar: str | None = None,
) -> str:
    resolved_time = time or (_SMOKE_TEST_DEFAULT_TIME if smoke else _PARTITION_DEFAULT_TIME.get(partition))
    if resolved_time is None:
        raise ValueError(f"--time is required for unknown partition {partition!r} (no default known)")
    with open(os.path.join(TEMPLATES_DIR, JOB_TEMPLATE)) as f:
        template_text = f.read()
    mapping = {
        "TPL_JOB_NAME": job_name, "TPL_PARTITION": partition, "TPL_CPUS_PER_TASK": str(cpus_per_task), "TPL_MEM": mem, "TPL_TIME": resolved_time,
        "TPL_GRES_DIRECTIVE": f"#SBATCH --gres=gpu:{gpus}" if gpus else "",
        "TPL_NV_FLAG": "--nv" if gpus else "",
        "TPL_IMAGE_PATH": image, "TPL_WORKDIR": run_dir, "TPL_DIGEST_SIDECAR": digest_sidecar or f"{image}.sha256",
        "TPL_SCRATCH_ROOT": scratch_root, "TPL_REPO_DIR": repo_dir, "TPL_COMMANDS": "\n".join(commands),
    }
    return Template(template_text).safe_substitute(mapping)


def write_script(script_text: str, out_path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(script_text)
    return out_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Render (never submit) one training+evaluation sbatch script. Use slurm.campaign for whole phases.")
    p.add_argument("--model", required=True, choices=sorted(MODEL_SPECS))
    p.add_argument("--export-dir", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--scratch-root", required=True)
    p.add_argument("--repo-dir", required=True)
    p.add_argument("--partition", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--time", default=None)
    p.add_argument("--out", required=True)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = MODEL_SPECS[args.model]
    script = render_sbatch(
        job_name=f"eqfsa_{args.model}_train", partition=args.partition, image=f"{args.scratch_root}/{spec.image}.sif", run_dir=args.run_dir,
        commands=train_commands(args.model, args.export_dir, args.run_dir, args.seed, args.epochs), scratch_root=args.scratch_root,
        repo_dir=args.repo_dir, time=args.time,
    )
    write_script(script, args.out)
    print(f"[launch] wrote {args.out}\nThis project's standing rule: Slurm actions are handed off, never submitted automatically.\n  sbatch {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
