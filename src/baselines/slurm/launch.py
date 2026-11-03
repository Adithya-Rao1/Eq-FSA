from __future__ import annotations

# Renders (never submits) Slurm sbatch scripts for this project's baseline model adapters, for
# CityUHK Burgundy. Template mechanism: stdlib string.Template, per this project's own
# no-new-dependency request -- https://docs.python.org/3/library/string.html#template-strings.
# Sbatch directive shape and the singularity exec invocation follow the cluster admin's own
# known-working skeleton (fixed /usr/bin/singularity path, --nv, one node per job, srun) from
# this project's Burgundy task brief. Standing rule (repo CLAUDE.md): heavy/remote actions are
# always handed off as exact commands for the user to run themselves, never executed here --
# this module must never call subprocess with "sbatch" (see test_slurm_launch.py's source grep).

import argparse
import os
from string import Template
from typing import NamedTuple

TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

_TEMPLATE_FILENAMES = {
    "smoke_test": "smoke_test.sbatch.tmpl",
    "train": "train.sbatch.tmpl",
    "eval": "eval.sbatch.tmpl",
}


class TierSpec(NamedTuple):
    module: str
    models: tuple[str, ...]
    needs_mode: bool


# Fixed adapter CLI contract (owned by sibling agents on other branches; given to this agent as
# an unchangeable interface -- see this project's Burgundy task brief):
#   python -m baselines.adapters.tier0_legacy      --model {schnet,painn,ani2x,aimnet2} ...
#   python -m baselines.adapters.tier1_mpnn        --model {nequip,allegro,mace} ...
#   python -m baselines.adapters.tier2_transformer --model {equiformer,equiformerv2} ...
#   python -m baselines.adapters.tier3_foundation  --model {mace_mp_0,mace_off23,orb_v2,orb_v3,uma} --mode {zero_shot,fine_tune} ...
TIER_SPECS: dict[str, TierSpec] = {
    "tier0_legacy": TierSpec("baselines.adapters.tier0_legacy", ("schnet", "painn", "ani2x", "aimnet2"), False),
    "tier1_mpnn": TierSpec("baselines.adapters.tier1_mpnn", ("nequip", "allegro", "mace"), False),
    "tier2_transformer": TierSpec("baselines.adapters.tier2_transformer", ("equiformer", "equiformerv2"), False),
    "tier3_foundation": TierSpec(
        "baselines.adapters.tier3_foundation",
        ("mace_mp_0", "mace_off23", "orb_v2", "orb_v3", "uma"),
        True,
    ),
}

# Known Burgundy partitions and a conservative default walltime for each, per the cluster admin
# (see task brief). An unlisted partition is still allowed (the cluster may grow new ones) but
# then --time becomes required since there is no sensible default to fall back on.
_PARTITION_DEFAULT_TIME = {
    "debugq": "00:30:00",        # 2h cap, no GPU -- sanity checks only
    "gpu_a40-test": "02:00:00",  # 12h cap -- quick smoke tests
    "stingy": "05:00:00",        # 5-8h cap -- short jobs
    "gpu_v100s": "5-00:00:00",   # 5 day cap
    "gpu_a100": "5-00:00:00",    # 5 day cap
}
_SMOKE_TEST_DEFAULT_TIME = "00:15:00"


def build_adapter_args(
    model: str, dataset: str, split: str, run_dir: str, mode: str | None, extra_adapter_args: str,
) -> str:
    parts = [f"--model {model}", f"--dataset {dataset}", f"--split {split}", f"--run-dir {run_dir}"]
    if mode:
        parts.append(f"--mode {mode}")
    if extra_adapter_args:
        parts.append(extra_adapter_args)
    return " ".join(parts)


def _validate(tier: str, model: str, mode: str | None) -> TierSpec:
    if tier not in TIER_SPECS:
        raise ValueError(f"unknown tier {tier!r}; expected one of {sorted(TIER_SPECS)}")
    spec = TIER_SPECS[tier]
    if model not in spec.models:
        raise ValueError(f"model {model!r} is not valid for tier {tier!r}; expected one of {spec.models}")
    if spec.needs_mode and not mode:
        raise ValueError(f"tier {tier!r} requires --mode {{zero_shot,fine_tune}}")
    if mode and not spec.needs_mode:
        raise ValueError(f"tier {tier!r} does not take --mode (only tier3_foundation does)")
    return spec


def render_sbatch(
    *,
    job_type: str,
    tier: str,
    model: str,
    dataset: str,
    split: str,
    partition: str,
    image: str,
    run_dir: str,
    digest_sidecar: str | None = None,
    mode: str | None = None,
    extra_adapter_args: str = "",
    cpus_per_task: int = 4,
    gpus: int = 1,
    mem: str = "32G",
    time: str | None = None,
    extra_sbatch_directives: str = "",
) -> str:
    if job_type not in _TEMPLATE_FILENAMES:
        raise ValueError(f"unknown job_type {job_type!r}; expected one of {sorted(_TEMPLATE_FILENAMES)}")
    spec = _validate(tier, model, mode)

    resolved_time = time or (
        _SMOKE_TEST_DEFAULT_TIME if job_type == "smoke_test" else _PARTITION_DEFAULT_TIME.get(partition)
    )
    if resolved_time is None:
        raise ValueError(f"--time is required for unknown partition {partition!r} (no default known)")

    resolved_digest_sidecar = digest_sidecar or f"{image}.sha256"
    job_name = f"eqfsa_{tier}_{model}_{job_type}"
    adapter_args = build_adapter_args(model, dataset, split, run_dir, mode, extra_adapter_args)

    template_path = os.path.join(TEMPLATES_DIR, _TEMPLATE_FILENAMES[job_type])
    with open(template_path) as f:
        template_text = f.read()

    # safe_substitute, not substitute: the rendered script is bash, which uses the exact same
    # "$IDENTIFIER" syntax we do for our own placeholders (e.g. the template's own `$IMAGE`,
    # `$WORKDIR`, `$SLURM_CPUS_PER_TASK` bash variables). Strict substitute() would raise KeyError
    # on every one of those, since string.Template can't tell them apart from an unresolved
    # placeholder of ours. safe_substitute() leaves anything not in `mapping` untouched; the
    # TPL_ prefix on every key below is a second, belt-and-suspenders safeguard so a real bash
    # variable name can never accidentally collide with (and get silently replaced by) one of
    # our own placeholders.
    mapping = {
        "TPL_JOB_NAME": job_name,
        "TPL_PARTITION": partition,
        "TPL_CPUS_PER_TASK": str(cpus_per_task),
        "TPL_GPUS": str(gpus),
        "TPL_MEM": mem,
        "TPL_TIME": resolved_time,
        "TPL_IMAGE_PATH": image,
        "TPL_WORKDIR": run_dir,
        "TPL_DIGEST_SIDECAR": resolved_digest_sidecar,
        "TPL_ADAPTER_MODULE": spec.module,
        "TPL_ADAPTER_ARGS": adapter_args,
        "TPL_EXTRA_SBATCH_DIRECTIVES": extra_sbatch_directives,
    }
    return Template(template_text).safe_substitute(mapping)


def write_script(script_text: str, out_path: str) -> str:
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(script_text)
    return out_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Render (never submit) an sbatch script for a baseline adapter run.")
    p.add_argument("--job-type", choices=sorted(_TEMPLATE_FILENAMES), default="train")
    p.add_argument("--tier", required=True, choices=sorted(TIER_SPECS))
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--partition", required=True)
    p.add_argument("--image", required=True, help="path to the .sif on the cluster, e.g. $HOME/scratch/mlip-mpnn.sif")
    p.add_argument("--run-dir", required=True, help="bind-mounted at /work; also this run's RunPaths root")
    p.add_argument("--digest-sidecar", default=None, help="defaults to <image>.sha256 next to --image")
    p.add_argument("--mode", default=None, choices=["zero_shot", "fine_tune"], help="tier3_foundation only")
    p.add_argument("--extra-adapter-args", default="", help="verbatim extra CLI text appended to the adapter invocation")
    p.add_argument("--cpus-per-task", type=int, default=4)
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--mem", default="32G")
    p.add_argument("--time", default=None, help="overrides the partition's default walltime")
    p.add_argument("--out", default=None, help="output path for the rendered .sbatch file")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    script = render_sbatch(
        job_type=args.job_type,
        tier=args.tier,
        model=args.model,
        dataset=args.dataset,
        split=args.split,
        partition=args.partition,
        image=args.image,
        run_dir=args.run_dir,
        digest_sidecar=args.digest_sidecar,
        mode=args.mode,
        extra_adapter_args=args.extra_adapter_args,
        cpus_per_task=args.cpus_per_task,
        gpus=args.gpus,
        mem=args.mem,
        time=args.time,
    )
    out_path = args.out or os.path.join(args.run_dir, "slurm", f"{args.tier}_{args.model}_{args.job_type}.sbatch")
    write_script(script, out_path)
    print(f"[launch] wrote {out_path}")
    print("This project's standing rule: remote/Slurm actions are handed off, never submitted automatically.")
    print("Run this yourself on Burgundy to submit it:")
    print(f"  sbatch {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
