from __future__ import annotations
# Tier 3 pretrained foundation models --> zero-shot and fine-tune
# MACE-MP-0 -- Batatia et al., arXiv:2401.00096.
# MACE-OFF23 -- Kovacs et al., arXiv:2312.15211.
# Orb-v2/v3 -- Rhodes et al., arXiv:2410.22570, (pinned to tag v0.7.0, commit 82bab14c091509f2f8f5800ed725dcc596a60953 (2026-05-26))         
# UMA -- Meta FAIR, arXiv:2506.23971.

import argparse
import os

from baselines.config.schema import CommonConfig, RunPaths, add_common_args
from baselines.provenance.env_manifest import write as write_env_manifest
from baselines.provenance.seeding import dataloader_generator, seed_worker, set_seed

TIER3_MODELS = ("mace_mp_0", "mace_off23", "orb_v2", "orb_v3", "uma")
TIER3_MODES = ("zero_shot", "fine_tune")

UMA_HF_REPO = "facebook/UMA"
UMA_LICENSE_URL = "https://huggingface.co/facebook/UMA"


class GatedAccessError(RuntimeError):
    pass


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument("--model", required=True, choices=list(TIER3_MODELS))
    p.add_argument(
        "--mode",
        required=True,
        choices=list(TIER3_MODES),
        help="Tier 3 is zero-shot or fine-tune only; there is no from-scratch mode for pretrained "
        "foundation models.",
    )
    return p


def _build_dataloader(dataset, cfg: CommonConfig, shuffle: bool):
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=4 if cfg.num_workers > 0 else None,
        worker_init_fn=seed_worker,
        generator=dataloader_generator(cfg.seed),
    )


def _init_single_node_ddp() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        nnodes = os.environ.get("GROUP_WORLD_SIZE") or os.environ.get("SLURM_NNODES") or "1"
        if int(nnodes) > 1:
            raise RuntimeError(
                f"Multi-node DDP is not supported by this adapter (GROUP_WORLD_SIZE/SLURM_NNODES={nnodes}); "
                "launch with a single-node `torchrun --standalone --nproc_per_node=<N>` instead."
            )
        import torch
        import torch.distributed as dist

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return rank, world_size, local_rank


def _check_hf_gate() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise GatedAccessError(
            f"UMA checkpoints are gated on Hugging Face ({UMA_HF_REPO}). Set HF_TOKEN (or "
            f"HUGGING_FACE_HUB_TOKEN) to a token for an account that has accepted the FAIR "
            f"Chemistry License at {UMA_LICENSE_URL} before running --model uma."
        )
    return token


def _check_mace_off23_license() -> None:
    if os.environ.get("MACE_OFF23_LICENSE_ACCEPTED") != "1":
        raise GatedAccessError(
            "MACE-OFF23 is distributed under the Academic Software License (non-commercial use "
            "only; see https://github.com/ACEsuit/mace-off#license). Set "
            "MACE_OFF23_LICENSE_ACCEPTED=1 to confirm you have read and accept that license before "
            "running --model mace_off23."
        )


def _mace_cache_dir(cfg: CommonConfig) -> str:
    cache_dir = os.path.join(cfg.run_dir, "checkpoints", "mace_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _build_mace_mp_0(cfg: CommonConfig):
    try:
        from mace.calculators import mace_mp  # type: ignore
    except ImportError as e:
        raise ImportError(
            "mace-torch is not installed. Run `pip install mace-torch` before dispatching MACE-MP-0."
        ) from e
    # NOTE: exact kwarg names (cache_dir vs model_path, etc.) should be confirmed against the installed mace-torch version's `inspect.signature(mace_mp)` 
    return mace_mp(model="small", device=cfg.device, default_dtype="float64")


def _build_mace_off23(cfg: CommonConfig):
    _check_mace_off23_license()
    try:
        from mace.calculators import mace_off  # type: ignore
    except ImportError as e:
        raise ImportError(
            "mace-torch is not installed. Run `pip install mace-torch` in the remote GPU "
            "environment before dispatching MACE-OFF23."
        ) from e
    cache_dir = _mace_cache_dir(cfg)
    return mace_off(model="small", device=cfg.device, default_dtype="float64", cache_dir=cache_dir)


def _build_orb_v2(cfg: CommonConfig):
    try:
        from orb_models.forcefield import pretrained as orb_pretrained  # type: ignore
    except ImportError as e:
        raise ImportError(
            "orb-models is not installed. Run `pip install -e src/baselines/codebases/orb-models` "
            "in the remote GPU environment before dispatching Orb-v2."
        ) from e
    model, atoms_adapter = orb_pretrained.orb_v2(device=cfg.device)
    return model, atoms_adapter


def _build_orb_v3(cfg: CommonConfig):
    try:
        from orb_models.forcefield import pretrained as orb_pretrained  # type: ignore
    except ImportError as e:
        raise ImportError(
            "orb-models is not installed. Run `pip install -e src/baselines/codebases/orb-models` "
            "in the remote GPU environment before dispatching Orb-v3."
        ) from e
    # orb_v3_conservative_inf_omat is the general-purpose v3 checkpoint upstream's README recommends --> conservative (forces = -dE/dx). 
    # Use orb_v3_direct_omol / similar from orb_models.forcefield.pretrained directly for the non-conservative v3
    model, atoms_adapter = orb_pretrained.orb_v3_conservative_inf_omat(device=cfg.device)
    return model, atoms_adapter


def _build_uma(cfg: CommonConfig):
    _check_hf_gate()
    try:
        from fairchem.core.calculate.pretrained_mlip import get_predict_unit  # type: ignore
    except ImportError as e:
        raise ImportError(
            "fairchem is not installed. Run `pip install -e src/baselines/codebases/fairchem` "
            "(or `pip install fairchem-core`) in the remote GPU environment before dispatching UMA."
        ) from e
    return get_predict_unit("uma-s-1p2p1", device=cfg.device)


_DISPATCH = {
    "mace_mp_0": _build_mace_mp_0,
    "mace_off23": _build_mace_off23,
    "orb_v2": _build_orb_v2,
    "orb_v3": _build_orb_v3,
    "uma": _build_uma,
}


def main() -> None:
    args = build_parser().parse_args()
    if args.mode not in TIER3_MODES:
        raise ValueError(
            f"--mode is required and must be one of {TIER3_MODES} for Tier 3 (zero-shot/fine-tune "
            "only; pretrained foundation models are never trained from scratch)."
        )
    cfg = CommonConfig.from_args(args)
    paths = RunPaths(cfg.run_dir).ensure()
    set_seed(cfg.seed, cfg.deterministic)
    write_env_manifest(cfg.run_dir)

    cpus_per_task = os.environ.get("SLURM_CPUS_PER_TASK")
    if cpus_per_task:
        import torch

        torch.set_num_threads(int(cpus_per_task))

    _init_single_node_ddp()

    if cfg.amp:
        # verify --amp
        pass

    built = _DISPATCH[args.model](cfg)
    if args.mode == "zero_shot":
        pass
    else:
        pass
    return built


if __name__ == "__main__":
    main()
