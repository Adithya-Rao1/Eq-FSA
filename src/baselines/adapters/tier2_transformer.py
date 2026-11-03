from __future__ import annotations
# Equiformer -- Liao & Smidt, "Equiformer: Equivariant Graph Attention Transformer for 3D Atomistic Graphs," ICLR 2023. 
# EquiformerV2 -- Liao, Wood, Das, Smidt, "EquiformerV2: Improved Equivariant Transformer for Scaling to Higher-Degree Representations," ICLR 2024, arXiv:2306.12059, 
# dispatched through the vendored fairchem submodule at src/baselines/codebases/fairchem, pinned to tag fairchem_core-2.25.0, commit baab686053cf52c143c387580b825233419942b2 (2026-10-27).

import argparse
import os

from baselines.config.schema import CommonConfig, RunPaths, add_common_args
from baselines.provenance.env_manifest import write as write_env_manifest
from baselines.provenance.seeding import dataloader_generator, seed_worker, set_seed

_FAIRCHEM_PIN_COMMIT = "baab686053cf52c143c387580b825233419942b2"  # tag fairchem_core-2.25.0, 2026-10-27
_FAIRCHEM_PIN_TAG = "fairchem_core-2.25.0"

TIER2_MODELS = ("equiformer", "equiformerv2")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument("--model", required=True, choices=list(TIER2_MODELS))
    return p


def _init_single_node_ddp() -> tuple[int, int, int]:
    # torchrun sets these; absence runs single-process (no DDP).
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


def _build_equiformer(cfg: CommonConfig):
    # Original Equiformer (v1) is not pip-installable
    raise NotImplementedError(
        "To support --model equiformer, add `git submodule add https://github.com/atomicarchitects/"
        "equiformer src/baselines/codebases/equiformer` and wire a dispatch path here first."
    )


def _build_equiformerv2(cfg: CommonConfig):
    try:
        import fairchem.core  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "fairchem is not installed. Run `pip install -e src/baselines/codebases/fairchem` "
            "(or `pip install fairchem-core`) in the remote GPU environment before dispatching "
            "EquiformerV2."
        ) from e
    try:
        from fairchem.core.models.equiformer_v2.equiformer_v2 import EquiformerV2  # type: ignore
    except ImportError as e:
        raise ImportError(
            "EquiformerV2 model code is absent at the pinned fairchem commit "
            f"({_FAIRCHEM_PIN_COMMIT}, tag {_FAIRCHEM_PIN_TAG}): upstream deleted "
            "src/fairchem/core/models/equiformer_v2/ when it pivoted to the UMA/eSEN model family "
            "in the fairchem_core 2.x line (this submodule is pinned there because Tier 3's UMA "
            "dispatch needs 2.x). To run literal EquiformerV2, use a separate fairchem checkout "
            "pinned to tag fairchem_core-1.10.0 (the last tag confirmed to contain "
            "models/equiformer_v2/equiformer_v2.py, pre-dating any UMA support) -- a single pinned "
            "commit cannot serve both models at once."
        ) from e
    return EquiformerV2


_DISPATCH = {
    "equiformer": _build_equiformer,
    "equiformerv2": _build_equiformerv2,
}


def main() -> None:
    args = build_parser().parse_args()
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

    model_cls = _DISPATCH[args.model](cfg)
    return model_cls


if __name__ == "__main__":
    main()
