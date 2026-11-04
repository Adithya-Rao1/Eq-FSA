from __future__ import annotations
# NequIP -- Batzner et al., Nat. Commun. 13, 2453 (2022), arXiv:2101.03164 submodule (pinned to tag v0.19.1, commit 27d9d218, 2026-08-30). 
# Allegro -- Musaelian et al., Nat. Commun. 2023, arXiv:2204.05249 (pinned to tag v0.8.3, commit 8bd074c8, 2026-05-27) 
# MACE -- Batatia et al., NeurIPS 2022, arXiv:2206.07697 (pinned to tag v0.3.17, commit e37c8583, 2026-10-19).

import argparse
import dataclasses
import os

import torch
import torch.nn.functional as F

from baselines.config.schema import add_common_args, CommonConfig, RunPaths
from baselines.provenance.seeding import (
    set_seed,
    seed_worker,
    dataloader_generator,
    capture_rng_state,
    restore_rng_state,
)
from baselines.provenance.env_manifest import write as write_env_manifest


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument("--model", required=True, choices=["nequip", "allegro", "mace"])
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--cutoff", type=float, default=5.0, help="r_max, angstrom")
    p.add_argument("--num-layers", type=int, default=2, help="MP/Allegro-layer depth")
    p.add_argument("--l-max", type=int, default=1, help="max spherical-harmonic rotation order")
    p.add_argument("--hidden-dim", type=int, default=32, help="feature multiplicity (num_features / num_scalar_features)")
    p.add_argument(
        "--type-names", type=str, default="H,C,N,O",
        help="comma-separated chemical-symbol vocabulary the model's type embedding is built over",
    )
    p.add_argument(
        "--avg-num-neighbors", type=float, default=8.0,
        help="nequip/allegro/mace all normalize messages by this; the default is only a placeholder"
        " -- recompute it from the real training split (e.g. nequip.data.compute_avg_num_neighbors)"
        " before trusting a real run",
    )
    return p


def _type_names(args: argparse.Namespace) -> list[str]:
    return [s.strip() for s in args.type_names.split(",") if s.strip()]


def _set_thread_count() -> None:
    n = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    torch.set_num_threads(n)


def _is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _wrap_ddp(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    # Single-node only: torchrun --standalone --nproc_per_node=$SLURM_GPUS_ON_NODE sets RANK/WORLD_SIZE/LOCAL_RANK
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    return DDP(model, device_ids=[device.index] if device.type == "cuda" else None)


def _save_checkpoint(path: str, epoch: int, global_step: int, model, optimizer, cfg: CommonConfig, args: argparse.Namespace) -> None:
    raw_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    torch.save({
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": {**dataclasses.asdict(cfg), **vars(args)},
        "rng_state": capture_rng_state(),  # closes the gap train_spice.py's checkpoint format left open
    }, path)


def _load_checkpoint(path: str, model, optimizer) -> tuple[int, int]:
    raw_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    ckpt = torch.load(path, map_location="cpu", weights_only=True)  # safe unpickler: the checkpoint holds only tensors and builtins
    raw_model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    restore_rng_state(ckpt["rng_state"])
    return ckpt["epoch"], ckpt["global_step"]


def _build_dataloader(cfg: CommonConfig, split: str):
    from baselines.adapters.dataset_loading import load_split

    dataset = load_split(cfg.dataset, cfg.split, split, cfg.data_root)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=(split == "train"),
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
        worker_init_fn=seed_worker,
        generator=dataloader_generator(cfg.seed),
    )


def _ensure_nequip_global_state(cfg: CommonConfig | None) -> None:
    # nequip >=0.19 need to run nequip.utils.global_state.set_global_state() once before any model.
    from nequip.utils.global_state import set_global_state, global_state_initialized

    if not global_state_initialized():
        set_global_state()
        if cfg is not None:
            set_seed(cfg.seed, cfg.deterministic)


def build_nequip_model(args: argparse.Namespace, cfg: CommonConfig | None = None) -> torch.nn.Module:
    from nequip.model import NequIPGNNModel

    _ensure_nequip_global_state(cfg)
    return NequIPGNNModel(
        r_max=args.cutoff,
        type_names=_type_names(args),
        num_layers=args.num_layers,
        l_max=args.l_max,
        num_features=args.hidden_dim,
        seed=0, 
        model_dtype="float32",
        avg_num_neighbors=args.avg_num_neighbors,
    )


def _nequip_allegro_loss(model: torch.nn.Module, batch: dict, device: torch.device) -> torch.Tensor:
    from nequip.data import AtomicDataDict

    data = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    # capture target first to prevent overwriting it as nequip/allegro's AtomicDataDict ground truth and model output share the same TOTAL_ENERGY_KEY/FORCE_KEY
    energy_target = data[AtomicDataDict.TOTAL_ENERGY_KEY].clone()
    force_target = data[AtomicDataDict.FORCE_KEY].clone()
    out = model(data)
    e_loss = F.mse_loss(out[AtomicDataDict.TOTAL_ENERGY_KEY], energy_target)
    f_loss = F.mse_loss(out[AtomicDataDict.FORCE_KEY], force_target)
    return e_loss + f_loss


def _train_nequip_or_allegro(model_name: str, cfg: CommonConfig, args: argparse.Namespace, paths: RunPaths) -> None:
    device = torch.device(cfg.device)
    model = (build_nequip_model(args, cfg) if model_name == "nequip" else build_allegro_model(args, cfg)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_epoch, global_step = 0, 0
    if cfg.resume:
        start_epoch, global_step = _load_checkpoint(cfg.resume, model, optimizer)
    if _is_distributed():
        model = _wrap_ddp(model, device)

    train_loader = _build_dataloader(cfg, "train")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            # TODO: test against fp32 before trusting amp
            with torch.autocast(device_type=device.type, enabled=cfg.amp):
                loss = _nequip_allegro_loss(model, batch, device)
            loss.backward()
            optimizer.step()
            global_step += 1
        _save_checkpoint(paths.checkpoint(f"{model_name}_epoch{epoch}"), epoch, global_step, model, optimizer, cfg, args)


def build_allegro_model(args: argparse.Namespace, cfg: CommonConfig | None = None) -> torch.nn.Module:
    from allegro.model import AllegroModel

    _ensure_nequip_global_state(cfg) 
    return AllegroModel(
        r_max=args.cutoff,
        type_names=_type_names(args),
        l_max=args.l_max,
        num_layers=args.num_layers,
        num_scalar_features=args.hidden_dim,
        num_tensor_features=max(args.hidden_dim // 4, 1),
        seed=0, 
        model_dtype="float32",
        avg_num_neighbors=args.avg_num_neighbors,
    )


def build_mace_model(args: argparse.Namespace) -> torch.nn.Module:
    # TODO: recompute self-energies and args.avg_num_neighbors from the actual split (mace.modules.utils.compute_average_E0s /compute_avg_num_neighbors) before real training.
    import numpy as np
    from e3nn import o3
    from mace.modules import MACE, interaction_classes

    type_names = _type_names(args)
    num_elements = len(type_names)
    hidden_irreps = o3.Irreps(f"{args.hidden_dim}x0e + {args.hidden_dim}x1o")
    return MACE(
        r_max=args.cutoff,
        num_bessel=8,
        num_polynomial_cutoff=6,
        max_ell=args.l_max + 1,
        interaction_cls=interaction_classes["RealAgnosticResidualInteractionBlock"],
        interaction_cls_first=interaction_classes["RealAgnosticInteractionBlock"],
        num_interactions=args.num_layers,
        num_elements=num_elements,
        hidden_irreps=hidden_irreps,
        MLP_irreps=o3.Irreps("16x0e"),
        atomic_energies=np.zeros(num_elements, dtype=float),
        avg_num_neighbors=args.avg_num_neighbors,
        atomic_numbers=list(range(1, num_elements + 1)),
        correlation=3,
        gate=F.silu,
    )


def _mace_loss(model: torch.nn.Module, batch: dict, device: torch.device) -> torch.Tensor:
    data = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    # Same problem as schnetpack/nequip above
    energy_target = data["energy"].clone()
    force_target = data["forces"].clone()
    out = model(data, training=True)
    e_loss = F.mse_loss(out["energy"], energy_target)
    f_loss = F.mse_loss(out["forces"], force_target)
    return e_loss + f_loss


def _train_mace(cfg: CommonConfig, args: argparse.Namespace, paths: RunPaths) -> None:
    device = torch.device(cfg.device)
    model = build_mace_model(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_epoch, global_step = 0, 0
    if cfg.resume:
        start_epoch, global_step = _load_checkpoint(cfg.resume, model, optimizer)
    if _is_distributed():
        model = _wrap_ddp(model, device)

    train_loader = _build_dataloader(cfg, "train")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=cfg.amp):
                loss = _mace_loss(model, batch, device)
            loss.backward()
            optimizer.step()
            global_step += 1
        _save_checkpoint(paths.checkpoint(f"mace_epoch{epoch}"), epoch, global_step, model, optimizer, cfg, args)


def main() -> None:
    args = build_parser().parse_args()
    cfg = CommonConfig.from_args(args)
    paths = RunPaths(cfg.run_dir).ensure()
    set_seed(cfg.seed, cfg.deterministic)
    _set_thread_count()
    write_env_manifest(cfg.run_dir)

    if args.model in ("nequip", "allegro"):
        _train_nequip_or_allegro(args.model, cfg, args, paths)
    elif args.model == "mace":
        _train_mace(cfg, args, paths)
    else:
        raise ValueError(f"unknown model {args.model!r}")


if __name__ == "__main__":
    main()
