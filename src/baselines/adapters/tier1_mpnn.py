from __future__ import annotations
# Tier 1 (core equivariant message-passing) MLIP baselines: a thin CLI -> native-trainer adapter,
# one function per model. NequIP -- Batzner et al., Nat. Commun. 13, 2453 (2022),
# arXiv:2101.03164 (full E(3)-equivariant tensor-feature message passing), via the nequip
# submodule (pinned to tag v0.19.1, commit 27d9d218, 2026-08-30). Allegro -- Musaelian et al.,
# Nat. Commun. 2023, arXiv:2204.05249 (strictly local, no message passing, scales to huge
# systems), via the allegro submodule (pinned to tag v0.8.3, commit 8bd074c8, 2026-05-27) -- built
# on the same nequip.data.AtomicDataDict / nequip.nn.GraphModel plumbing as NequIP itself. MACE --
# Batatia et al., NeurIPS 2022, arXiv:2206.07697 (higher-body-order equivariant messages, only 2
# MP layers), via the mace submodule (pinned to tag v0.3.17, commit e37c8583, 2026-10-19).

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
    # Single-node only: torchrun --standalone --nproc_per_node=$SLURM_GPUS_ON_NODE sets RANK/
    # WORLD_SIZE/LOCAL_RANK; no multi-node rendezvous is wired up here on purpose.
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
    ckpt = torch.load(path, map_location="cpu")
    raw_model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    restore_rng_state(ckpt["rng_state"])
    return ckpt["epoch"], ckpt["global_step"]


def _build_dataloader(cfg: CommonConfig, split: str):
    # baselines.datasets is owned by a sibling agent on a separate branch and does not exist in
    # this worktree; this is the agreed integration seam. It is assumed to yield batches already
    # collated into each library's native format: nequip.data.AtomicDataDict-shaped dicts for
    # nequip/allegro (both share that plumbing), and mace.data.AtomicData-derived
    # torch_geometric Batch objects for mace. Exact collation is the datasets module's
    # responsibility, not this adapter's.
    from baselines.datasets.api import load_split  # type: ignore[import-not-found]

    dataset = load_split(cfg.dataset, cfg.split, split)
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


# ---------------------------------------------------------------------------
# NequIP
# ---------------------------------------------------------------------------

def _ensure_nequip_global_state(cfg: CommonConfig | None) -> None:
    # nequip >=0.19 refuses to build ANY model (NequIPGNNModel or AllegroModel, both go through
    # nequip.model.utils) until nequip.utils.global_state.set_global_state() has run once; that
    # call also reseeds everything via pytorch-lightning's seed_everything(123, ...) as a side
    # effect, which would silently stomp this repo's own set_seed(cfg.seed) if called afterwards.
    # Call it, then re-apply our own seed so cfg.seed (not nequip's hardcoded 123) wins.
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
        seed=0,  # nequip's own init seed knob; global reproducibility still comes from set_seed()
        model_dtype="float32",
        avg_num_neighbors=args.avg_num_neighbors,
    )


def _nequip_allegro_loss(model: torch.nn.Module, batch: dict, device: torch.device) -> torch.Tensor:
    from nequip.data import AtomicDataDict

    data = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    # Same pitfall as schnetpack (see tier0_legacy._schnet_painn_loss): nequip/allegro's
    # AtomicDataDict ground truth and model output share the same TOTAL_ENERGY_KEY/FORCE_KEY, and
    # GraphModel writes its prediction back into the same dict -- snapshot the targets first.
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
            # AMP is opt-in (default off) and NOT numerically verified for this model's
            # forces-via-autograd path -- train_spice.py found bf16 corrupted forces for its own
            # architecture, so check forces against an fp32 run before trusting --amp here.
            with torch.autocast(device_type=device.type, enabled=cfg.amp):
                loss = _nequip_allegro_loss(model, batch, device)
            loss.backward()
            optimizer.step()
            global_step += 1
        _save_checkpoint(paths.checkpoint(f"{model_name}_epoch{epoch}"), epoch, global_step, model, optimizer, cfg, args)


# ---------------------------------------------------------------------------
# Allegro
# ---------------------------------------------------------------------------

def build_allegro_model(args: argparse.Namespace, cfg: CommonConfig | None = None) -> torch.nn.Module:
    from allegro.model import AllegroModel

    _ensure_nequip_global_state(cfg)  # Allegro is built on the same nequip global-state machinery
    return AllegroModel(
        r_max=args.cutoff,
        type_names=_type_names(args),
        l_max=args.l_max,
        num_layers=args.num_layers,
        num_scalar_features=args.hidden_dim,
        num_tensor_features=max(args.hidden_dim // 4, 1),
        seed=0,  # allegro's own init seed knob; global reproducibility still comes from set_seed()
        model_dtype="float32",
        avg_num_neighbors=args.avg_num_neighbors,
    )


# ---------------------------------------------------------------------------
# MACE
# ---------------------------------------------------------------------------

def build_mace_model(args: argparse.Namespace) -> torch.nn.Module:
    # Ambiguous-mapping note: unlike nequip/allegro's config-style builder functions, mace.modules
    # MACE is instantiated with a large, low-level kwarg surface (atomic_energies and
    # avg_num_neighbors in particular are normally fit from real training-set statistics, not
    # guessed). The placeholders below (zero self-energies, args.avg_num_neighbors) are only
    # adequate for a structural/gradient-flow smoke test, not for a real run -- recompute them
    # from the actual split (mace.modules.utils.compute_average_E0s /
    # compute_avg_num_neighbors) before training for real.
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
    # Same pitfall as schnetpack/nequip above: snapshot targets before the forward pass in case
    # MACE's output dict construction reuses/overwrites the input dict's "energy"/"forces" keys.
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
