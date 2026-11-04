from __future__ import annotations
# SchNet -- Schutt, Arbabzadah, Chmiela, Tkatchenko, Muller, NeurIPS 2017, arXiv:1706.08566
# PaiNN -- Schutt, Unke, Gastegger, ICML 2021, arXiv:2102.03150 
# ANI-2x -- Devereux/Smith/Isayev et al., JCTC 2020 
# AIMNet2 -- Anstine, Zubatyuk, Isayev, Chem. Sci. 16, 10228 (2025)

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
    reseed_dataloader_for_epoch,
    capture_rng_state,
    restore_rng_state,
)
from baselines.provenance.env_manifest import write as write_env_manifest


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument("--model", required=True, choices=["schnet", "painn", "ani2x", "aimnet2"])
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=128, help="n_atom_basis for schnet/painn")
    p.add_argument("--n-interactions", type=int, default=3, help="message-passing depth for schnet/painn")
    p.add_argument("--cutoff", type=float, default=5.0, help="angstrom cutoff radius for schnet/painn")
    p.add_argument("--n-rbf", type=int, default=20, help="number of Gaussian radial basis functions for schnet/painn")
    p.add_argument(
        "--ani2x-ensemble-size", type=int, default=1,
        help="number of ANI-2x sub-networks to train (public ANI-2x is an 8-net ensemble; default trains a single net from scratch, cheaper)",
    )
    p.add_argument(
        "--ani2x-pretrained", action="store_true",
        help="fine-tune from the public ANI-2x weights instead of training the same architecture from scratch",
    )
    p.add_argument(
        "--aimnet2-config", type=str, default=None,
        help="path to an aimnetcentral model yaml; defaults to the submodule's aimnet/models/aimnet2.yaml",
    )
    return p


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


def build_schnet_painn_model(model_name: str, args: argparse.Namespace) -> torch.nn.Module:
    import schnetpack.nn as snn
    import schnetpack.properties as properties
    from schnetpack.representation import SchNet, PaiNN
    from schnetpack.atomistic import PairwiseDistances, Atomwise
    from schnetpack.atomistic.response import Forces
    from schnetpack.model import NeuralNetworkPotential

    radial_basis = snn.GaussianRBF(n_rbf=args.n_rbf, cutoff=args.cutoff)
    cutoff_fn = snn.CosineCutoff(args.cutoff)
    if model_name == "schnet":
        representation = SchNet(
            n_atom_basis=args.hidden_dim, n_interactions=args.n_interactions,
            radial_basis=radial_basis, cutoff_fn=cutoff_fn,
        )
    else:
        representation = PaiNN(
            n_atom_basis=args.hidden_dim, n_interactions=args.n_interactions,
            radial_basis=radial_basis, cutoff_fn=cutoff_fn,
        )
    energy_head = Atomwise(n_in=args.hidden_dim, output_key=properties.energy)
    force_head = Forces(energy_key=properties.energy, force_key=properties.forces)
    return NeuralNetworkPotential(
        representation=representation,
        input_modules=[PairwiseDistances()],
        output_modules=[energy_head, force_head],
    )


def _schnet_painn_loss(model: torch.nn.Module, batch: dict) -> torch.Tensor:
    import schnetpack.properties as properties

    # targets must be recorded before calling the model, or `batch[properties.energy]` becomes the model's prediction
    energy_target = batch[properties.energy].clone()
    forces_target = batch[properties.forces].clone()
    out = model(batch)
    e_loss = F.mse_loss(out[properties.energy], energy_target)
    f_loss = F.mse_loss(out[properties.forces], forces_target)
    return e_loss + f_loss


def _train_schnet_painn(model_name: str, cfg: CommonConfig, args: argparse.Namespace, paths: RunPaths) -> None:
    device = torch.device(cfg.device)
    model = build_schnet_painn_model(model_name, args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_epoch, global_step = 0, 0
    if cfg.resume:
        start_epoch, global_step = _load_checkpoint(cfg.resume, model, optimizer)
    if _is_distributed():
        model = _wrap_ddp(model, device)

    train_loader = _build_dataloader(cfg, "train")
    for epoch in range(start_epoch, args.epochs):
        reseed_dataloader_for_epoch(train_loader, cfg.seed, epoch)
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            # TODO: train_spice.py found bf16 corrupted forces for its own architecture, so check forces against an fp32 run before running --amp
            with torch.autocast(device_type=device.type, enabled=cfg.amp):
                loss = _schnet_painn_loss(model, batch)
            loss.backward()
            optimizer.step()
            global_step += 1
        _save_checkpoint(paths.checkpoint(f"{model_name}_epoch{epoch}"), epoch, global_step, model, optimizer, cfg, args)

def build_ani2x_model(args: argparse.Namespace) -> torch.nn.Module:
    from torchani.arch import Assembler
    from torchani.utils import SYMBOLS_2X

    if args.ani2x_pretrained:
        from torchani.models import ANI2x
        model = ANI2x(periodic_table_index=True)
        model.requires_grad_(True)
        return model

    asm = Assembler(periodic_table_index=True)
    asm.set_symbols(SYMBOLS_2X)
    asm.set_global_cutoff_fn("cosine")
    asm.set_aev_computer(radial="ani2x", angular="ani2x", strategy="pyaev")
    asm.set_atomic_networks(ctor="ani2x")
    asm.set_neighborlist("all_pairs")
    return asm.assemble(args.ani2x_ensemble_size)


def _ani2x_loss(model: torch.nn.Module, batch: dict, device: torch.device) -> torch.Tensor:
    species = batch["species"].to(device, non_blocking=True)
    coordinates = batch["coordinates"].to(device, non_blocking=True).requires_grad_(True)
    energies = model((species, coordinates)).energies
    (neg_forces,) = torch.autograd.grad(energies.sum(), coordinates, create_graph=True)
    e_loss = F.mse_loss(energies, batch["energies"].to(device, non_blocking=True))
    f_loss = F.mse_loss(-neg_forces, batch["forces"].to(device, non_blocking=True))
    return e_loss + f_loss


def _train_ani2x(cfg: CommonConfig, args: argparse.Namespace, paths: RunPaths) -> None:
    device = torch.device(cfg.device)
    model = build_ani2x_model(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_epoch, global_step = 0, 0
    if cfg.resume:
        start_epoch, global_step = _load_checkpoint(cfg.resume, model, optimizer)
    if _is_distributed():
        model = _wrap_ddp(model, device)

    train_loader = _build_dataloader(cfg, "train")
    for epoch in range(start_epoch, args.epochs):
        reseed_dataloader_for_epoch(train_loader, cfg.seed, epoch)
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=cfg.amp):
                loss = _ani2x_loss(model, batch, device)
            loss.backward()
            optimizer.step()
            global_step += 1
        _save_checkpoint(paths.checkpoint(f"ani2x_epoch{epoch}"), epoch, global_step, model, optimizer, cfg, args)


def build_aimnet2_model(args: argparse.Namespace) -> torch.nn.Module:
    import aimnet
    from aimnet.config import load_yaml, build_module

    # Building aimnet model sub-graph directly from the submodule's packaged aimnet2.yaml via aimnet.config.build_module
    cfg_path = args.aimnet2_config or os.path.join(os.path.dirname(aimnet.__file__), "models", "aimnet2.yaml")
    model_cfg = load_yaml(cfg_path)
    return build_module(model_cfg)


def _aimnet2_loss(model: torch.nn.Module, batch: dict, device: torch.device) -> torch.Tensor:
    # TODO: double-check output keys for forces and energies 
    data = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k != "energy_target" and k != "forces_target"}
    data["coord"] = data["coord"].requires_grad_(True)
    data = model.prepare_input(data)
    out = model(data)
    (neg_forces,) = torch.autograd.grad(out["energy"].sum(), data["coord"], create_graph=True)
    e_loss = F.mse_loss(out["energy"], batch["energy_target"].to(device, non_blocking=True))
    f_loss = F.mse_loss(-neg_forces, batch["forces_target"].to(device, non_blocking=True))
    return e_loss + f_loss


def _train_aimnet2(cfg: CommonConfig, args: argparse.Namespace, paths: RunPaths) -> None:
    device = torch.device(cfg.device)
    model = build_aimnet2_model(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_epoch, global_step = 0, 0
    if cfg.resume:
        start_epoch, global_step = _load_checkpoint(cfg.resume, model, optimizer)
    if _is_distributed():
        model = _wrap_ddp(model, device)

    train_loader = _build_dataloader(cfg, "train")
    for epoch in range(start_epoch, args.epochs):
        reseed_dataloader_for_epoch(train_loader, cfg.seed, epoch)
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=cfg.amp):
                loss = _aimnet2_loss(model, batch, device)
            loss.backward()
            optimizer.step()
            global_step += 1
        _save_checkpoint(paths.checkpoint(f"aimnet2_epoch{epoch}"), epoch, global_step, model, optimizer, cfg, args)


def main() -> None:
    args = build_parser().parse_args()
    cfg = CommonConfig.from_args(args)
    paths = RunPaths(cfg.run_dir).ensure()
    set_seed(cfg.seed, cfg.deterministic)
    _set_thread_count()
    write_env_manifest(cfg.run_dir)

    if args.model in ("schnet", "painn"):
        _train_schnet_painn(args.model, cfg, args, paths)
    elif args.model == "ani2x":
        _train_ani2x(cfg, args, paths)
    elif args.model == "aimnet2":
        _train_aimnet2(cfg, args, paths)
    else:
        raise ValueError(f"unknown model {args.model!r}")


if __name__ == "__main__":
    main()
