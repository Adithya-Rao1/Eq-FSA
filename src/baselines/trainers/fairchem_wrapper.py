from __future__ import annotations

# fairchem-core 1.2.0:
#  - non-periodic graphs go through torch_geometric.nn.radius_graph, so every structure is placed in a vacuum cell padded by `pad = cutoff` on each side, which keeps
#    periodic images farther apart than the cutoff and therefore changes no energy or force. The same padding is applied inside the ASE calculator used for evaluation 
#  - per-element reference energies and normalizers are fitted by the trainer from the training set
#  - avg_num_nodes / avg_degree default to OC20 constants and measured from the training frames here.

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import yaml

TIMESTAMP_ID = "run"
TRAIN_DONE = "TRAIN_DONE"
ENERGY, FORCES = "energy", "forces"


def checkpoint_dir(run_dir: str) -> str:
    return os.path.join(run_dir, "checkpoints", TIMESTAMP_ID)


def best_checkpoint_path(run_dir: str) -> str:
    return os.path.join(checkpoint_dir(run_dir), "best_checkpoint.pt")


def read_split_record(data_dir: str) -> dict:
    with open(os.path.join(data_dir, "split.json")) as f:
        rec = json.load(f)
    missing = [k for k in ("species", "fixed_composition", "splits") if k not in rec]
    if missing:
        raise KeyError(f"{data_dir}/split.json lacks {missing}; re-export with the current export_splits")
    return rec


def pad_cell(atoms, pad: float):
    # Returns a copy in an orthorhombic vacuum cell with `pad` of empty space on every side and pbc on. 
    # Atoms stay at their relative positions (rigid translation only), so any translation-invariant model sees the identical geometry.
    out = atoms.copy()
    pos = out.get_positions()
    lo, hi = pos.min(axis=0), pos.max(axis=0)
    out.set_positions(pos - lo + pad)
    out.set_cell(np.diag(hi - lo + 2.0 * pad))
    out.set_pbc(True)
    return out


def measure_graph_stats(frames, cutoff: float, max_frames: int = 1000) -> dict:
    from ase.neighborlist import neighbor_list

    frames = frames[:max_frames]
    degrees = []
    for a in frames:
        i = neighbor_list("i", a, cutoff)  # non-periodic, so no image contributions
        degrees.append(len(i) / len(a))
    return {"avg_num_nodes": float(np.mean([len(a) for a in frames])), "avg_degree": float(np.mean(degrees))}


def build_databases(data_dir: str, out_dir: str, cutoff: float) -> dict:
    import ase.db
    from ase.calculators.singlepoint import SinglePointCalculator

    from baselines.export.extxyz import ENERGY_KEY, FORCES_KEY, read_extxyz

    rec = read_split_record(data_dir)
    frames = {name: read_extxyz(os.path.join(data_dir, f"{name}.xyz")) for name in ("train", "val", "test")}
    has_forces = all(FORCES_KEY in a.arrays for k in frames for a in frames[k])
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for name, items in frames.items():
        path = os.path.join(out_dir, f"{name}.db")
        if os.path.exists(path):
            os.remove(path)  # rebuilt from the exported files every time
        with ase.db.connect(path) as db:
            for a in items:
                padded = pad_cell(a, cutoff)
                results = {"energy": float(a.info[ENERGY_KEY])}
                if has_forces:
                    results["forces"] = np.asarray(a.arrays[FORCES_KEY], dtype=np.float64)
                clean = padded.copy()
                clean.info.clear()
                for k in [k for k in clean.arrays if k not in ("numbers", "positions")]:
                    del clean.arrays[k]
                clean.calc = SinglePointCalculator(clean, **results)
                db.write(clean)
        paths[name] = path
    return {"paths": paths, "has_forces": has_forces, "fixed_composition": rec["fixed_composition"],
            "counts": {k: len(v) for k, v in frames.items()}, **measure_graph_stats(frames["train"], cutoff)}


def build_config(
    summary: dict, run_dir: str, *, device: str = "cuda", epochs: int = 1, batch_size: int = 8, lr: float = 4e-4,
    cutoff: float = 5.0, max_neighbors: int = 50, energy_coeff: float = 20.0, forces_coeff: float = 100.0, ema_decay: float = 0.99,
    model_overrides: dict | None = None, optim_overrides: dict | None = None,
) -> dict:
    has_forces, fixed = summary["has_forces"], summary["fixed_composition"]
    a2g = {"r_energy": True, "r_forces": has_forces, "r_edges": False, "r_pbc": True}
    transforms: dict = {"normalizer": {"fit": {"targets": [ENERGY] + ([FORCES] if has_forces else []), "batch_size": batch_size}}}
    # override_values must always be a dict: fit_normalizers defaults it to None and then evaluates `target in override_values` (1.2.0 bug).
    transforms["normalizer"]["fit"]["override_values"] = {FORCES: {"mean": 0.0}} if has_forces else {}  # forces: zero mean by symmetry
    if not fixed:
        transforms["element_references"] = {"fit": {"targets": [ENERGY], "batch_size": batch_size}}
    split = lambda name: {"format": "ase_db", "src": summary["paths"][name], "a2g_args": a2g}  # noqa: E731
    backbone = {
        "model": "equiformer_v2_backbone", "use_pbc": True, "use_pbc_single": False, "regress_forces": has_forces, "otf_graph": True,
        "max_neighbors": max_neighbors, "max_radius": cutoff, "max_num_elements": 90, "num_layers": 12, "sphere_channels": 128,
        "attn_hidden_channels": 64, "num_heads": 8, "attn_alpha_channels": 64, "attn_value_channels": 16, "ffn_hidden_channels": 128,
        "norm_type": "layer_norm_sh", "lmax_list": [4], "mmax_list": [2], "grid_resolution": 18, "edge_channels": 128,
        "num_distance_basis": 128, "distance_function": "gaussian", "attn_activation": "silu", "use_s2_act_attn": False,
        "ffn_activation": "silu", "use_gate_act": False, "use_grid_mlp": True, "use_sep_s2_act": True,
        "alpha_drop": 0.1, "drop_path_rate": 0.05, "proj_drop": 0.0, "weight_init": "uniform",
        "avg_num_nodes": summary["avg_num_nodes"], "avg_degree": summary["avg_degree"],
    }
    backbone.update(model_overrides or {})
    heads = {ENERGY: {"module": "equiformer_v2_energy_head"}}
    outputs = {ENERGY: {"shape": 1, "level": "system", "property": ENERGY}}
    losses = [{ENERGY: {"fn": "mae", "coefficient": energy_coeff}}]
    metrics = {ENERGY: ["mae"]}
    if has_forces:
        heads[FORCES] = {"module": "equiformer_v2_force_head"}  # direct force prediction, as in the EquiformerV2 paper (forces are not -dE/dx)
        outputs[FORCES] = {"irrep_dim": 1, "level": "atom", "property": FORCES, "train_on_free_atoms": True, "eval_on_free_atoms": True}
        losses.append({FORCES: {"fn": "l2mae", "coefficient": forces_coeff}})
        metrics[FORCES] = ["mae"]
    # fairchem 1.2.0's LRScheduler reads FLAT keys from `optim`
    total_steps = max(1, epochs * -(-summary["counts"]["train"] // batch_size))
    optim = {
        "batch_size": batch_size, "eval_batch_size": batch_size, "num_workers": 0, "lr_initial": lr, "optimizer": "AdamW",
        "optimizer_params": {"weight_decay": 1e-3}, "warmup_steps": max(1, min(1000, total_steps // 10)), "warmup_factor": 0.2,
        "lr_gamma": 0.5, "lr_milestones": [int(f * total_steps) for f in (0.6, 0.8, 0.95)] if total_steps >= 500 else [],
        "max_epochs": epochs, "clip_grad_norm": 100, "ema_decay": ema_decay, "eval_every": None,
    }
    optim.update(optim_overrides or {})
    if optim["eval_every"] is None:
        del optim["eval_every"]  # library default: once per epoch
    return {
        "trainer": "forces" if has_forces else "energy",
        "dataset": {"train": {**split("train"), "transforms": transforms}, "val": split("val"), "test": split("test")},
        "outputs": outputs, "loss_functions": losses,
        "evaluation_metrics": {"metrics": metrics, "primary_metric": "forces_mae" if has_forces else "energy_mae"},
        "model": {"name": "hydra", "backbone": backbone, "heads": heads},
        "optim": optim, "logger": "tensorboard",
        # the library saves its own test-set predictions in float16 by default, which overflows to -inf beyond |65504| eV
        # Has ~8 eV resolution near 1e4 eV; evaluation here goes through the calculator.
        "task": {"prediction_dtype": "float32"},
    }


def write_config(cfg: dict, run_dir: str) -> str:
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "config.yml")
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path


def build_command(cfg_path: str, run_dir: str, seed: int, device: str, resume: bool) -> list[str]:
    cmd = [sys.executable, "-c", "from fairchem.core._cli import main; main()", "--mode", "train", "--config-yml", cfg_path,
           "--run-dir", os.path.abspath(run_dir), "--identifier", "", "--timestamp-id", TIMESTAMP_ID, "--seed", str(seed),
           "--print-every", "10"]
    if not device.startswith("cuda"):
        cmd.append("--cpu")
    if resume:
        cmd += ["--checkpoint", os.path.join(checkpoint_dir(run_dir), "checkpoint.pt")]
    return cmd


def run(data_dir: str, run_dir: str, seed: int, dry_run: bool = False, cutoff: float = 5.0, device: str = "cuda", **kw) -> int:
    marker = os.path.join(run_dir, TRAIN_DONE)
    if os.path.exists(marker):
        print(f"[equiformerv2] {marker} exists; nothing to do")
        return 0
    os.makedirs(run_dir, exist_ok=True)
    summary = build_databases(data_dir, os.path.join(run_dir, "fc_data"), cutoff)
    cfg = build_config(summary, run_dir, device=device, cutoff=cutoff, **kw)
    cfg_path = write_config(cfg, run_dir)
    with open(os.path.join(run_dir, "wrapper.json"), "w") as f:
        json.dump({"cutoff": cutoff, "has_forces": summary["has_forces"]}, f)
    resume = os.path.exists(os.path.join(checkpoint_dir(run_dir), "checkpoint.pt"))
    cmd = build_command(cfg_path, run_dir, seed, device, resume)
    print("[equiformerv2] " + " ".join(cmd))
    if dry_run:
        return 0
    rc = subprocess.call(cmd)
    if rc == 0:
        open(marker, "w").close()
    return rc


def load_calculator(run_dir: str, device: str = "cpu"):
    from ase.calculators.calculator import Calculator, all_changes
    from fairchem.core import OCPCalculator

    with open(os.path.join(run_dir, "wrapper.json")) as f:
        meta = json.load(f)
    # The checkpoint stores a legacy trainer name ("s2ef") that this release's registry rejects, so the trainer is passed explicitly.
    inner = OCPCalculator(checkpoint_path=best_checkpoint_path(run_dir), cpu=not device.startswith("cuda"), seed=0,
                          trainer="forces" if meta["has_forces"] else "energy", cutoff=int(np.ceil(meta["cutoff"])))
    pad = meta["cutoff"]

    class PaddedCellCalculator(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            padded = pad_cell(atoms, pad)
            padded.calc = inner
            self.results = {"energy": float(padded.get_potential_energy()),
                            "forces": np.asarray(padded.get_forces(), dtype=np.float64)} if meta["has_forces"] else \
                {"energy": float(padded.get_potential_energy())}

    calc = PaddedCellCalculator()
    calc.inner = inner  # tests can inspect the underlying trainer / model dtype
    return calc


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train EquiformerV2 with fairchem's native trainer on exported extxyz splits.")
    p.add_argument("--data-dir", required=True, help="directory written by export/export_splits.py")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--cutoff", type=float, default=5.0)
    p.add_argument("--ema-decay", type=float, default=0.99, help="EMA weight decay; the saved/evaluated weights lag by ~1/(1-decay) steps (0.999 left forces at 546 meV/A after 5 epochs on rMD17 aspirin)")
    p.add_argument("--energy-coeff", type=float, default=20.0, help="2 is the OC20 value; on rMD17 aspirin 2 left the energy at the predict-the-mean baseline, 20 learned it")
    p.add_argument("--forces-coeff", type=float, default=100.0)
    p.add_argument("--dry-run", action="store_true", help="build databases and config, print the command, do not train")
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return run(a.data_dir, a.run_dir, a.seed, a.dry_run, cutoff=a.cutoff, device=a.device, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
               ema_decay=a.ema_decay, energy_coeff=a.energy_coeff, forces_coeff=a.forces_coeff)


if __name__ == "__main__":
    raise SystemExit(main())
