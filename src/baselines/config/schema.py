from __future__ import annotations

import argparse
import dataclasses
import os


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, required=True, help="split convention name, e.g. molecule_95_5, rmd17_fixed_1000")
    parser.add_argument("--data-root", type=str, default=None, help="dataset root dir (or file path for ani1x/spice)")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--amp", action="store_true", help="opt-in mixed precision; verify forces against fp32 before trusting for a given architecture")
    parser.add_argument("--grad-checkpoint", action="store_true")
    parser.add_argument("--resume", type=str, default=None, help="checkpoint path to resume from")
    parser.add_argument("--no-deterministic", dest="deterministic", action="store_false")
    parser.set_defaults(deterministic=True)


@dataclasses.dataclass
class CommonConfig:
    seed: int
    dataset: str
    split: str
    run_dir: str
    data_root: str | None = None
    device: str = "cuda"
    num_workers: int = 8
    batch_size: int = 32
    amp: bool = False
    grad_checkpoint: bool = False
    resume: str | None = None
    deterministic: bool = True

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "CommonConfig":
        field_names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in vars(args).items() if k in field_names})


class RunPaths:
    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.checkpoint_dir = os.path.join(run_dir, "checkpoints")
        self.env_manifest_path = os.path.join(run_dir, "env_manifest.json")
        self.log_path = os.path.join(run_dir, "run.log")
        self.metrics_path = os.path.join(run_dir, "metrics.jsonl")

    def ensure(self) -> "RunPaths":
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        return self

    def checkpoint(self, tag: str) -> str:
        return os.path.join(self.checkpoint_dir, f"checkpoint_{tag}.pt")
