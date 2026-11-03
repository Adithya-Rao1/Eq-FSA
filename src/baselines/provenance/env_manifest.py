from __future__ import annotations

# Hardware/software spec reporting follows "Enhancing the Quality and Reliability of MLIPs through Better Reporting Practices," arXiv:2401.02284.

import json
import os
import platform
import socket
import subprocess
import sys
from typing import Any

import torch

# Used in each Singularity image at build time (containers/build_and_verify.sh);
# Absent when running outside a container (e.g. local dev).
_CONTAINER_DIGEST_FILE = "/opt/image_digest.txt"

_SLURM_VARS = (
    "SLURM_JOB_ID", "SLURM_JOB_PARTITION", "SLURMD_NODENAME",
    "SLURM_CPUS_PER_TASK", "SLURM_GPUS_ON_NODE", "SLURM_NTASKS",
)


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=15, check=True)
        return out.stdout.strip()
    except Exception:
        return None


def _git_info(repo_dir: str | None = None) -> dict[str, Any]:
    cwd = repo_dir or os.path.dirname(os.path.abspath(__file__))
    commit = _run(["git", "-C", cwd, "rev-parse", "HEAD"])
    dirty = _run(["git", "-C", cwd, "status", "--porcelain"])
    return {"commit": commit, "dirty": bool(dirty) if dirty is not None else None}


def _container_digest() -> str | None:
    if os.path.isfile(_CONTAINER_DIGEST_FILE):
        with open(_CONTAINER_DIGEST_FILE) as f:
            return f.read().strip()
    return None


def _gpu_info() -> list[dict[str, Any]]:
    gpus = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        gpus.append({
            "index": i,
            "name": props.name,
            "total_memory_bytes": props.total_memory,
            "multi_processor_count": props.multi_processor_count,
        })
    return gpus


def _driver_version() -> str | None:
    return _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])


def _slurm_context() -> dict[str, str | None]:
    return {k: os.environ.get(k) for k in _SLURM_VARS}


def _pip_freeze() -> list[str]:
    out = _run([sys.executable, "-m", "pip", "freeze"])
    return out.splitlines() if out else []


def collect(repo_dir: str | None = None) -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "cuda_available": torch.cuda.is_available(),
        "gpus": _gpu_info(),
        "nvidia_driver_version": _driver_version(),
        "git": _git_info(repo_dir),
        "container_image_digest": _container_digest(),
        "slurm": _slurm_context(),
        "pip_freeze": _pip_freeze(),
    }


def write(run_dir: str, repo_dir: str | None = None) -> dict[str, Any]:
    manifest = collect(repo_dir)
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "env_manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return manifest
