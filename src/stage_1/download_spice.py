from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from pathlib import Path

import yaml

from stage_1.spice_dataset import SPICE_SUBSET_GROUPS

SPICE_REPO_URL = "https://github.com/openmm/spice-dataset.git"
REQUIRED_MODULES = ("qcportal", "yaml", "h5py", "numpy", "rdkit")


def _check_dependencies() -> None:
    missing = []
    for mod in REQUIRED_MODULES:
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        pip_names = {"yaml": "pyyaml"}.get
        install_names = " ".join(pip_names(m, m) for m in missing)
        raise RuntimeError(
            f"Missing required package(s) for the SPICE downloader: {', '.join(missing)}. "
            f"Install with: pip install {install_names}"
        )


def _clone_repo(repo_dir: Path) -> None:
    if repo_dir.exists():
        print(f"[download_spice] Reusing existing clone at {repo_dir}")
        return
    print(f"[download_spice] Cloning {SPICE_REPO_URL} -> {repo_dir}")
    subprocess.run(
        ["git", "clone", "--depth", "1", SPICE_REPO_URL, str(repo_dir)],
        check=True,
    )


def _resolve_subsets(subsets: list[str] | None) -> list[str] | None:
    if subsets is None:
        return None
    resolved: list[str] = []
    for s in subsets:
        resolved.extend(SPICE_SUBSET_GROUPS.get(s, [s]))
    seen: set[str] = set()
    out = []
    for s in resolved:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _write_config(
    src_config_path: Path,
    dst_config_path: Path,
    subsets: list[str] | None,
    max_force: float | None,
) -> None:
    with open(src_config_path) as f:
        config = yaml.safe_load(f)

    resolved_subsets = _resolve_subsets(subsets)
    if resolved_subsets is not None:
        available = set(config["subsets"])
        unknown = [s for s in resolved_subsets if s not in available]
        if unknown:
            raise ValueError(
                f"Requested subset(s) not found in downloader/config.yaml: {unknown}. "
                f"Known friendly names: {sorted(SPICE_SUBSET_GROUPS)}. "
                f"Known literal subset strings: {sorted(available)}."
            )
        config["subsets"] = resolved_subsets

    if max_force is not None:
        config["max_force"] = max_force

    with open(dst_config_path, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"[download_spice] Wrote {dst_config_path} with {len(config['subsets'])} subset(s):")
    for s in config["subsets"]:
        print(f"    - {s}")


def download_spice(
    output_dir: str,
    subsets: list[str] | None = None,
    repo_dir: str | None = None,
    max_force: float | None = None,
) -> Path:
    _check_dependencies()

    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if repo_dir is not None:
        repo_path = Path(repo_dir).expanduser().resolve()
        if not repo_path.exists():
            raise FileNotFoundError(f"--repo-dir {repo_path} does not exist")
    else:
        repo_path = out_dir / "_spice_repo"
        _clone_repo(repo_path)

    downloader_script = repo_path / "downloader" / "downloader.py"
    src_config = repo_path / "downloader" / "config.yaml"
    if not downloader_script.exists() or not src_config.exists():
        raise FileNotFoundError(
            f"Expected {downloader_script} and {src_config} to exist -- "
            f"is {repo_path} a valid clone of {SPICE_REPO_URL}?"
        )
        
    _write_config(src_config, out_dir / "config.yaml", subsets, max_force)

    print(f"[download_spice] Running downloader.py (cwd={out_dir}) -- this can take a long time.")
    subprocess.run(
        [sys.executable, str(downloader_script)],
        cwd=str(out_dir),
        check=True,
    )

    output_file = out_dir / "SPICE.hdf5"
    print(f"[download_spice] Done. Dataset written to {output_file}")
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=str, required=True, help="Directory SPICE.hdf5 is written into")
    parser.add_argument(
        "--subsets", type=str, nargs="+", default=None,
        help=(
            "Which subsets to download: friendly names "
            f"({', '.join(sorted(SPICE_SUBSET_GROUPS))}) or literal config.yaml subset "
            "strings. Omit to download every subset."
        ),
    )
    parser.add_argument(
        "--repo-dir", type=str, default=None,
        help="Path to an already-cloned openmm/spice-dataset repo; skips cloning if given",
    )
    parser.add_argument(
        "--max-force", type=float, default=None,
        help="Skip samples where any force component exceeds this (hartree/bohr); default matches the repo's config.yaml",
    )
    args = parser.parse_args()

    download_spice(
        output_dir=args.output_dir,
        subsets=args.subsets,
        repo_dir=args.repo_dir,
        max_force=args.max_force,
    )
