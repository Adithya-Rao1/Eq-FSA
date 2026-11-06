#!/usr/bin/env bash
# Builds, sha256-digests, and smoke-tests each of this project's 5 baseline Singularity images (mlip-legacy, mlip-mpnn, mlip-transformer, mlip-foundation, mlip-classical). 
# Singularity 3.7 build command syntax: https://docs.sylabs.io/guides/3.7/user-guide/build_a_container.html
# Intended to run on a machine where `sudo` is available. 

set -euo pipefail

SINGULARITY_BIN="${SINGULARITY_BIN:-singularity}"
CONTAINER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${OUT_DIR:-$CONTAINER_DIR}"

IMAGES=(mlip-legacy mlip-mpnn mlip-transformer mlip-foundation mlip-classical)

# One import check per library each image is supposed to carry, plus the shared torch/cuda check run separately below.
declare -A SMOKE_IMPORTS=(
  [mlip-legacy]="import torch.utils.data.dataloader as _d; _d.T_co = getattr(_d, 'T_co', None) or _d._T_co; import schnetpack; import torchani; import aimnetcentral; import h5py; import rdkit; import pytest"
  [mlip-mpnn]="import nequip; import allegro; import mace; import h5py; import rdkit; import pytest"
  [mlip-transformer]="import fairchem.core; from fairchem.core.models.equiformer_v2.equiformer_v2 import EquiformerV2Backbone; import torch_geometric, torch_scatter, torch_sparse, torch_cluster; import h5py; import rdkit; import pytest"
  [mlip-classical]="import openmm, openff.toolkit, openmmforcefields, rdkit, ase, pytest, shutil; assert shutil.which('antechamber') and shutil.which('sqm'), 'AmberTools missing from PATH'; from openff.toolkit import Molecule; m = Molecule.from_smiles('CCO'); m.generate_conformers(n_conformers=1); m.assign_partial_charges('am1bcc')"
  [mlip-foundation]="import mace; import orb_models; import fairchem; import huggingface_hub; import pytest"
)

build_one() {
  local name="$1"
  local def_path="${CONTAINER_DIR}/${name}.def"
  local sif_path="${OUT_DIR}/${name}.sif"
  local digest_path="${sif_path}.sha256"

  if [[ ! -f "$def_path" ]]; then
    echo "ERROR: missing def file: $def_path" >&2
    return 1
  fi


  local tmp_sif="${sif_path}.building"
  rm -f "$tmp_sif"
  echo "[build_and_verify] building ${name}.sif ..."
  sudo "$SINGULARITY_BIN" build "$tmp_sif" "$def_path"
  mv -f "$tmp_sif" "$sif_path"

  echo "[build_and_verify] hashing ${name}.sif ..."
  sha256sum "$sif_path" | awk '{print $1}' > "$digest_path"
  echo "[build_and_verify] wrote $(cat "$digest_path")  ->  ${digest_path}"

  echo "[build_and_verify] smoke-testing ${name}.sif ..."
  if [[ "$name" != "mlip-classical" ]]; then  # the classical image is CPU-only and has no torch
    "$SINGULARITY_BIN" exec --nv "$sif_path" \
      python3 -c "import torch; print('cuda_available=', torch.cuda.is_available())"
  fi
  "$SINGULARITY_BIN" exec --nv "$sif_path" \
    python3 -c "${SMOKE_IMPORTS[$name]}"
  echo "[build_and_verify] ${name}.sif OK"
}

main() {
  local only="${1:-}"
  for name in "${IMAGES[@]}"; do
    if [[ -n "$only" && "$only" != "$name" ]]; then
      continue
    fi
    build_one "$name"
  done
}

main "$@"
