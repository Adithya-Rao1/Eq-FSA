from __future__ import annotations
# OMol25 -- Levine et al. (Meta FAIR), "The Open Molecules 2025 (OMol25) Dataset, Evaluations, and Models," arXiv:2505.08762. 
#
# OC20 (Chanussot et al., arXiv:2010.09990) based bucket assignment: in-domain / unseen-element-combination / unseen-charge-spin-state / unseen-solvents based on priority (element --> charge --> solvent)

import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from random import Random
from typing import Any

from baselines.config.units import to_ev_angstrom

HF_OMOL25_REPO = "facebook/OMol25"  # TODO: confirm against HF listing
OMOL25_LICENSE_URL = "https://huggingface.co/facebook/OMol25"


class OMol25AccessError(RuntimeError):
    pass


def check_hf_gate() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise OMol25AccessError(
            f"OMol25 is gated on Hugging Face ({HF_OMOL25_REPO}). Set HF_TOKEN (or "
            f"HUGGING_FACE_HUB_TOKEN) to a token for an account that has accepted the license at "
            f"{OMOL25_LICENSE_URL} before loading OMol25 -- this never falls back to an empty "
            "dataset."
        )
    return token


def read_charge_spin(atoms_info: Mapping[str, Any]) -> tuple[int, int]:
    if "charge" not in atoms_info or "spin" not in atoms_info:
        raise KeyError(
            "OMol25 record is missing explicit 'charge' and/or 'spin' in atoms.info; refusing to "
            "assume a neutral-singlet default (charge=0, spin=0) since OMol25 structures are "
            "frequently ionic/open-shell and that default would silently mislabel them."
        )
    return int(atoms_info["charge"]), int(atoms_info["spin"])


def _open_aselmdb(src: str | list[str]):
    check_hf_gate()
    try:
        from fairchem.core.datasets import AseDBDataset  # type: ignore
    except ImportError as e:
        raise ImportError(
            "fairchem is not installed. Run `pip install -e src/baselines/codebases/fairchem` "
            "(or `pip install fairchem-core`) in the remote GPU environment before loading OMol25 "
            "*.aselmdb files."
        ) from e
    # r_energy/r_forces so AtomsToGraphs actually reads energy+forces off each row
    return AseDBDataset({"src": src, "a2g_args": {"r_energy": True, "r_forces": True}})


@dataclass(frozen=True)
class OMol25Example:
    atomic_numbers: list[int]
    positions: list[tuple[float, float, float]]
    energy_ev: float
    forces_ev_angstrom: list[tuple[float, float, float]]
    charge: int
    spin: int
    solvent: str | None


def load_omol25_split(src: str | list[str]) -> list[OMol25Example]:
    dataset = _open_aselmdb(src)
    examples: list[OMol25Example] = []
    for idx in range(len(dataset)):
        atoms = dataset.get_atoms(idx)
        charge, spin = read_charge_spin(atoms.info)
        energy_ev, forces_ev_angstrom = to_ev_angstrom(
            atoms.get_potential_energy(), atoms.get_forces(), source="ev_angstrom"
        )
        examples.append(
            OMol25Example(
                atomic_numbers=list(atoms.get_atomic_numbers()),
                positions=[tuple(p) for p in atoms.get_positions()],
                energy_ev=float(energy_ev),
                forces_ev_angstrom=[tuple(f) for f in forces_ev_angstrom],
                charge=charge,
                spin=spin,
                solvent=atoms.info.get("solvent"),
            )
        )
    return examples


def _element_combination(atomic_numbers: Iterable[int]) -> frozenset[int]:
    return frozenset(atomic_numbers)


def build_omol25_ood_splits(
    element_combinations: Sequence[frozenset[int]],
    charge_spin_states: Sequence[tuple[int, int]],
    solvents: Sequence[str | None],
    seed: int = 0,
    held_out_fraction: float = 0.1,
) -> dict[str, list[int]]:
    # OC20-style 4-way OOD split.
    n = len(element_combinations)
    if len(charge_spin_states) != n or len(solvents) != n:
        raise ValueError(
            "element_combinations, charge_spin_states, and solvents must all be the same length "
            f"(got {n}, {len(charge_spin_states)}, {len(solvents)})"
        )

    rng = Random(seed)

    def _held_out(values: Sequence[Any]) -> set[Any]:
        uniq = sorted(set(values), key=repr)
        if len(uniq) < 2:
            return set() 
        k = max(1, round(len(uniq) * held_out_fraction))
        k = min(k, len(uniq) - 1)  # never hold out every category --> val_id must stay non-empty-able
        return set(rng.sample(uniq, k))

    held_out_elements = _held_out(element_combinations)
    held_out_charge_spin = _held_out(charge_spin_states)
    held_out_solvents = _held_out(solvents)

    splits: dict[str, list[int]] = {
        "val_id": [],
        "val_ood_elements": [],
        "val_ood_charge_spin": [],
        "val_ood_solvent": [],
    }
    for i in range(n):
        # Priority order (elements, then charge/spin, then solvent) is the explicit tie-break for records that hit more than one held-out axis at once to keep the 4 buckets disjoint.
        if element_combinations[i] in held_out_elements:
            splits["val_ood_elements"].append(i)
        elif charge_spin_states[i] in held_out_charge_spin:
            splits["val_ood_charge_spin"].append(i)
        elif solvents[i] in held_out_solvents:
            splits["val_ood_solvent"].append(i)
        else:
            splits["val_id"].append(i)
    return splits
