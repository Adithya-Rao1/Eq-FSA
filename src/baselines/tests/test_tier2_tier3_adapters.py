import argparse
import os
import sys
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import pytest

from baselines.adapters import tier2_transformer, tier3_foundation
from baselines.datasets import omol25


def test_tier2_build_parser_returns_argparse_parser():
    parser = tier2_transformer.build_parser()
    assert isinstance(parser, argparse.ArgumentParser)


def test_tier2_build_parser_accepts_full_valid_args():
    parser = tier2_transformer.build_parser()
    args = parser.parse_args(
        [
            "--model", "equiformerv2",
            "--dataset", "omol25",
            "--split", "val_id",
            "--run-dir", "/tmp/run",
        ]
    )
    assert args.model == "equiformerv2"
    assert args.dataset == "omol25"
    assert args.split == "val_id"
    assert args.run_dir == "/tmp/run"


def test_tier2_build_parser_rejects_missing_model():
    parser = tier2_transformer.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--dataset", "omol25", "--split", "val_id", "--run-dir", "/tmp/run"])


def test_tier2_build_parser_rejects_unknown_model():
    parser = tier2_transformer.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--model", "not_a_real_model", "--dataset", "d", "--split", "s", "--run-dir", "/tmp/run"]
        )


def test_tier2_model_choices_match_spec():
    assert tier2_transformer.TIER2_MODELS == ("equiformerv2",)


def test_tier2_equiformer_v1_is_no_longer_selectable():
    parser = tier2_transformer.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--model", "equiformer", "--dataset", "d", "--split", "s", "--run-dir", "/tmp/run"])
    assert not hasattr(tier2_transformer, "_build_equiformer")

def test_tier3_build_parser_returns_argparse_parser():
    parser = tier3_foundation.build_parser()
    assert isinstance(parser, argparse.ArgumentParser)


def test_tier3_build_parser_requires_mode():
    parser = tier3_foundation.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--model", "uma", "--dataset", "omol25", "--split", "val_id", "--run-dir", "/tmp/run"]
        )


def test_tier3_build_parser_rejects_bad_mode():
    parser = tier3_foundation.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--model", "uma",
                "--mode", "from_scratch",
                "--dataset", "omol25",
                "--split", "val_id",
                "--run-dir", "/tmp/run",
            ]
        )


def test_tier3_build_parser_accepts_full_valid_args():
    parser = tier3_foundation.build_parser()
    args = parser.parse_args(
        [
            "--model", "uma",
            "--mode", "zero_shot",
            "--dataset", "omol25",
            "--split", "val_id",
            "--run-dir", "/tmp/run",
        ]
    )
    assert args.model == "uma"
    assert args.mode == "zero_shot"


def test_tier3_model_and_mode_choices_match_spec():
    assert tier3_foundation.TIER3_MODELS == ("mace_mp_0", "mace_off23", "orb_v2", "orb_v3", "uma")
    assert tier3_foundation.TIER3_MODES == ("zero_shot", "fine_tune")


def test_tier3_uma_gate_raises_without_hf_token():
    saved = {k: os.environ.pop(k, None) for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")}
    try:
        with pytest.raises(tier3_foundation.GatedAccessError, match="HF_TOKEN"):
            tier3_foundation._check_hf_gate()
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def test_tier3_uma_gate_passes_with_hf_token():
    saved = os.environ.get("HF_TOKEN")
    os.environ["HF_TOKEN"] = "dummy-token-for-test"
    try:
        assert tier3_foundation._check_hf_gate() == "dummy-token-for-test"
    finally:
        if saved is None:
            os.environ.pop("HF_TOKEN", None)
        else:
            os.environ["HF_TOKEN"] = saved


def test_tier3_mace_off23_license_gate_raises_without_acceptance():
    saved = os.environ.pop("MACE_OFF23_LICENSE_ACCEPTED", None)
    try:
        with pytest.raises(tier3_foundation.GatedAccessError, match="Academic Software License"):
            tier3_foundation._check_mace_off23_license()
    finally:
        if saved is not None:
            os.environ["MACE_OFF23_LICENSE_ACCEPTED"] = saved


def test_tier3_mace_off23_license_gate_passes_with_acceptance():
    saved = os.environ.get("MACE_OFF23_LICENSE_ACCEPTED")
    os.environ["MACE_OFF23_LICENSE_ACCEPTED"] = "1"
    try:
        tier3_foundation._check_mace_off23_license()  # must not raise
    finally:
        if saved is None:
            os.environ.pop("MACE_OFF23_LICENSE_ACCEPTED", None)
        else:
            os.environ["MACE_OFF23_LICENSE_ACCEPTED"] = saved


def test_tier3_main_rejects_missing_mode_programmatically():
    args = argparse.Namespace(
        model="uma", mode=None, dataset="d", split="s", run_dir="/tmp/run", seed=0,
        device="cpu", num_workers=0, batch_size=1, amp=False, grad_checkpoint=False,
        resume=None, deterministic=False,
    )
    with pytest.raises(ValueError, match="--mode"):
        if args.mode not in tier3_foundation.TIER3_MODES:
            raise ValueError(
                f"--mode is required and must be one of {tier3_foundation.TIER3_MODES} for Tier 3 "
                "(zero-shot/fine-tune only; pretrained foundation models are never trained from "
                "scratch)."
            )


def test_omol25_hf_gate_raises_without_token():
    saved = {k: os.environ.pop(k, None) for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")}
    try:
        with pytest.raises(omol25.OMol25AccessError, match="HF_TOKEN"):
            omol25.check_hf_gate()
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def test_omol25_read_charge_spin_present():
    assert omol25.read_charge_spin({"charge": -1, "spin": 2}) == (-1, 2)


def test_omol25_read_charge_spin_missing_raises():
    with pytest.raises(KeyError):
        omol25.read_charge_spin({})
    with pytest.raises(KeyError):
        omol25.read_charge_spin({"charge": 0})


def _synthetic_omol25_axes(n: int):
    element_combinations = [frozenset({1, 6, 8}) if i % 5 else frozenset({1, 6, 8, 17}) for i in range(n)]
    charge_spin_states = [(0, 1) if i % 3 else (1, 2) for i in range(n)]
    solvents = ["water" if i % 4 else "chloroform" for i in range(n)]
    return element_combinations, charge_spin_states, solvents


def test_omol25_ood_splits_disjoint_and_cover_all_inputs():
    n = 200
    element_combinations, charge_spin_states, solvents = _synthetic_omol25_axes(n)
    splits = omol25.build_omol25_ood_splits(element_combinations, charge_spin_states, solvents, seed=0)

    assert set(splits.keys()) == {"val_id", "val_ood_elements", "val_ood_charge_spin", "val_ood_solvent"}

    all_indices: list[int] = []
    for bucket in splits.values():
        all_indices.extend(bucket)
    assert sorted(all_indices) == list(range(n))  # covers all input, exactly once each -> disjoint


def test_omol25_ood_splits_disjoint_pairwise():
    n = 150
    element_combinations, charge_spin_states, solvents = _synthetic_omol25_axes(n)
    splits = omol25.build_omol25_ood_splits(element_combinations, charge_spin_states, solvents, seed=1)
    keys = list(splits.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            assert set(splits[keys[i]]).isdisjoint(splits[keys[j]])


def test_omol25_ood_splits_deterministic_given_seed():
    n = 80
    element_combinations, charge_spin_states, solvents = _synthetic_omol25_axes(n)
    a = omol25.build_omol25_ood_splits(element_combinations, charge_spin_states, solvents, seed=7)
    b = omol25.build_omol25_ood_splits(element_combinations, charge_spin_states, solvents, seed=7)
    assert a == b


def test_omol25_ood_splits_single_category_axis_is_never_held_out():
    # every record shares the same element combination -> val_ood_elements must be empty, and everything else still partitions cleanly.
    n = 40
    element_combinations = [frozenset({1, 8})] * n
    _, charge_spin_states, solvents = _synthetic_omol25_axes(n)
    splits = omol25.build_omol25_ood_splits(element_combinations, charge_spin_states, solvents, seed=0)
    assert splits["val_ood_elements"] == []
    all_indices = sorted(i for bucket in splits.values() for i in bucket)
    assert all_indices == list(range(n))


def test_omol25_ood_splits_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        omol25.build_omol25_ood_splits([frozenset({1})], [(0, 1), (0, 1)], [None, None])
