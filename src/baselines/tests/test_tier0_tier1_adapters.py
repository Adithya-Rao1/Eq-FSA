import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest

from baselines.adapters.tier0_legacy import build_parser as build_tier0_parser
from baselines.adapters.tier1_mpnn import build_parser as build_tier1_parser


def _action_dict(parser):
    return {a.option_strings[0]: a for a in parser._actions if a.option_strings}


def test_tier0_parser_model_choices():
    actions = _action_dict(build_tier0_parser())
    assert actions["--model"].required
    assert set(actions["--model"].choices) == {"schnet", "painn", "ani2x", "aimnet2"}


def test_tier0_parser_common_flags_present():
    actions = _action_dict(build_tier0_parser())
    for flag in ("--seed", "--dataset", "--split", "--run-dir", "--device", "--num-workers", "--batch-size", "--amp", "--resume"):
        assert flag in actions, f"missing common flag {flag}"
    assert actions["--dataset"].required
    assert actions["--split"].required
    assert actions["--run-dir"].required


def test_tier0_parser_model_specific_flags_present():
    actions = _action_dict(build_tier0_parser())
    for flag in ("--epochs", "--lr", "--hidden-dim", "--n-interactions", "--cutoff", "--ani2x-ensemble-size", "--aimnet2-config"):
        assert flag in actions, f"missing tier0 flag {flag}"


def test_tier0_parser_parses_minimal_args():
    args = build_tier0_parser().parse_args([
        "--model", "schnet", "--dataset", "d", "--split", "s", "--run-dir", "/tmp/r",
    ])
    assert args.model == "schnet"
    assert args.seed == 0
    assert args.deterministic is True


def test_tier0_parser_rejects_unknown_model():
    with pytest.raises(SystemExit):
        build_tier0_parser().parse_args([
            "--model", "not-a-model", "--dataset", "d", "--split", "s", "--run-dir", "/tmp/r",
        ])


def test_tier1_parser_model_choices():
    actions = _action_dict(build_tier1_parser())
    assert actions["--model"].required
    assert set(actions["--model"].choices) == {"nequip", "allegro", "mace"}


def test_tier1_parser_common_flags_present():
    actions = _action_dict(build_tier1_parser())
    for flag in ("--seed", "--dataset", "--split", "--run-dir", "--device", "--num-workers", "--batch-size", "--amp", "--resume"):
        assert flag in actions, f"missing common flag {flag}"


def test_tier1_parser_model_specific_flags_present():
    actions = _action_dict(build_tier1_parser())
    for flag in ("--epochs", "--lr", "--cutoff", "--num-layers", "--l-max", "--hidden-dim", "--type-names"):
        assert flag in actions, f"missing tier1 flag {flag}"


def test_tier1_parser_parses_minimal_args():
    args = build_tier1_parser().parse_args([
        "--model", "nequip", "--dataset", "d", "--split", "s", "--run-dir", "/tmp/r",
    ])
    assert args.model == "nequip"
    assert args.type_names == "H,C,N,O"


def test_tier1_parser_rejects_unknown_model():
    with pytest.raises(SystemExit):
        build_tier1_parser().parse_args([
            "--model", "not-a-model", "--dataset", "d", "--split", "s", "--run-dir", "/tmp/r",
        ])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
