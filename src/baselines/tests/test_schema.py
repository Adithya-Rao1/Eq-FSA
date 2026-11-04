# Tests for src/baselines/config/schema.py
from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import pytest

from baselines.config.schema import CommonConfig, RunPaths, add_common_args

REQUIRED_ARGV = ["--dataset", "qm9", "--split", "molecule_95_5", "--run-dir", "runs/a"]
COMMON_FIELD_NAMES = {
    "seed", "dataset", "split", "run_dir", "data_root", "device", "num_workers", "batch_size",
    "amp", "grad_checkpoint", "resume", "deterministic",
}


def _parse(extra: list[str] | None = None, required: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    return parser.parse_args((REQUIRED_ARGV if required is None else required) + (extra or []))


def test_add_common_args_defaults():
    args = _parse()
    assert args.seed == 0
    assert args.data_root is None
    assert args.device == "cuda"
    assert args.num_workers == 8
    assert args.batch_size == 32
    assert args.amp is False
    assert args.grad_checkpoint is False
    assert args.resume is None
    assert args.deterministic is True


def test_add_common_args_required_values_are_parsed_verbatim():
    args = _parse()
    assert (args.dataset, args.split, args.run_dir) == ("qm9", "molecule_95_5", "runs/a")


@pytest.mark.parametrize("missing", ["--dataset", "--split", "--run-dir"])
def test_add_common_args_required_flags_are_enforced(missing, capsys):
    argv = list(REQUIRED_ARGV)
    index = argv.index(missing)
    del argv[index:index + 2]
    with pytest.raises(SystemExit) as excinfo:
        _parse(required=argv)
    assert excinfo.value.code == 2
    assert missing in capsys.readouterr().err


def test_add_common_args_attribute_names_match_commonconfig_fields():
    assert COMMON_FIELD_NAMES <= set(vars(_parse()))


def test_no_deterministic_flag_flips_deterministic_off():
    assert _parse(["--no-deterministic"]).deterministic is False
    assert _parse().deterministic is True


def test_boolean_opt_in_flags():
    args = _parse(["--amp", "--grad-checkpoint"])
    assert args.amp is True
    assert args.grad_checkpoint is True


def test_value_flags_are_parsed_with_correct_types():
    args = _parse([
        "--seed", "7", "--data-root", "/data/qm9", "--device", "cpu", "--num-workers", "0",
        "--batch-size", "4", "--resume", "ckpt/checkpoint_x.pt",
    ])
    assert args.seed == 7 and isinstance(args.seed, int)
    assert args.data_root == "/data/qm9"
    assert args.device == "cpu"
    assert args.num_workers == 0 and isinstance(args.num_workers, int)
    assert args.batch_size == 4 and isinstance(args.batch_size, int)
    assert args.resume == "ckpt/checkpoint_x.pt"


def test_non_integer_seed_is_rejected():
    with pytest.raises(SystemExit):
        _parse(["--seed", "abc"])


def test_add_common_args_can_be_combined_with_adapter_specific_flags():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--model", required=True)
    args = parser.parse_args(REQUIRED_ARGV + ["--model", "schnet"])
    assert args.model == "schnet"
    assert args.dataset == "qm9"


def test_commonconfig_dataclass_fields_and_defaults():
    assert {f.name for f in dataclasses.fields(CommonConfig)} == COMMON_FIELD_NAMES
    config = CommonConfig(seed=1, dataset="qm9", split="s", run_dir="r")
    assert config.data_root is None
    assert config.device == "cuda"
    assert config.num_workers == 8
    assert config.batch_size == 32
    assert config.amp is False
    assert config.grad_checkpoint is False
    assert config.resume is None
    assert config.deterministic is True


def test_from_args_maps_every_field_from_parsed_args():
    args = _parse([
        "--seed", "3", "--data-root", "/d", "--device", "cpu", "--num-workers", "2", "--batch-size", "16",
        "--amp", "--grad-checkpoint", "--resume", "c.pt", "--no-deterministic",
    ])
    assert CommonConfig.from_args(args) == CommonConfig(
        seed=3, dataset="qm9", split="molecule_95_5", run_dir="runs/a", data_root="/d", device="cpu",
        num_workers=2, batch_size=16, amp=True, grad_checkpoint=True, resume="c.pt", deterministic=False,
    )


def test_from_args_of_defaults_equals_default_commonconfig():
    assert CommonConfig.from_args(_parse()) == CommonConfig(seed=0, dataset="qm9", split="molecule_95_5", run_dir="runs/a")


def test_from_args_ignores_extra_namespace_keys():
    args = _parse()
    args.model = "schnet"
    args.epochs = 5
    args.lr = 1e-3
    config = CommonConfig.from_args(args)
    assert not hasattr(config, "model")
    assert config.dataset == "qm9"


def test_from_args_from_hand_built_namespace_uses_dataclass_defaults_for_absent_keys():
    namespace = argparse.Namespace(seed=4, dataset="rmd17:aspirin", split="rmd17_split_0", run_dir="r", unrelated=True)
    config = CommonConfig.from_args(namespace)
    assert config == CommonConfig(seed=4, dataset="rmd17:aspirin", split="rmd17_split_0", run_dir="r")


def test_from_args_missing_required_field_raises_type_error():
    with pytest.raises(TypeError):
        CommonConfig.from_args(argparse.Namespace(seed=0, dataset="qm9", split="s"))


def test_runpaths_layout(tmp_path):
    run_dir = str(tmp_path / "run")
    paths = RunPaths(run_dir)
    assert paths.run_dir == run_dir
    assert paths.checkpoint_dir == os.path.join(run_dir, "checkpoints")
    assert paths.env_manifest_path == os.path.join(run_dir, "env_manifest.json")
    assert paths.log_path == os.path.join(run_dir, "run.log")
    assert paths.metrics_path == os.path.join(run_dir, "metrics.jsonl")


def test_runpaths_constructor_does_not_touch_the_filesystem(tmp_path):
    RunPaths(str(tmp_path / "never_created"))
    assert not (tmp_path / "never_created").exists()


def test_runpaths_ensure_creates_checkpoint_dir_and_returns_self(tmp_path):
    paths = RunPaths(str(tmp_path / "nested" / "run"))
    assert paths.ensure() is paths
    assert os.path.isdir(paths.checkpoint_dir)
    assert os.path.isdir(paths.run_dir)


def test_runpaths_ensure_is_idempotent_and_keeps_existing_files(tmp_path):
    paths = RunPaths(str(tmp_path / "run")).ensure()
    marker = Path(paths.checkpoint("keep"))
    marker.write_text("data")
    paths.ensure()
    assert marker.read_text() == "data"


def test_runpaths_ensure_creates_only_directories_not_files(tmp_path):
    paths = RunPaths(str(tmp_path / "run")).ensure()
    assert sorted(os.listdir(paths.run_dir)) == ["checkpoints"]
    assert os.listdir(paths.checkpoint_dir) == []


def test_runpaths_checkpoint_naming(tmp_path):
    paths = RunPaths(str(tmp_path / "run"))
    assert paths.checkpoint("schnet_epoch3") == os.path.join(paths.checkpoint_dir, "checkpoint_schnet_epoch3.pt")
    assert paths.checkpoint("latest") == os.path.join(str(tmp_path / "run"), "checkpoints", "checkpoint_latest.pt")
    assert paths.checkpoint("a") != paths.checkpoint("b")


def test_runpaths_checkpoint_does_not_create_the_file(tmp_path):
    paths = RunPaths(str(tmp_path / "run")).ensure()
    assert not os.path.exists(paths.checkpoint("x"))
