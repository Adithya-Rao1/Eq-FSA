# Tests for src/baselines/provenance/env_manifest.py
# reference for the recorded fields: arXiv:2401.02284.
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import pytest

from baselines.config.schema import RunPaths
from baselines.provenance import env_manifest

EXPECTED_KEYS = {
    "hostname", "python_version", "platform", "cpu_count", "torch_version", "torch_cuda_version",
    "cudnn_version", "cuda_available", "gpus", "nvidia_driver_version", "git", "container_image_digest",
    "slurm", "pip_freeze",
}
EXPECTED_SLURM_KEYS = {
    "SLURM_JOB_ID", "SLURM_JOB_PARTITION", "SLURMD_NODENAME", "SLURM_CPUS_PER_TASK", "SLURM_GPUS_ON_NODE", "SLURM_NTASKS",
}
FAKE_COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _fake_run_factory(outputs: dict[str, str | None], calls: list[list[str]] | None = None):
    # Dispatches on the executable / git subcommand so each probe can be answered independently.
    def fake_run(cmd: list[str]) -> str | None:
        if calls is not None:
            calls.append(list(cmd))
        if cmd[0] == "git" and "rev-parse" in cmd:
            return outputs.get("rev-parse")
        if cmd[0] == "git" and "status" in cmd:
            return outputs.get("status")
        if cmd[0] == "nvidia-smi":
            return outputs.get("nvidia-smi")
        if "pip" in cmd:
            return outputs.get("pip")
        return None
    return fake_run


def _clear_slurm_env(monkeypatch) -> None:
    for name in EXPECTED_SLURM_KEYS:
        monkeypatch.delenv(name, raising=False)


def test_collect_returns_json_serialisable_dict_with_expected_keys(monkeypatch):
    monkeypatch.setattr(env_manifest, "_run", _fake_run_factory({
        "rev-parse": FAKE_COMMIT, "status": "", "nvidia-smi": "550.54.15", "pip": "numpy==1.26.4\ntorch==2.3.0",
    }))
    manifest = env_manifest.collect()
    assert set(manifest) == EXPECTED_KEYS
    assert json.loads(json.dumps(manifest)) == manifest


def test_collect_field_types():
    manifest = env_manifest.collect()
    # platform.python_version() may append a build suffix (e.g. "3.12.3+"), so compare the numeric prefix.
    assert manifest["python_version"].startswith(".".join(str(v) for v in sys.version_info[:3]))
    assert isinstance(manifest["hostname"], str) and manifest["hostname"]
    assert isinstance(manifest["cuda_available"], bool)
    assert isinstance(manifest["gpus"], list)
    assert isinstance(manifest["pip_freeze"], list)
    assert set(manifest["git"]) == {"commit", "dirty"}
    assert set(manifest["slurm"]) == EXPECTED_SLURM_KEYS


def test_collect_reports_probe_results_from_run_helper(monkeypatch):
    monkeypatch.setattr(env_manifest, "_run", _fake_run_factory({
        "rev-parse": FAKE_COMMIT, "status": " M file.py", "nvidia-smi": "550.54.15", "pip": "numpy==1.26.4\ntorch==2.3.0",
    }))
    manifest = env_manifest.collect()
    assert manifest["git"] == {"commit": FAKE_COMMIT, "dirty": True}
    assert manifest["nvidia_driver_version"] == "550.54.15"
    assert manifest["pip_freeze"] == ["numpy==1.26.4", "torch==2.3.0"]


def test_git_info_clean_tree_is_dirty_false(monkeypatch):
    # `git status --porcelain` prints nothing for a clean tree; that is "known clean", not "unknown".
    monkeypatch.setattr(env_manifest, "_run", _fake_run_factory({"rev-parse": FAKE_COMMIT, "status": ""}))
    assert env_manifest._git_info() == {"commit": FAKE_COMMIT, "dirty": False}


def test_git_info_uses_given_repo_dir_in_both_git_commands(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(env_manifest, "_run", _fake_run_factory({"rev-parse": FAKE_COMMIT, "status": ""}, calls))
    env_manifest._git_info("/some/repo")
    assert ["git", "-C", "/some/repo", "rev-parse", "HEAD"] in calls
    assert ["git", "-C", "/some/repo", "status", "--porcelain"] in calls


def test_git_info_default_dir_is_the_provenance_package_dir(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(env_manifest, "_run", _fake_run_factory({}, calls))
    env_manifest._git_info()
    expected_dir = os.path.dirname(os.path.abspath(env_manifest.__file__))
    assert all(cmd[2] == expected_dir for cmd in calls)


def test_collect_passes_repo_dir_through_to_git(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(env_manifest, "_run", _fake_run_factory({}, calls))
    env_manifest.collect(repo_dir="/another/repo")
    assert any(cmd[:3] == ["git", "-C", "/another/repo"] for cmd in calls)


def test_nvidia_smi_query_command(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(env_manifest, "_run", _fake_run_factory({"nvidia-smi": "550.54.15"}, calls))
    assert env_manifest._driver_version() == "550.54.15"
    assert calls == [["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]]


def test_pip_freeze_uses_current_interpreter_and_splits_lines(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(env_manifest, "_run", _fake_run_factory({"pip": "a==1\nb==2"}, calls))
    assert env_manifest._pip_freeze() == ["a==1", "b==2"]
    assert calls == [[sys.executable, "-m", "pip", "freeze"]]


@pytest.mark.parametrize("pip_output", [None, ""])
def test_pip_freeze_empty_when_unavailable(monkeypatch, pip_output):
    monkeypatch.setattr(env_manifest, "_run", _fake_run_factory({"pip": pip_output}))
    assert env_manifest._pip_freeze() == []


def test_collect_degrades_gracefully_when_subprocess_binaries_are_missing(monkeypatch):
    def raise_file_not_found(*args, **kwargs):
        raise FileNotFoundError("no such executable")

    platform.platform()  # warm platform's internal cache, which may itself shell out on some systems
    monkeypatch.setattr(subprocess, "run", raise_file_not_found)
    manifest = env_manifest.collect()
    assert set(manifest) == EXPECTED_KEYS
    assert manifest["nvidia_driver_version"] is None
    assert manifest["git"] == {"commit": None, "dirty": None}
    assert manifest["pip_freeze"] == []
    json.dumps(manifest)


def test_collect_degrades_gracefully_when_run_helper_returns_none(monkeypatch):
    monkeypatch.setattr(env_manifest, "_run", lambda cmd: None)
    manifest = env_manifest.collect()
    assert manifest["nvidia_driver_version"] is None
    assert manifest["git"] == {"commit": None, "dirty": None}
    assert manifest["pip_freeze"] == []
    assert manifest["hostname"]


@pytest.mark.parametrize("error", [
    FileNotFoundError("missing"),
    subprocess.CalledProcessError(returncode=128, cmd=["git"]),
    subprocess.TimeoutExpired(cmd=["nvidia-smi"], timeout=15),
    PermissionError("denied"),
])
def test_run_returns_none_on_any_subprocess_failure(monkeypatch, error):
    def failing_run(*args, **kwargs):
        raise error

    monkeypatch.setattr(subprocess, "run", failing_run)
    assert env_manifest._run(["anything"]) is None


def test_run_returns_stripped_stdout_and_passes_safe_arguments(monkeypatch):
    seen: dict = {}

    def fake_subprocess_run(cmd, **kwargs):
        seen["cmd"], seen["kwargs"] = cmd, kwargs
        return SimpleNamespace(stdout="  abc123 \n")

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)
    assert env_manifest._run(["git", "rev-parse", "HEAD"]) == "abc123"
    assert seen["cmd"] == ["git", "rev-parse", "HEAD"]
    assert seen["kwargs"]["check"] is True
    assert seen["kwargs"]["capture_output"] is True
    assert seen["kwargs"]["text"] is True
    assert seen["kwargs"]["timeout"] == 15


def test_run_real_subprocess_success_and_failure_and_missing_binary():
    assert env_manifest._run([sys.executable, "-c", "print('hello')"]) == "hello"
    assert env_manifest._run([sys.executable, "-c", "import sys; sys.exit(3)"]) is None
    assert env_manifest._run(["definitely-not-a-real-binary-xyz"]) is None


def test_gpu_info_empty_without_devices(monkeypatch):
    monkeypatch.setattr(env_manifest.torch.cuda, "device_count", lambda: 0)
    assert env_manifest._gpu_info() == []


def test_gpu_info_reports_each_device(monkeypatch):
    fake_props = {
        0: SimpleNamespace(name="Fake A100", total_memory=40 * 2**30, multi_processor_count=108),
        1: SimpleNamespace(name="Fake H100", total_memory=80 * 2**30, multi_processor_count=132),
    }
    monkeypatch.setattr(env_manifest.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(env_manifest.torch.cuda, "get_device_properties", lambda i: fake_props[i])
    assert env_manifest._gpu_info() == [
        {"index": 0, "name": "Fake A100", "total_memory_bytes": 40 * 2**30, "multi_processor_count": 108},
        {"index": 1, "name": "Fake H100", "total_memory_bytes": 80 * 2**30, "multi_processor_count": 132},
    ]


def test_container_digest_read_and_stripped_when_present(monkeypatch, tmp_path):
    digest_file = tmp_path / "image_digest.txt"
    digest_file.write_text("sha256:deadbeef\n")
    monkeypatch.setattr(env_manifest, "_CONTAINER_DIGEST_FILE", str(digest_file))
    assert env_manifest._container_digest() == "sha256:deadbeef"
    monkeypatch.setattr(env_manifest, "_run", lambda cmd: None)
    assert env_manifest.collect()["container_image_digest"] == "sha256:deadbeef"


def test_container_digest_none_when_file_absent(monkeypatch, tmp_path):
    monkeypatch.setattr(env_manifest, "_CONTAINER_DIGEST_FILE", str(tmp_path / "does_not_exist.txt"))
    assert env_manifest._container_digest() is None
    monkeypatch.setattr(env_manifest, "_run", lambda cmd: None)
    assert env_manifest.collect()["container_image_digest"] is None


def test_container_digest_none_when_path_is_a_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(env_manifest, "_CONTAINER_DIGEST_FILE", str(tmp_path))
    assert env_manifest._container_digest() is None


def test_container_digest_default_path_is_the_image_build_location():
    assert env_manifest._CONTAINER_DIGEST_FILE == "/opt/image_digest.txt"


def test_slurm_context_reflects_environment(monkeypatch):
    _clear_slurm_env(monkeypatch)
    monkeypatch.setenv("SLURM_JOB_ID", "424242")
    monkeypatch.setenv("SLURM_JOB_PARTITION", "gpu")
    monkeypatch.setenv("SLURMD_NODENAME", "node07")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "16")
    monkeypatch.setenv("SLURM_GPUS_ON_NODE", "4")
    monkeypatch.setenv("SLURM_NTASKS", "1")
    expected = {
        "SLURM_JOB_ID": "424242", "SLURM_JOB_PARTITION": "gpu", "SLURMD_NODENAME": "node07",
        "SLURM_CPUS_PER_TASK": "16", "SLURM_GPUS_ON_NODE": "4", "SLURM_NTASKS": "1",
    }
    assert env_manifest._slurm_context() == expected
    monkeypatch.setattr(env_manifest, "_run", lambda cmd: None)
    assert env_manifest.collect()["slurm"] == expected


def test_slurm_context_all_none_outside_slurm(monkeypatch):
    _clear_slurm_env(monkeypatch)
    assert env_manifest._slurm_context() == {name: None for name in EXPECTED_SLURM_KEYS}


def test_slurm_context_partial_environment(monkeypatch):
    _clear_slurm_env(monkeypatch)
    monkeypatch.setenv("SLURM_JOB_ID", "7")
    context = env_manifest._slurm_context()
    assert context["SLURM_JOB_ID"] == "7"
    assert context["SLURM_NTASKS"] is None


def test_write_creates_valid_json_at_runpaths_env_manifest_path(monkeypatch, tmp_path):
    monkeypatch.setattr(env_manifest, "_run", _fake_run_factory({"rev-parse": FAKE_COMMIT, "status": ""}))
    run_dir = str(tmp_path / "run")
    returned = env_manifest.write(run_dir)
    path = RunPaths(run_dir).env_manifest_path
    assert os.path.isfile(path)
    with open(path) as f:
        on_disk = json.load(f)
    assert set(on_disk) == EXPECTED_KEYS
    assert on_disk == json.loads(json.dumps(returned, default=str))
    assert on_disk["git"] == {"commit": FAKE_COMMIT, "dirty": False}


def test_write_creates_missing_nested_run_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(env_manifest, "_run", lambda cmd: None)
    run_dir = str(tmp_path / "a" / "b" / "c")
    env_manifest.write(run_dir)
    assert os.path.isfile(os.path.join(run_dir, "env_manifest.json"))


def test_write_overwrites_existing_manifest(monkeypatch, tmp_path):
    monkeypatch.setattr(env_manifest, "_run", lambda cmd: None)
    run_dir = str(tmp_path)
    (tmp_path / "env_manifest.json").write_text("stale")
    env_manifest.write(run_dir)
    with open(tmp_path / "env_manifest.json") as f:
        assert set(json.load(f)) == EXPECTED_KEYS


def test_write_stringifies_non_json_native_values_instead_of_raising(monkeypatch, tmp_path):
    monkeypatch.setattr(env_manifest, "_run", lambda cmd: None)
    monkeypatch.setattr(env_manifest, "_container_digest", lambda: Path("/opt/odd"))
    env_manifest.write(str(tmp_path))
    with open(tmp_path / "env_manifest.json") as f:
        assert json.load(f)["container_image_digest"] == "/opt/odd"


def test_write_passes_repo_dir_through(monkeypatch, tmp_path):
    calls: list[list[str]] = []
    monkeypatch.setattr(env_manifest, "_run", _fake_run_factory({}, calls))
    env_manifest.write(str(tmp_path), repo_dir="/pinned/repo")
    assert any(cmd[:3] == ["git", "-C", "/pinned/repo"] for cmd in calls)
