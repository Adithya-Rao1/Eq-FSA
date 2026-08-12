from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from diagnostics.run_metrics import extract_run_metrics

SRC_DIR = Path(__file__).resolve().parents[4]
TRAIN_MODULE = "stage_1.train_spice"
GPU_PROFILER_MODULE = "diagnostics.gpu_utilization_profiler"


def to_cli_args(overrides: dict) -> list[str]:
    args = []
    for key, value in overrides.items():
        if value is None:
            continue
        flag = "--" + key.replace("_", "-")
        if key in ("diagnostics", "diag_skip_hooks", "diag_skip_buckets"):
            if value:
                args.append(flag)
            continue
        if isinstance(value, bool):
            raise ValueError(f"Unexpected bool value for non-flag key {key!r}: {value!r}")
        if isinstance(value, (list, tuple)):
            args.append(flag)
            args.extend(str(v) for v in value)
        else:
            args.append(flag)
            args.append(str(value))
    return args


def estimate_batches_per_epoch(config: dict) -> int:
    from stage_1.spice_dataset import SPICEDataset
    from stage_1.train_spice import compute_split_sizes

    dataset = SPICEDataset(
        hdf5_paths=config["hdf5_paths"], subsets=config.get("subsets"),
        exclude_solvated=True, box_padding=config["box_padding"], edge_cutoff=config["edge_cutoff"],
    )
    test_fraction = config.get("test_fraction", 0.05)
    n_train, _n_val, _n_test = compute_split_sizes(len(dataset), config["val_fraction"], test_fraction)
    return max(n_train // config["batch_size"], 1)


def _start_gpu_profiler(run_dir: str, interval_s: float) -> tuple[subprocess.Popen | None, str]:
    csv_path = os.path.join(run_dir, "gpu_utilization.csv")
    cmd = [sys.executable, "-m", GPU_PROFILER_MODULE, "--csv-path", csv_path, "--interval", str(interval_s)]
    try:
        proc = subprocess.Popen(cmd, cwd=str(SRC_DIR), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        return None, csv_path
    return proc, csv_path


def _stop_gpu_profiler(proc: subprocess.Popen | None, csv_path: str) -> dict | None:
    if proc is None:
        return None
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    except Exception:
        pass
    from diagnostics.gpu_utilization_profiler import summarize
    return summarize(csv_path)


def _run_subprocess(
    cmd: list[str], run_dir: str, label: str, timeout_s: float | None,
    profile_gpu: bool = False, gpu_sample_interval: float = 1.0,
) -> dict:
    os.makedirs(run_dir, exist_ok=True)
    stdout_path = os.path.join(run_dir, "stdout.log")
    meta = {
        "label": label, "cmd": cmd, "run_dir": run_dir,
        "start_time": datetime.now(timezone.utc).isoformat(),
    }

    gpu_proc, gpu_csv_path = (None, None)
    if profile_gpu:
        gpu_proc, gpu_csv_path = _start_gpu_profiler(run_dir, gpu_sample_interval)

    t0 = time.monotonic()
    try:
        with open(stdout_path, "w") as logf:
            proc = subprocess.run(
                cmd, cwd=str(SRC_DIR), stdout=logf, stderr=subprocess.STDOUT, timeout=timeout_s,
            )
        meta["returncode"] = proc.returncode
        meta["timed_out"] = False
    except subprocess.TimeoutExpired:
        meta["returncode"] = None
        meta["timed_out"] = True
    except Exception as e:
        meta["returncode"] = None
        meta["timed_out"] = False
        meta["exception"] = repr(e)

    meta["elapsed_s"] = time.monotonic() - t0
    meta["end_time"] = datetime.now(timezone.utc).isoformat()
    meta["stdout_log"] = stdout_path

    if profile_gpu:
        meta["gpu_utilization"] = _stop_gpu_profiler(gpu_proc, gpu_csv_path)
        meta["gpu_utilization_csv"] = gpu_csv_path

    return meta


def _status_str(meta: dict) -> str:
    if meta.get("returncode") == 0:
        return "OK"
    if meta.get("timed_out"):
        return "TIMEOUT"
    return f"FAILED(rc={meta.get('returncode')})"


def run_train_spice(
    cli_overrides: dict,
    run_dir: str,
    label: str,
    timeout_s: float | None = None,
    python_exe: str | None = None,
    profile_gpu: bool = True,
    gpu_sample_interval: float = 1.0,
) -> dict:
    os.makedirs(run_dir, exist_ok=True)
    ckpt_dir = os.path.join(run_dir, "ckpt")
    csv_path = os.path.join(run_dir, "loss_log.csv")
    overrides = dict(cli_overrides)
    overrides.setdefault("ckpt_dir", ckpt_dir)
    overrides.setdefault("csv_path", csv_path)

    cmd = [python_exe or sys.executable, "-m", TRAIN_MODULE] + to_cli_args(overrides)
    meta = _run_subprocess(cmd, run_dir, label, timeout_s, profile_gpu=profile_gpu, gpu_sample_interval=gpu_sample_interval)
    meta["cli_overrides"] = overrides
    meta["csv_path"] = csv_path
    meta["ckpt_dir"] = ckpt_dir
    meta["metrics"] = extract_run_metrics(csv_path)

    with open(os.path.join(run_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    gpu_util = (meta.get("gpu_utilization") or {}).get("mean_utilization_pct")
    print(f"[phase2_runner] {label}: {_status_str(meta)}  elapsed={meta['elapsed_s']:.1f}s  "
          f"n_rows={meta['metrics'].get('n_train_batch_rows', 0)}  "
          f"seconds_per_batch={meta['metrics'].get('seconds_per_batch')}  "
          f"mean_gpu_util={gpu_util}")

    return meta


def run_diagnostics_script(
    module: str,
    args: list[str],
    run_dir: str,
    label: str,
    timeout_s: float | None = None,
    python_exe: str | None = None,
    json_out_arg: str | None = "json-out",
    profile_gpu: bool = False,
    gpu_sample_interval: float = 1.0,
) -> dict:
    os.makedirs(run_dir, exist_ok=True)
    result_json_path = os.path.join(run_dir, "result.json")
    full_args = list(args)
    if json_out_arg:
        full_args += [f"--{json_out_arg}", result_json_path]
    cmd = [python_exe or sys.executable, "-m", module] + full_args

    meta = _run_subprocess(cmd, run_dir, label, timeout_s, profile_gpu=profile_gpu, gpu_sample_interval=gpu_sample_interval)
    meta["module"] = module
    meta["args"] = full_args

    meta["result"] = None
    if os.path.exists(result_json_path):
        try:
            with open(result_json_path) as f:
                meta["result"] = json.load(f)
        except json.JSONDecodeError:
            pass

    with open(os.path.join(run_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"[phase2_runner] {label}: {_status_str(meta)}  elapsed={meta['elapsed_s']:.1f}s")
    return meta