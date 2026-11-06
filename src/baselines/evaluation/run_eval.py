from __future__ import annotations

import argparse
import json
import os

from baselines.evaluation.evaluate import evaluate_file

MODELS = ("schnet", "painn", "nequip", "allegro", "mace", "equiformerv2")
EVAL_SPLITS = ("val", "test")


def build_calculator(model: str, record: dict, run_dir: str, device: str):
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}, got {model!r}")
    if model in ("schnet", "painn"):
        from baselines.trainers import schnetpack_wrapper as w

        return w.load_calculator(run_dir, device)
    if model in ("nequip", "allegro"):
        from baselines.trainers import nequip_wrapper as w

        return w.load_calculator(w.deployed_path(run_dir), record["species"], device)
    if model == "mace":
        from baselines.trainers import mace_wrapper as w

        return w.load_calculator(w.final_model_path(run_dir), device)
    from baselines.trainers import fairchem_wrapper as w

    return w.load_calculator(run_dir, device)


def evaluate_run(model: str, data_dir: str, run_dir: str, device: str = "cuda", seed: int | None = None, out_path: str | None = None) -> dict:
    with open(os.path.join(data_dir, "split.json")) as f:
        record = json.load(f)
    calculator = build_calculator(model, record, run_dir, device)
    result: dict = {"model": model, "dataset": record["dataset"], "split": record["split"], "seed": seed, "export_dir": data_dir}
    for name in EVAL_SPLITS:
        info = record["splits"][name]
        if info["n"] == 0:  # SPICE follows MACE-OFF23's train/test-only protocol: no validation set exists
            result[name] = None
            continue
        result[name] = evaluate_file(calculator, os.path.join(data_dir, info["file"]))
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        tmp = f"{out_path}.tmp"
        with open(tmp, "w") as f:
            json.dump(result, f, indent=1)
        os.replace(tmp, out_path)  
    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a trained baseline on the val and test files of its export directory.")
    p.add_argument("--model", required=True, choices=MODELS)
    p.add_argument("--data-dir", required=True, help="export directory (split.json, val.xyz, test.xyz)")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=None, help="recorded in the metrics file only")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = evaluate_run(args.model, args.data_dir, args.run_dir, args.device, args.seed, args.out)
    print(json.dumps({k: (v if not isinstance(v, dict) else {m: v[m] for m in v if "mae" in m}) for k, v in result.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
