from __future__ import annotations

import argparse
import itertools
import json
import sys
import time

from torch.utils.data import DataLoader

from diagnostics.common import CSVWriter
from stage_1.spice_dataset import SPICEDataset, spice_collate_fn


def build_dataset(config: dict) -> SPICEDataset:
    return SPICEDataset(
        hdf5_paths=config["hdf5_paths"], subsets=config.get("subsets"),
        exclude_solvated=True, box_padding=config["box_padding"], edge_cutoff=config["edge_cutoff"],
    )


def _time_loader(loader: DataLoader, n_batches: int) -> dict:
    it = iter(loader)
    try:
        next(it)  
    except StopIteration:
        it = iter(loader)

    n_samples, n_done = 0, 0
    t0 = time.perf_counter()
    for _ in range(n_batches):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        n_samples += batch["all_solute_x"].shape[0]
        n_done += 1
    elapsed = time.perf_counter() - t0
    return {
        "n_batches": n_done, "elapsed_s": elapsed,
        "samples_per_sec": n_samples / elapsed if elapsed > 0 else 0.0,
        "batches_per_sec": n_done / elapsed if elapsed > 0 else 0.0,
    }


def sweep(config: dict, num_workers_grid: list[int], batches_per_config: int, csv_path: str) -> list[dict]:
    dataset = build_dataset(config)
    bs = config["batch_size"]
    results: list[dict] = []
    csv = CSVWriter(csv_path, header=[
        "variant", "num_workers", "pin_memory", "persistent_workers", "prefetch_factor",
        "n_batches", "elapsed_s", "samples_per_sec", "batches_per_sec",
    ])

    def _run(variant: str, nw: int, pin_memory: bool, persistent_workers: bool, prefetch_factor: int) -> dict:
        kwargs = dict(
            batch_size=bs, shuffle=True, drop_last=True, collate_fn=spice_collate_fn,
            num_workers=nw, pin_memory=pin_memory,
        )
        pw = persistent_workers if nw > 0 else None
        pf = prefetch_factor if nw > 0 else None
        if nw > 0:
            kwargs["persistent_workers"] = persistent_workers
            kwargs["prefetch_factor"] = prefetch_factor
        loader = DataLoader(dataset, **kwargs)
        timing = _time_loader(loader, batches_per_config)
        row = {"variant": variant, "num_workers": nw, "pin_memory": pin_memory, "persistent_workers": pw, "prefetch_factor": pf, **timing}
        results.append(row)
        csv.writerow([variant, nw, pin_memory, pw, pf, timing["n_batches"], timing["elapsed_s"], timing["samples_per_sec"], timing["batches_per_sec"]])
        print(f"[spice_dataloader_sweep] {variant:28s} num_workers={nw:2d} pin_memory={str(pin_memory):5s} "
              f"persistent_workers={str(pw):5s}  {timing['samples_per_sec']:8.1f} samples/sec")
        del loader
        return row

    for nw in num_workers_grid:
        _run(f"num_workers={nw}", nw, pin_memory=True, persistent_workers=(nw > 0), prefetch_factor=2)

    positive_nw_results = [r for r in results if r["num_workers"] > 0]
    if positive_nw_results:
        best_nw = max(positive_nw_results, key=lambda r: r["samples_per_sec"])["num_workers"]
        for pin_memory, persistent_workers in itertools.product([False, True], [False, True]):
            _run(f"ablation(nw={best_nw})", best_nw, pin_memory, persistent_workers, prefetch_factor=2)

    csv.close()
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hdf5-paths", type=str, nargs="+", required=True)
    p.add_argument("--subsets", type=str, nargs="+", default=["pubchem"])
    p.add_argument("--box-padding", type=float, default=12.0)
    p.add_argument("--edge-cutoff", type=float, default=5.0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers-grid", type=int, nargs="+", default=[0, 2, 4, 8])
    p.add_argument("--batches-per-config", type=int, default=100)
    p.add_argument("--csv-path", type=str, default="spice_dataloader_sweep.csv")
    p.add_argument("--json-out", type=str, default=None)
    args = p.parse_args()

    config = {
        "hdf5_paths": args.hdf5_paths, "subsets": args.subsets, "box_padding": args.box_padding,
        "edge_cutoff": args.edge_cutoff, "batch_size": args.batch_size,
    }
    results = sweep(config, args.num_workers_grid, args.batches_per_config, args.csv_path)

    best = max(results, key=lambda r: r["samples_per_sec"])
    print(f"\nBest config: {best['variant']} (num_workers={best['num_workers']}, "
          f"pin_memory={best['pin_memory']}, persistent_workers={best['persistent_workers']}) "
          f"-> {best['samples_per_sec']:.1f} samples/sec")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"config": config, "results": results, "best": best}, f, indent=2, default=str)
        print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    sys.exit(main())