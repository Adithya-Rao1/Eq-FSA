from __future__ import annotations

import sys

import torch

def main() -> None:
    from stage_1.experiments.neurreps.lr_sr.mace_lr_model import batch_configurations
    from stage_1.experiments.neurreps.lr_sr.dataset import generate_configuration

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float32)

    from mace.tools.utils import AtomicNumberTable
    z_table = AtomicNumberTable([1, 8])  # H, O -- matches CHARGE_TO_ELEMENT in mace_lr_model.py
    cutoff = 5.0

    cfg_a = generate_configuration(seed=1)
    cfg_b = generate_configuration(seed=2)
    positions_batch = torch.stack([cfg_a["x"], cfg_b["x"]], dim=0)
    types_batch = torch.stack([cfg_a["types"], cfg_b["types"]], dim=0)
    boxes = [cfg_a["box"], cfg_b["box"]]
    idx_batch = torch.tensor([0, 1], dtype=torch.long)

    # Pass 1: fresh build, populates the cache.
    cache: dict = {}
    batch1 = batch_configurations(positions_batch, types_batch, boxes, z_table, cutoff, device,
                                   idx_batch=idx_batch, graph_cache=cache)
    edge_index_1 = batch1.edge_index.detach().clone()
    node_attrs_1 = batch1.node_attrs.detach().clone()
    shifts_1 = batch1.shifts.detach().clone()

    print(f"Cache populated after pass 1: {len(cache)} entries (expected 2)")
    cached_device_after_pass1 = next(iter(cache.values())).edge_index.device
    print(f"Cached object's own device after pass 1 (should be cpu, NOT mutated to {device}): "
          f"{cached_device_after_pass1}")

    batch2 = batch_configurations(positions_batch, types_batch, boxes, z_table, cutoff, device,
                                   idx_batch=idx_batch, graph_cache=cache)
    edge_index_2 = batch2.edge_index.detach().clone()
    node_attrs_2 = batch2.node_attrs.detach().clone()
    shifts_2 = batch2.shifts.detach().clone()

    edge_index_diff = (edge_index_1 != edge_index_2).sum().item()
    node_attrs_diff = (node_attrs_1 - node_attrs_2).abs().max().item()
    shifts_diff = (shifts_1 - shifts_2).abs().max().item()

    batch3 = batch_configurations(positions_batch, types_batch, boxes, z_table, cutoff, device,
                                   idx_batch=idx_batch, graph_cache=cache)
    edge_index_3 = batch3.edge_index.detach().clone()
    cached_device_after_pass3 = next(iter(cache.values())).edge_index.device

    print(f"\nedge_index diff (pass1 vs pass2, expect 0): {edge_index_diff}")
    print(f"node_attrs max abs diff (pass1 vs pass2, expect 0.0): {node_attrs_diff:.3e}")
    print(f"shifts max abs diff (pass1 vs pass2, expect 0.0): {shifts_diff:.3e}")
    print(f"edge_index diff (pass2 vs pass3, expect 0): {(edge_index_2 != edge_index_3).sum().item()}")
    print(f"Cached object's own device after pass 3 (should still be cpu): {cached_device_after_pass3}")
    print(f"len(cache) after 3 passes (expect still 2, not growing): {len(cache)}")

    print("\n=== VERDICT ===")
    ok = (
        edge_index_diff == 0 and node_attrs_diff == 0.0 and shifts_diff == 0.0
        and str(cached_device_after_pass1) == "cpu" and str(cached_device_after_pass3) == "cpu"
        and len(cache) == 2
    )
    if ok:
        print("SAFE: repeated cache reuse produces bit-identical batches, the cached objects "
              "themselves stay on cpu (never mutated in place by .to(device) or Batch.from_data_list), "
              "and the cache doesn't silently grow. The graph cache is correct.")
    else:
        print("NOT SAFE -- see the diffs/devices above. Do not trust the cached training run until "
              "this is fixed.")


if __name__ == "__main__":
    sys.exit(main())
