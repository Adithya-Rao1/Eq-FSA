from __future__ import annotations

import torch

from stage_1.experiments.neurreps.synthetic_geometry import (
    DEFAULT_CUTOFF,
    build_topology_geometry,
)

R_VALUES = [2, 4, 6, 8, 10, 12, 16, 20]


def test_readout_is_always_node_zero():
    for topology in ["chain", "ring", "lollipop"]:
        rng = torch.Generator().manual_seed(1000)
        inst = build_topology_geometry(topology, 8, rng=rng)
        assert inst["readout_idx"] == 0, topology
        assert inst["label_idx"] != inst["readout_idx"], topology


def test_realized_r_close_to_nominal():
    for topology in ["chain", "ring", "lollipop"]:
        for r in R_VALUES:
            rng = torch.Generator().manual_seed(2000 + r)
            inst = build_topology_geometry(topology, r, rng=rng)
            realized = inst["realized_r"]
            assert realized > 0
            assert abs(realized - r) <= max(2, r // 4), (
                f"{topology} r={r}: realized_r={realized} is too far off nominal"
            )


def test_lollipop_clique_is_fully_connected():
    rng = torch.Generator().manual_seed(3000)
    r = 10
    inst = build_topology_geometry("lollipop", r, rng=rng)
    positions = inst["positions"]
    clique_size = max(3, r // 4 + 3)
    clique_pos = positions[:clique_size]
    dmat = torch.cdist(clique_pos, clique_pos)
    assert dmat.max() < DEFAULT_CUTOFF, "not every clique pair is within cutoff of each other"


def test_graph_is_undirected_no_self_loops():
    rng = torch.Generator().manual_seed(4000)
    for topology in ["chain", "ring", "lollipop"]:
        inst = build_topology_geometry(topology, 8, rng=rng)
        G = inst["graph"]
        for u, v in G.edges():
            assert G.has_edge(v, u)
            assert u != v


def test_ring_realizations_are_stochastic():
    r = 12
    outcomes = set()
    for seed in range(5):
        rng = torch.Generator().manual_seed(5000 + seed)
        inst = build_topology_geometry("ring", r, rng=rng)
        outcomes.add((inst["positions"].shape[0], inst["realized_r"]))
    assert len(outcomes) > 1, f"ring geometry looks deterministic across seeds: {outcomes}"


def test_chain_realizations_are_stochastic():
    r = 12
    outcomes = set()
    for seed in range(5):
        rng = torch.Generator().manual_seed(6000 + seed)
        inst = build_topology_geometry("chain", r, rng=rng)
        outcomes.add(inst["positions"].shape[0])
    assert len(outcomes) > 1, f"chain geometry looks deterministic across seeds: {outcomes}"
