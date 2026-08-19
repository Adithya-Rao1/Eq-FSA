from __future__ import annotations

import torch

from stage_1.experiments.neurreps.synthetic_geometry import (
    DEFAULT_CUTOFF,
    build_topology_geometry,
    realized_graph_and_r,
)

R_VALUES = [2, 4, 6, 8, 10, 12]


def test_chain_realized_r_matches_nominal():
    for r in R_VALUES:
        rng = torch.Generator().manual_seed(1000 + r)
        inst = build_topology_geometry("chain", r, rng=rng)
        assert inst["realized_r"] == r, f"chain r={r}: realized {inst['realized_r']}"


def test_chain_euclid_distance_bounded():
    confinement_radius = 25.0
    euclids = []
    for r in R_VALUES:
        rng = torch.Generator().manual_seed(2000 + r)
        inst = build_topology_geometry("chain", r, confinement_radius=confinement_radius, rng=rng)
        euclids.append(inst["realized_euclid"])
        assert inst["realized_euclid"] <= 2 * confinement_radius + 1e-6
    first_half_growth = euclids[len(euclids) // 2] - euclids[0]
    second_half_growth = euclids[-1] - euclids[len(euclids) // 2]
    assert second_half_growth < first_half_growth


def test_clique_chain_clique_is_fully_connected():
    rng = torch.Generator().manual_seed(3000)
    r = 8
    inst = build_topology_geometry("clique_chain", r, rng=rng)
    clique_size = max(3, r // 4 + 3)
    positions = inst["positions"]
    clique_pos = positions[:clique_size]
    dmat = torch.cdist(clique_pos, clique_pos)
    assert dmat.max() < DEFAULT_CUTOFF, "not every clique pair is within cutoff of each other"


def test_folded_chain_shorter_than_plain_chain_at_matched_atom_count():
    rng_plain = torch.Generator().manual_seed(4000)
    rng_folded = torch.Generator().manual_seed(4000)
    r = 6
    plain = build_topology_geometry("chain", 2 * r, rng=rng_plain)
    folded = build_topology_geometry("folded_chain", r, rng=rng_folded)

    n = min(plain["positions"].shape[0], folded["positions"].shape[0])
    _, r_plain_matched, _ = realized_graph_and_r(plain["positions"][:n], DEFAULT_CUTOFF, 0, n - 1)
    assert folded["realized_r"] <= r_plain_matched, (
        f"folded_chain's shortcut should shorten realized r vs a plain chain of the same length "
        f"(folded={folded['realized_r']}, plain={r_plain_matched})"
    )


def test_realized_graph_and_r_edge_symmetry():
    rng = torch.Generator().manual_seed(5000)
    inst = build_topology_geometry("chain", 6, rng=rng)
    G = inst["graph"]
    for u, v in G.edges():
        assert G.has_edge(v, u)
        assert u != v
