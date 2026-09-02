from __future__ import annotations

import math
from typing import Optional, Tuple

import networkx as nx
import torch

DEFAULT_CUTOFF = 5.0
MIN_BOND_LENGTH = 0.5
MAX_BOND_LENGTH = 2.0

TOPOLOGIES = ("chain", "ring", "lollipop")

FILLER_TYPE_IDX = 0
READOUT_TYPE_IDX = 1
LABEL_TYPE_BASE_IDX = 2  # label node's type = LABEL_TYPE_BASE_IDX + class


def _random_unit_vector(rng: torch.Generator) -> torch.Tensor:
    v = torch.randn(3, generator=rng)
    n = v.norm()
    return v / n if n > 1e-8 else _random_unit_vector(rng)


def _random_bond_length(rng: torch.Generator) -> float:
    return MIN_BOND_LENGTH + (MAX_BOND_LENGTH - MIN_BOND_LENGTH) * torch.rand(1, generator=rng).item()


def _perturb_direction(prev_dir: torch.Tensor, max_angle: float, rng: torch.Generator) -> torch.Tensor:
    rand = torch.randn(3, generator=rng)
    perp = rand - (rand @ prev_dir) * prev_dir
    if perp.norm() < 1e-6:
        rand = torch.randn(3, generator=rng)
        perp = rand - (rand @ prev_dir) * prev_dir
    perp = perp / perp.norm()
    angle = torch.rand(1, generator=rng).item() * max_angle
    new_dir = math.cos(angle) * prev_dir + math.sin(angle) * perp
    return new_dir / new_dir.norm()


def realized_graph_and_r(
    positions: torch.Tensor, cutoff: float, readout_idx: int, label_idx: int,
) -> Tuple[nx.Graph, Optional[int], float]:
    n = positions.shape[0]
    dmat = torch.cdist(positions, positions)
    adj = (dmat <= cutoff) & ~torch.eye(n, dtype=torch.bool)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    ii, jj = adj.nonzero(as_tuple=True)
    G.add_edges_from(zip(ii.tolist(), jj.tolist()))
    try:
        r = nx.shortest_path_length(G, readout_idx, label_idx)
    except nx.NetworkXNoPath:
        r = None
    euclid = float((positions[readout_idx] - positions[label_idx]).norm())
    return G, r, euclid

def grow_chain_to_r(
    r_target: int, cutoff: float = DEFAULT_CUTOFF, persistence: float = 0.5,
    max_atoms: int = 500, rng: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if rng is None:
        rng = torch.Generator().manual_seed(0)
    positions = [torch.zeros(3)]
    dist_from_readout = [0]
    if r_target <= 0:
        return torch.stack(positions)

    first_dir = _random_unit_vector(rng)
    positions.append(positions[-1] + _random_bond_length(rng) * first_dir)
    dist_from_readout.append(1)
    prev_dir = first_dir

    while dist_from_readout[-1] < r_target:
        if len(positions) >= max_atoms:
            raise RuntimeError(
                f"grow_chain_to_r: exceeded max_atoms={max_atoms} at r={dist_from_readout[-1]} "
                f"(target {r_target})"
            )
        new_dir = _perturb_direction(prev_dir, persistence, rng)
        new_pos = positions[-1] + _random_bond_length(rng) * new_dir
        dists = (torch.stack(positions) - new_pos).norm(dim=-1)
        neighbor_idx = (dists <= cutoff).nonzero(as_tuple=True)[0].tolist()
        new_dist = min(dist_from_readout[j] for j in neighbor_idx) + 1
        positions.append(new_pos)
        dist_from_readout.append(new_dist)
        prev_dir = new_dir

    return torch.stack(positions)

def build_ring_positions(nodes: int, bond_length: float) -> torch.Tensor:
    radius = bond_length / (2.0 * math.sin(math.pi / nodes))
    idx = torch.arange(nodes, dtype=torch.float64)
    theta = 2.0 * math.pi * idx / nodes
    x = radius * torch.cos(theta)
    y = radius * torch.sin(theta)
    z = torch.zeros(nodes, dtype=torch.float64)
    return torch.stack([x, y, z], dim=-1).float()

def _random_points_in_ball(n_points: int, radius: float, rng: torch.Generator, min_pairwise: float) -> torch.Tensor:
    positions = torch.zeros(n_points, 3)
    for i in range(1, n_points):
        for _ in range(2000):
            direction = _random_unit_vector(rng)
            r = radius * (torch.rand(1, generator=rng).item() ** (1.0 / 3.0))
            candidate = direction * r
            if (positions[:i] - candidate).norm(dim=-1).min() < min_pairwise:
                continue
            positions[i] = candidate
            break
        else:
            raise RuntimeError(f"_random_points_in_ball: could not place point {i}/{n_points}")
    return positions


def build_lollipop_positions(
    clique_size: int, chain_r_target: int, cutoff: float = DEFAULT_CUTOFF,
    rng: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if rng is None:
        rng = torch.Generator().manual_seed(0)
    clique_radius = cutoff * 0.3
    clique_pos = _random_points_in_ball(clique_size, clique_radius, rng, min_pairwise=MIN_BOND_LENGTH * 0.5)

    anchor = clique_pos[-1]
    away_dir = anchor / anchor.norm() if anchor.norm() > 1e-6 else _random_unit_vector(rng)
    bridge_len = _random_bond_length(rng)

    chain_pos = grow_chain_to_r(max(1, chain_r_target), cutoff=cutoff, rng=rng)
    chain_pos = chain_pos + (anchor + bridge_len * away_dir - chain_pos[0])

    return torch.cat([clique_pos, chain_pos], dim=0)

def build_topology_geometry(
    topology: str, r: int, cutoff: float = DEFAULT_CUTOFF, rng: Optional[torch.Generator] = None,
) -> dict:
    if rng is None:
        rng = torch.Generator().manual_seed(0)

    if topology == "chain":
        positions = grow_chain_to_r(r, cutoff=cutoff, rng=rng)
        readout_idx, label_idx = 0, positions.shape[0] - 1

    elif topology == "ring":
        bond_length = _random_bond_length(rng)
        readout_idx = 0
        best = None
        stall = 0
        candidate_nodes = 3
        while candidate_nodes < 2000 and stall < 40:
            pos = build_ring_positions(candidate_nodes, bond_length)
            label_idx_c = candidate_nodes // 2
            _, r_c, _ = realized_graph_and_r(pos, cutoff, readout_idx, label_idx_c)
            candidate_nodes += 1
            if r_c is None:
                continue
            if best is None or abs(r_c - r) < abs(best[1] - r):
                best = (candidate_nodes - 1, r_c, pos, label_idx_c)
                stall = 0
            else:
                stall += 1
            if r_c == r:
                break
        if best is None:
            raise RuntimeError(f"build_topology_geometry: could not realize ring r={r}")
        nodes, realized_r, positions, label_idx = best

    elif topology == "lollipop":
        clique_size = max(3, r // 4 + 3)
        positions = build_lollipop_positions(clique_size, max(1, r - 1), cutoff=cutoff, rng=rng)
        readout_idx, label_idx = 0, positions.shape[0] - 1

    else:
        raise ValueError(f"Unknown topology {topology!r}, expected one of {TOPOLOGIES}")

    graph, realized_r, realized_euclid = realized_graph_and_r(positions, cutoff, readout_idx, label_idx)
    if realized_r is None:
        raise RuntimeError(
            f"build_topology_geometry: readout and label nodes are disconnected under the cutoff "
            f"graph for topology={topology!r}, nominal r={r}."
        )

    return {
        "topology": topology,
        "nominal_r": r,
        "positions": positions,
        "readout_idx": readout_idx,
        "label_idx": label_idx,
        "source_idx": label_idx,
        "target_idx": readout_idx,
        "realized_r": realized_r,
        "realized_euclid": realized_euclid,
        "graph": graph,
    }
