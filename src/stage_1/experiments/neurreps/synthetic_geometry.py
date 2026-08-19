from __future__ import annotations

import math
from typing import Optional, Tuple

import networkx as nx
import torch

DEFAULT_CUTOFF = 5.0
DEFAULT_BOND_LENGTH = 2.0
DEFAULT_CONFINEMENT_RADIUS = 25.0
DEFAULT_MIN_SEPARATION_MARGIN = 0.5

TOPOLOGIES = ("chain", "folded_chain", "clique_chain")


def _random_unit_vector(rng: torch.Generator) -> torch.Tensor:
    v = torch.randn(3, generator=rng)
    n = v.norm()
    if n < 1e-8:
        return _random_unit_vector(rng)
    return v / n


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
    positions: torch.Tensor, cutoff: float, source_idx: int, target_idx: int,
) -> Tuple[nx.Graph, Optional[int], float]:
    n = positions.shape[0]
    dmat = torch.cdist(positions, positions)
    adj = (dmat <= cutoff) & ~torch.eye(n, dtype=torch.bool)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    ii, jj = adj.nonzero(as_tuple=True)
    G.add_edges_from(zip(ii.tolist(), jj.tolist()))
    try:
        r = nx.shortest_path_length(G, source_idx, target_idx)
    except nx.NetworkXNoPath:
        r = None
    euclid = float((positions[source_idx] - positions[target_idx]).norm())
    return G, r, euclid


def _try_place_next_atom(
    positions: torch.Tensor,
    bond_length: float,
    confinement_radius: float,
    min_separation: float,
    local_window: int,
    persistence: float,
    max_attempts: int,
    rng: torch.Generator,
) -> Optional[torch.Tensor]:
    n = positions.shape[0]
    prev_dir = positions[-1] - positions[-2] if n >= 2 else _random_unit_vector(rng)
    prev_dir = prev_dir / prev_dir.norm()
    far = positions[: max(0, n - local_window)]

    for _ in range(max_attempts):
        new_dir = _perturb_direction(prev_dir, persistence, rng) if n >= 2 else _random_unit_vector(rng)
        candidate = positions[-1] + bond_length * new_dir
        if candidate.norm() > confinement_radius:
            continue
        if far.shape[0] > 0:
            dists = (far - candidate).norm(dim=-1)
            if (dists < min_separation).any():
                continue
        return torch.cat([positions, candidate.unsqueeze(0)], dim=0)
    return None


def grow_confined_saw_to_r(
    r_target: int,
    cutoff: float = DEFAULT_CUTOFF,
    bond_length: float = DEFAULT_BOND_LENGTH,
    confinement_radius: float = DEFAULT_CONFINEMENT_RADIUS,
    local_window: Optional[int] = None,
    persistence: float = 0.6,
    max_attempts_per_atom: int = 200,
    max_atoms: int = 500,
    max_backtrack_steps: int = 4000,
    rng: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if rng is None:
        rng = torch.Generator().manual_seed(0)
    if local_window is None:
        local_window = max(2, math.ceil(cutoff / bond_length) + 2)
    min_separation = cutoff + DEFAULT_MIN_SEPARATION_MARGIN

    positions = torch.zeros(1, 3)
    if r_target <= 0:
        return positions
    positions = torch.cat([positions, positions[0:1] + bond_length * _random_unit_vector(rng)], dim=0)
    dist_from_source = [0, 1]

    backtracks = 0
    while True:
        n = positions.shape[0]
        if dist_from_source[-1] >= r_target:
            return positions
        if n >= max_atoms:
            raise RuntimeError(
                f"grow_confined_saw_to_r: exceeded max_atoms={max_atoms} while only reaching "
                f"r={dist_from_source[-1]} (target {r_target})."
            )
        candidate = _try_place_next_atom(
            positions, bond_length, confinement_radius, min_separation, local_window,
            persistence, max_attempts_per_atom, rng,
        )
        if candidate is not None:
            new_pos = candidate[-1]
            window_start = max(0, n - local_window)
            nearby = positions[window_start:]
            dists = (nearby - new_pos).norm(dim=-1)
            neighbor_local = (dists <= cutoff).nonzero(as_tuple=True)[0].tolist()
            new_dist = min(dist_from_source[window_start + i] for i in neighbor_local) + 1
            positions = candidate
            dist_from_source.append(new_dist)
            continue

        backtracks += 1
        if backtracks > max_backtrack_steps:
            raise RuntimeError(
                f"grow_confined_saw_to_r: exceeded max_backtrack_steps={max_backtrack_steps} "
                f"while only reaching r={dist_from_source[-1]} (target {r_target}, n_atoms={n}). "
                f"This r is likely infeasible for confinement_radius={confinement_radius}, "
                f"bond_length={bond_length}, cutoff={cutoff}."
            )
        if positions.shape[0] > 2:
            positions = positions[:-1]
            dist_from_source.pop()


def add_shortcut(
    positions: torch.Tensor,
    i_short: int,
    j_short: int,
    shortcut_length: float = DEFAULT_BOND_LENGTH,
    rng: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if rng is None:
        rng = torch.Generator().manual_seed(0)
    positions = positions.clone()
    direction = _random_unit_vector(rng)
    target_pos = positions[i_short] + shortcut_length * direction
    shift = target_pos - positions[j_short]
    positions[j_short:] = positions[j_short:] + shift
    return positions


def _random_points_in_ball(
    n_points: int, radius: float, rng: torch.Generator, min_pairwise: float,
) -> torch.Tensor:
    positions = torch.zeros(n_points, 3)
    if n_points <= 1:
        return positions
    for i in range(1, n_points):
        placed = False
        for _ in range(2000):
            direction = _random_unit_vector(rng)
            r = radius * (torch.rand(1, generator=rng).item() ** (1.0 / 3.0))
            candidate = direction * r
            dists = (positions[:i] - candidate).norm(dim=-1)
            if (dists < min_pairwise).any():
                continue
            positions[i] = candidate
            placed = True
            break
        if not placed:
            raise RuntimeError(f"_random_points_in_ball: could not place point {i}/{n_points}")
    return positions


def build_clique_chain_positions(
    clique_size: int,
    chain_r_target: int,
    bond_length: float = DEFAULT_BOND_LENGTH,
    clique_radius: Optional[float] = None,
    confinement_radius: float = DEFAULT_CONFINEMENT_RADIUS,
    cutoff: float = DEFAULT_CUTOFF,
    rng: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if rng is None:
        rng = torch.Generator().manual_seed(0)
    if clique_radius is None:
        clique_radius = cutoff * 0.3

    clique_pos = _random_points_in_ball(
        clique_size, clique_radius, rng, min_pairwise=bond_length * 0.3,
    )
    chain_pos = grow_confined_saw_to_r(
        max(1, chain_r_target), cutoff=cutoff, bond_length=bond_length,
        confinement_radius=confinement_radius, rng=rng,
    )

    anchor = clique_pos[0]
    away_dir = anchor / anchor.norm() if anchor.norm() > 1e-6 else _random_unit_vector(rng)
    bridge_target = anchor + bond_length * away_dir
    chain_pos = chain_pos + (bridge_target - chain_pos[0])

    return torch.cat([clique_pos, chain_pos], dim=0)


def build_topology_geometry(
    topology: str,
    r: int,
    cutoff: float = DEFAULT_CUTOFF,
    bond_length: float = DEFAULT_BOND_LENGTH,
    confinement_radius: float = DEFAULT_CONFINEMENT_RADIUS,
    rng: Optional[torch.Generator] = None,
    max_retries: int = 8,
) -> dict:
    if rng is None:
        rng = torch.Generator().manual_seed(0)

    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            return _build_topology_geometry_once(
                topology, r, cutoff, bond_length, confinement_radius, rng,
            )
        except RuntimeError as e:
            last_error = e
            rng = torch.Generator().manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=rng).item()))
    raise RuntimeError(
        f"build_topology_geometry: failed {max_retries} times for topology={topology!r}, r={r} "
        f"(last error: {last_error})"
    )


def _build_topology_geometry_once(
    topology: str,
    r: int,
    cutoff: float,
    bond_length: float,
    confinement_radius: float,
    rng: torch.Generator,
) -> dict:
    if topology == "chain":
        positions = grow_confined_saw_to_r(
            r, cutoff=cutoff, bond_length=bond_length, confinement_radius=confinement_radius, rng=rng,
        )
        source_idx, target_idx = 0, positions.shape[0] - 1

    elif topology == "folded_chain":
        base = grow_confined_saw_to_r(
            max(2, int(round(2.2 * r))), cutoff=cutoff, bond_length=bond_length,
            confinement_radius=confinement_radius, rng=rng,
        )
        n_atoms = base.shape[0]
        i_short, j_short = n_atoms // 4, (3 * n_atoms) // 4
        positions = add_shortcut(base, i_short, j_short, shortcut_length=bond_length, rng=rng)
        source_idx, target_idx = 0, n_atoms - 1

    elif topology == "clique_chain":
        clique_size = max(3, r // 4 + 3)
        positions = build_clique_chain_positions(
            clique_size, max(1, r - 1), bond_length=bond_length,
            confinement_radius=confinement_radius, cutoff=cutoff, rng=rng,
        )
        source_idx, target_idx = 0, positions.shape[0] - 1

    else:
        raise ValueError(f"Unknown topology {topology!r}, expected one of {TOPOLOGIES}")

    graph, realized_r, realized_euclid = realized_graph_and_r(positions, cutoff, source_idx, target_idx)
    if realized_r is None:
        raise RuntimeError(
            f"build_topology_geometry: source and target are disconnected under the cutoff "
            f"graph for topology={topology!r}, nominal r={r}. Geometry parameters need tuning."
        )

    return {
        "topology": topology,
        "nominal_r": r,
        "positions": positions,
        "source_idx": source_idx,
        "target_idx": target_idx,
        "realized_r": realized_r,
        "realized_euclid": realized_euclid,
        "graph": graph,
    }
