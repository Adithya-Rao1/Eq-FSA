# MD trajectory observables (RDF, MSD, VACF, NVE energy drift, kinetic temperature)
# Frenkel & Smit, "Understanding Molecular Simulation" (2002) and Allen & Tildesley, "Computer Simulation of Liquids" (2017);
# MLIPAudit (github.com/instadeepai/mlipaudit); Fu et al., "Forces are not Enough" (arXiv:2210.07237).

from __future__ import annotations

import numpy as np

KB_EV_PER_K = 8.617333262e-5
# 1 amu * (Angstrom/fs)^2 in eV; 1.66053906660e-27 kg * 1e-20 m^2 / 1e-30 s^2 / 1.602176634e-19 J/eV
AMU_ANG2_PER_FS2_TO_EV = 103.6426965


def rdf(
    positions: np.ndarray, box: np.ndarray, r_max: float, n_bins: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    # positions (T, N, 3), box (3,) orthorhombic edge lengths; minimum-image convention needs r_max <= min(box)/2.
    positions = np.asarray(positions, dtype=np.float64)
    box = np.asarray(box, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError("positions must have shape (T, N, 3)")
    if box.shape != (3,):
        raise ValueError("box must be orthorhombic edge lengths with shape (3,)")
    if r_max > box.min() / 2 + 1e-12:
        raise ValueError("r_max must be <= half the smallest box edge")
    n_frames, n_atoms, _ = positions.shape
    edges = np.linspace(0.0, r_max, n_bins + 1)
    counts = np.zeros(n_bins, dtype=np.float64)
    iu = np.triu_indices(n_atoms, k=1)
    for frame in positions:
        delta = frame[iu[0]] - frame[iu[1]]
        delta -= box * np.round(delta / box)
        dist = np.linalg.norm(delta, axis=1)
        counts += np.histogram(dist, bins=edges)[0]
    shell_volume = 4.0 / 3.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    # Unordered pairs are counted once, so the ideal-gas expectation is N(N-1)/2 * shell_volume / V per frame.
    expected = n_frames * n_atoms * (n_atoms - 1) / 2.0 * shell_volume / np.prod(box)
    return 0.5 * (edges[1:] + edges[:-1]), counts / expected


def msd(unwrapped_positions: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    # Averaged over atoms and all time origins; positions must be unwrapped (no periodic jumps). Returns Angstrom^2 for lags 0..max_lag in frames.
    x = np.asarray(unwrapped_positions, dtype=np.float64)
    n_frames = x.shape[0]
    max_lag = n_frames - 1 if max_lag is None else min(max_lag, n_frames - 1)
    out = np.zeros(max_lag + 1)
    for lag in range(1, max_lag + 1):
        disp = x[lag:] - x[:-lag]
        out[lag] = np.mean(np.sum(disp**2, axis=-1))
    return out


def vacf(velocities: np.ndarray, max_lag: int | None = None, normalize: bool = True) -> np.ndarray:
    # <v(t0) . v(t0+lag)> averaged over atoms and time origins; normalized so that C(0) = 1 unless normalize=False.
    v = np.asarray(velocities, dtype=np.float64)
    n_frames = v.shape[0]
    max_lag = n_frames - 1 if max_lag is None else min(max_lag, n_frames - 1)
    out = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        out[lag] = np.mean(np.sum(v[: n_frames - lag] * v[lag:], axis=-1))
    if normalize:
        if out[0] == 0.0:
            raise ValueError("cannot normalize a VACF with zero C(0)")
        out = out / out[0]
    return out


def energy_drift(times_fs: np.ndarray, total_energy_ev: np.ndarray, n_atoms: int) -> dict[str, float]:
    # Least-squares slope of total energy vs time. relative_drift = (slope * duration) / |mean(E)|: fractional change over the run.
    t = np.asarray(times_fs, dtype=np.float64)
    e = np.asarray(total_energy_ev, dtype=np.float64)
    if t.shape != e.shape or t.ndim != 1 or t.size < 2:
        raise ValueError("times and energies must be 1D, equal length, with at least 2 samples")
    # Closed-form centered least-squares slope --> centering first makes a constant exactly zero and matches a long-double reference to the last bit on linear data.
    t_centered = t - t.mean()
    time_variance = (t_centered * t_centered).sum()
    if time_variance == 0.0:
        raise ValueError("times must not all be equal")
    slope = float((t_centered * (e - e.mean())).sum() / time_variance)
    duration = float(t[-1] - t[0])
    mean_abs = float(abs(e.mean()))
    return {
        "slope_ev_per_fs": slope,
        "slope_ev_per_atom_per_ps": slope * 1000.0 / n_atoms,
        "total_drift_ev": slope * duration,
        "relative_drift": slope * duration / mean_abs if mean_abs > 0 else float("nan"),
        "duration_fs": duration,
    }


def kinetic_temperature(velocities: np.ndarray, masses_amu: np.ndarray, remove_com_dof: bool = True) -> np.ndarray:
    # velocities (T, N, 3) in Angstrom/fs, masses (N,) in amu -> (T,) in K, using 3N-3 degrees of freedom by default.
    v = np.asarray(velocities, dtype=np.float64)
    m = np.asarray(masses_amu, dtype=np.float64)
    ke_ev = 0.5 * np.sum(m[None, :, None] * v**2, axis=(1, 2)) * AMU_ANG2_PER_FS2_TO_EV
    dof = 3 * v.shape[1] - (3 if remove_com_dof else 0)
    return 2.0 * ke_ev / (dof * KB_EV_PER_K)
