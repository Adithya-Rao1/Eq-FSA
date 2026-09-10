from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from diagnostics import counters

FEATURE_DIMS = {0: 1, 1: 3, 2: 5}

def safe_inv(V: torch.Tensor, jitter: float = 1e-6, max_tries: int = 5,
             min_singular_value: float = 1e-2) -> torch.Tensor:
    orig_dtype = V.dtype

    diag = torch.diagonal(V, dim1=-2, dim2=-1)
    off_diag_mask = ~torch.eye(V.shape[-1], dtype=torch.bool, device=V.device)
    off_diag_max = (V * off_diag_mask).abs().amax(dim=(-2, -1))
    is_clean_diagonal = (
        (off_diag_max < 1e-10)
        & (diag.abs().amin(dim=-1) >= min_singular_value)
        & torch.isfinite(diag).all(dim=-1)
    )
    if bool(is_clean_diagonal.all()):
        return torch.diag_embed(1.0 / diag).to(orig_dtype)

    V_work = V.clone()
    I = torch.eye(V.shape[-1], device=V.device, dtype=V_work.dtype)
    eps = jitter

    for _ in range(max_tries):
        try:
            U, S, Vh = torch.linalg.svd(V_work)
            bad = (S.min(dim=-1).values < min_singular_value) | ~torch.isfinite(S).all(dim=-1)
            if bad.any():
                counters.incr("safe_inv_retry")
                V_work = V_work.clone()
                V_work[bad] = V_work[bad] + eps * I
                eps *= 10
                continue
            S_safe = S.clamp(min=min_singular_value)
            V_inv = Vh.mT @ torch.diag_embed(1.0 / S_safe) @ U.mT
            return V_inv.to(orig_dtype)
        except torch._C._LinAlgError:
            counters.incr("safe_inv_retry")
            V_work = V_work + eps * I
            eps *= 10

    U, S, Vh = torch.linalg.svd(V_work)
    S_safe = S.clamp(min=min_singular_value)
    return (Vh.mT @ torch.diag_embed(1.0 / S_safe) @ U.mT).to(orig_dtype)

def to_fractional(x: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    V_inv = safe_inv(V)                    
    s = torch.matmul(V_inv, x.mT).mT 
        
    return s

def wrap_fractional(s: torch.Tensor) -> torch.Tensor:
    return s - torch.floor(s)

def to_cartesian(s: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    return (V.unsqueeze(-3) @ s.unsqueeze(-1)).squeeze(-1)  

def wrap_positions(x: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    s = to_fractional(x, V)
    s_wrapped = wrap_fractional(s)
    return s_wrapped

def minimum_image(dx: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    ds = to_fractional(dx, V)
    ds_mic = ds - torch.round(ds)
    return to_cartesian(ds_mic, V)

def minimum_image_like(dx: torch.Tensor, dx_reference: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Wraps dx into the periodic image chosen by dx_reference, instead of wrapping dx on its own.

    Needed whenever dx has been rescaled in a way that changes its magnitude without changing the
    underlying physical position it represents (e.g. an artificially inflated COM reference).
    Wrapping such a dx independently can flip to a different periodic image purely because the
    rescaling happened to land its fractional coordinate near a rounding boundary (round() has a
    hard discontinuity there) -- producing a spurious, discontinuous jump uncorrelated with any
    real geometric signal. Tying the image choice to a reference with the true physical magnitude
    avoids this: confirmed directly to eliminate a case where an inflated COM's fractional
    coordinate sat at 7.4988-7.5064 across 5 classes, crossing the round-to-nearest boundary
    between two of them and producing a fake, non-generalizable "signal".
    """
    shift = torch.round(to_fractional(dx_reference, V))
    ds = to_fractional(dx, V) - shift
    return to_cartesian(ds, V)

def reciprocal_lattice(V: torch.Tensor) -> torch.Tensor:
    return 2 * math.pi * safe_inv(V).mT

_MILLER_INDICES_CACHE: dict = {}

def make_miller_indices(
    n_max: int | Sequence[int],
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if isinstance(n_max, int):
        h_max = k_max = l_max = n_max
    else:
        h_max, k_max, l_max = n_max

    key = ((h_max, k_max, l_max), torch.device(device), dtype)
    if key in _MILLER_INDICES_CACHE:
        return _MILLER_INDICES_CACHE[key]

    hs = torch.arange(-h_max, h_max + 1, device=device)
    ks = torch.arange(-k_max, k_max + 1, device=device)
    ls = torch.arange(-l_max, l_max + 1, device=device)
    grid = torch.stack(torch.meshgrid(hs, ks, ls, indexing="ij"), dim=-1)

    result = grid.reshape(-1, 3).to(dtype=dtype)
    _MILLER_INDICES_CACHE[key] = result
    return result

def make_wavevectors(V, n_max, e_cutoff=None):
    indices = make_miller_indices(n_max, device=V.device, dtype=V.dtype)  

    if V.ndim == 2:
        Brec = reciprocal_lattice(V)
        kvecs = (Brec @ indices.T).T
        if e_cutoff is not None:
            k2 = (kvecs ** 2).sum(dim=-1)
            mode_mask = k2 / 2 <= e_cutoff
            kvecs = kvecs * mode_mask.unsqueeze(-1).to(kvecs.dtype)  
        else:
            mode_mask = torch.ones(indices.shape[0], dtype=torch.bool, device=V.device)
        return kvecs, indices, mode_mask

    brec = reciprocal_lattice(V)
    with torch.autocast(device_type="cuda", enabled=False):
        kvecs = torch.einsum("bij,mj->bmi", brec, indices)

    if e_cutoff is not None:
        k2 = (kvecs ** 2).sum(dim=-1)
        mode_mask = k2 / 2 <= e_cutoff                     
        kvecs = kvecs * mode_mask.unsqueeze(-1).to(kvecs.dtype)
    else:
        mode_mask = torch.ones(kvecs.shape[:2], dtype=torch.bool, device=V.device)

    return kvecs, indices, mode_mask                      

def compute_molecular_box(
    x: torch.Tensor, padding: float, min_half_extent: float, edge_cutoff: float,
) -> torch.Tensor:
    padding = max(padding, edge_cutoff)
    center = x.mean(dim=0, keepdim=True)
    x_centered = x - center
    cov = x_centered.T @ x_centered / max(x.shape[0], 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    if torch.det(eigvecs) < 0:
        eigvecs = eigvecs.clone()
        eigvecs[:, 0] = -eigvecs[:, 0]
    proj = x_centered @ eigvecs
    half_extent = proj.abs().max(dim=0).values.clamp(min=min_half_extent)
    return eigvecs * (2 * (half_extent + padding)).unsqueeze(0)

class PWProjection(nn.Module):
    """
    out(x) = sum_k Q_k(h_global) * e^{-i k^\top x}
    """
    def __init__(self, n_max: int, h_global_dim: int, e_cutoff: float | None = None):
        super().__init__()
        self.n_max = n_max
        self.e_cutoff = e_cutoff
        self.num_modes = (2 * n_max + 1) ** 3
        self.q_re = nn.Linear(h_global_dim, self.num_modes)
        self.q_im = nn.Linear(h_global_dim, self.num_modes)

    def forward(self, x: torch.Tensor, h_global: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        M = self.num_modes

        kvecs, _, mode_mask = make_wavevectors(V, self.n_max, self.e_cutoff)

        phase = -torch.einsum("bni,bmi->bnm", x, kvecs)    

        mode_mask_f = mode_mask.to(x.dtype)
        if mode_mask_f.dim() == 1:
            mode_mask_f = mode_mask_f.unsqueeze(0).expand(B, -1)
        mode_mask_f = mode_mask_f.unsqueeze(1)   

        cos_kx = torch.cos(phase) * mode_mask_f
        sin_kx = torch.sin(phase) * mode_mask_f

        Q_re = self.q_re(h_global).view(B, M, 1)
        Q_im = self.q_im(h_global).view(B, M, 1)

        out_re = torch.einsum("bnm,bmc->bnc", cos_kx, Q_re) - torch.einsum("bnm,bmc->bnc", sin_kx, Q_im)
        out_im = torch.einsum("bnm,bmc->bnc", sin_kx, Q_re) + torch.einsum("bnm,bmc->bnc", cos_kx, Q_im)

        return torch.cat([out_re, out_im], dim=-1)   # [B, N, 2 * attn_dim]                                             