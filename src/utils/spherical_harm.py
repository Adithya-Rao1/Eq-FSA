from math import pi, sqrt
from functools import reduce, lru_cache
from operator import mul
from typing import Union

import torch

# Code adopted from se3-transformer-pytorch

CACHE = {}

def clear_spherical_harmonics_cache():
    """No-op stub kept for backward compatibility."""
    CACHE.clear()

@lru_cache(maxsize=1000)
def semifactorial(x: int) -> float:
    """x!! = x * (x-2) * (x-4) * ... (down to 1 or 2)."""
    if x <= 1:
        return 1.0
    return float(reduce(mul, range(x, 1, -2), 1))

@lru_cache(maxsize=1000)
def pochhammer(x: int, k: int) -> float:
    """Rising factorial x^(k) = x(x+1)...(x+k-1)."""
    if k <= 0:
        return 1.0
    return float(reduce(mul, range(x, x + k), 1))

@lru_cache(maxsize=512)
def _sh_norm(l: int, m: int) -> float:
    """
    Normalization factor for the real tesseral harmonic Y_l^m.
    For m == 0:   sqrt((2l+1) / (4*pi))
    For m != 0:   sqrt((2l+1) / (4*pi)) * sqrt(2 / pochhammer(l-|m|+1, 2|m|))
    """
    m_abs = abs(m)
    N = sqrt((2 * l + 1) / (4 * pi))
    if m_abs == 0:
        return N
    return N * sqrt(2.0 / pochhammer(l - m_abs + 1, 2 * m_abs))

@lru_cache(maxsize=512)
def _cs_phase_semifact(m_abs: int) -> float:
    """(-1)^m * (2m-1)!!  — the seed of the P_l^l recursion."""
    return float((-1) ** m_abs * semifactorial(2 * m_abs - 1))

def lpmv(l: int, m: int, x: torch.Tensor) -> torch.Tensor:
    """
    Associated Legendre polynomial P_l^m(x) including Condon-Shortley phase.

    Args:
        l : int  — degree
        m : int  — order (may be negative)
        x : Tensor[N]  — argument (typically cos θ)
    Returns:
        Tensor[N]
    """
    m_abs = abs(m)

    if m_abs > l:
        return torch.zeros_like(x)

    # seed: P_m^m 
    # P_m^m(x) = (-1)^m * (2m-1)!! * (1-x^2)^(m/2)
    pmm: torch.Tensor
    if m_abs == 0:
        pmm = torch.ones_like(x)
    else:
        sin_theta_sq = (1.0 - x * x).clamp(1e-8)      
        pmm = _cs_phase_semifact(m_abs) * sin_theta_sq.pow(m_abs * 0.5)

    if l == m_abs:
        y = pmm
        if m < 0:
            # P_l^{-|m|} = (-1)^|m| * (l-|m|)! / (l+|m|)! * P_l^{|m|}
            y = y * ((-1) ** m_abs / pochhammer(l - m_abs + 1, 2 * m_abs))
        return y

    # one-step: P_{m+1}^m 
    pmmp1 = x * (2 * m_abs + 1) * pmm                     

    if l == m_abs + 1:
        y = pmmp1
        if m < 0:
            y = y * ((-1) ** m_abs / pochhammer(l - m_abs + 1, 2 * m_abs))
        return y

    # standard three-term recurrence up to degree l
    p_prev2 = pmm     # P_{ll}^m
    p_prev1 = pmmp1   # P_{l+1,l}^m  (i.e. P_{m+1}^m at start)

    for ll in range(m_abs + 2, l + 1):
        # (ll - m_abs) * P_ll^m = (2*ll-1)*x*P_{ll-1}^m - (ll+m_abs-1)*P_{ll-2}^m
        p_curr  = ((2 * ll - 1) * x * p_prev1 - (ll + m_abs - 1) * p_prev2) / (ll - m_abs)
        p_prev2 = p_prev1
        p_prev1 = p_curr

    y = p_curr
    if m < 0:
        y = y * ((-1) ** m_abs / pochhammer(l - m_abs + 1, 2 * m_abs))
    return y

def get_spherical_harmonics_element(
    l: int,
    m: int,
    theta: torch.Tensor,   # [N]
    phi:   torch.Tensor,   # [N]
) -> torch.Tensor:          # [N]
    """
    Single (l, m) tesseral spherical harmonic.
    """
    m_abs = abs(m)
    assert m_abs <= l

    norm = _sh_norm(l, m)                              
    leg  = lpmv(l, m_abs, torch.cos(theta))           

    if m == 0:
        return norm * leg

    trig = torch.cos(m * phi) if m > 0 else torch.sin(m_abs * phi)
    return norm * leg * trig

def _cos_sin_multiples(cos_phi: torch.Tensor, sin_phi: torch.Tensor, l: int):
    """
    cos(m*phi), sin(m*phi) for m=0..l via the angle-addition (de Moivre) recursion,
    computed purely from cos(phi)/sin(phi) -- no atan2 or other inverse trig involved,
    so no pole-adjacent curvature blow-up under double-backward (create_graph=True).
    """
    cos_m = [torch.ones_like(cos_phi)]
    sin_m = [torch.zeros_like(sin_phi)]
    if l >= 1:
        cos_m.append(cos_phi)
        sin_m.append(sin_phi)
    for m in range(2, l + 1):
        cos_m.append(cos_m[m - 1] * cos_phi - sin_m[m - 1] * sin_phi)
        sin_m.append(sin_m[m - 1] * cos_phi + cos_m[m - 1] * sin_phi)
    return cos_m, sin_m

def get_spherical_harmonics(
    l:         int,
    cos_theta: torch.Tensor,
    cos_phi:   torch.Tensor,
    sin_phi:   torch.Tensor,
) -> torch.Tensor:
    """
    All 2l+1 tesseral spherical harmonics for degree l.
    """
    scalar_input = cos_theta.dim() == 0
    if scalar_input:
        cos_theta = cos_theta.unsqueeze(0)
        cos_phi   = cos_phi.unsqueeze(0)
        sin_phi   = sin_phi.unsqueeze(0)

    # Pre-compute lpmv for all unique |m| values in one pass
    lpmv_cache: dict[int, torch.Tensor] = {}
    for m_abs in range(l + 1):
        lpmv_cache[m_abs] = lpmv(l, m_abs, cos_theta)  # [N]

    cos_m, sin_m = _cos_sin_multiples(cos_phi, sin_phi, l)

    m_range = list(range(-l, l + 1))
    m_abs_range = [abs(m) for m in m_range]

    trig_cols = []
    for m in m_range:
        m_abs = abs(m)
        if m > 0:
            trig_cols.append(cos_m[m_abs])
        elif m < 0:
            trig_cols.append(sin_m[m_abs])
        else:
            trig_cols.append(torch.ones_like(cos_theta))
    trig = torch.stack(trig_cols, dim=1)

    leg_stack = torch.stack(
        [lpmv_cache[m_abs] for m_abs in m_abs_range],
        dim=1,
    )

    norms = torch.tensor(
        [_sh_norm(l, m) for m in m_range],
        dtype=cos_theta.dtype, device=cos_theta.device,
    )

    Y = norms.unsqueeze(0) * leg_stack * trig

    if scalar_input:
        return Y.squeeze(0)
    return Y

def get_spherical_harmonics_batched(
    l:         int,
    cos_theta: torch.Tensor,
    cos_phi:   torch.Tensor,
    sin_phi:   torch.Tensor,
) -> torch.Tensor:
    """
    All 2l+1 tesseral spherical harmonics for degree l.
    """
    scalar_input = cos_theta.dim() == 0
    if scalar_input:
        cos_theta = cos_theta.unsqueeze(0)
        cos_phi   = cos_phi.unsqueeze(0)
        sin_phi   = sin_phi.unsqueeze(0)

    lpmv_cache = {}
    for m_abs in range(l + 1):
        lpmv_cache[m_abs] = lpmv(l, m_abs, cos_theta)

    cos_m, sin_m = _cos_sin_multiples(cos_phi, sin_phi, l)

    m_range = list(range(-l, l + 1))
    m_abs_range = [abs(m) for m in m_range]

    trig_cols = []
    for m in m_range:
        m_abs = abs(m)
        if m > 0:
            trig_cols.append(cos_m[m_abs])
        elif m < 0:
            trig_cols.append(sin_m[m_abs])
        else:
            trig_cols.append(torch.ones_like(cos_theta))
    trig = torch.stack(trig_cols, dim=-1)

    leg_stack = torch.stack(
        [lpmv_cache[m_abs] for m_abs in m_abs_range],
        dim=-1,
    )

    norms = torch.tensor(
        [_sh_norm(l,m) for m in m_range],
        dtype=cos_theta.dtype,
        device=cos_theta.device,
    )

    Y = norms.view(1,1,-1) * leg_stack * trig

    if scalar_input:
        return Y.squeeze(0)
    return Y

def safe_divide(numerator: torch.Tensor, denom: torch.Tensor, floor: float = 1.0) -> torch.Tensor:
    safe_denom = denom.clamp(min=floor)
    safe = numerator / safe_denom
    true = numerator / denom
    true = torch.where(torch.isfinite(true), true, safe)
    return safe + (true - safe).detach()

def safe_norm(x: torch.Tensor, dim: int = -1, keepdim: bool = False, tol: float = 1e-7) -> torch.Tensor:
    is_degenerate = x.detach().norm(dim=dim, keepdim=True) < tol
    x_safe = torch.where(is_degenerate, torch.ones_like(x), x)
    norm = x_safe.norm(dim=dim, keepdim=True)
    norm = torch.where(is_degenerate, torch.zeros_like(norm), norm)
    return norm if keepdim else norm.squeeze(dim)

def divide_or_zero(numerator: torch.Tensor, denom: torch.Tensor, tol: float = 1e-7) -> torch.Tensor:
    is_degenerate = denom.detach().abs() < tol
    denom_safe = torch.where(is_degenerate, torch.ones_like(denom), denom)
    result = numerator / denom_safe
    return torch.where(is_degenerate, torch.zeros_like(result), result)

def floored_norm(x: torch.Tensor, dim: int = -1, keepdim: bool = False, floor: float = 1e-8) -> torch.Tensor:
    sq = (x ** 2).sum(dim=dim, keepdim=True)
    norm = torch.sqrt(sq + floor ** 2)
    return norm if keepdim else norm.squeeze(dim)

def x_to_trig(x: torch.Tensor, floor: float = 1e-8):
    norms = floored_norm(x, dim=-1, keepdim=True, floor=floor)
    x_norm = x / norms

    cos_theta = -x_norm[..., 2]

    r_xy = floored_norm(x_norm[..., :2], dim=-1, keepdim=False, floor=floor)
    cos_phi = x_norm[..., 0] / r_xy
    sin_phi = x_norm[..., 1] / r_xy

    return cos_theta, cos_phi, sin_phi