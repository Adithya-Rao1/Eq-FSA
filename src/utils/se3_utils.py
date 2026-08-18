from math import pi, sqrt
from functools import reduce, lru_cache
from operator import mul
from typing import Dict, Optional, Tuple
import math

import torch
import torch.nn as nn
import numpy as np

from utils.spherical_harm import *
from diagnostics import counters

from e3nn import o3

_CG_CACHE: Dict[Tuple[int, int], Dict[int, torch.Tensor]] = {}
_HE_NONLINEARITIES = {"relu", "leaky_relu", "silu", "gelu", "selu", "elu"}

def init_linear_(linear: nn.Linear, nonlinearity: str = "linear") -> None:
    if nonlinearity in _HE_NONLINEARITIES:
        kaiming_kind = "relu" if nonlinearity == "relu" else "leaky_relu"
        a = 0.01 if nonlinearity == "leaky_relu" else 0.0
        nn.init.kaiming_normal_(linear.weight, a=a, nonlinearity=kaiming_kind)
    else:
        nn.init.xavier_normal_(linear.weight)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)

def init_sequential_(seq: nn.Sequential, nonlinearity: str, final_nonlinearity: str = "linear") -> None:
    linears = [m for m in seq if isinstance(m, nn.Linear)]
    for lin in linears[:-1]:
        init_linear_(lin, nonlinearity)
    if linears:
        init_linear_(linears[-1], final_nonlinearity)


def scalar_ffn(feature_dim: int, hidden_dim: int) -> nn.Sequential:
    seq = nn.Sequential(
        nn.LayerNorm(feature_dim),
        nn.Linear(feature_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.LeakyReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.LeakyReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.LeakyReLU(),
        nn.Linear(hidden_dim, feature_dim),
    )
    init_sequential_(seq, "leaky_relu")
    return seq

def high_ord_ffn(feature_dim, hidden_dim, degree) -> nn.Sequential:
    seq = nn.Sequential(
            EquivariantRMSNorm(feature_dim, degree),
            EquivariantLinear(feature_dim, hidden_dim),
            EquivariantRMSNorm(hidden_dim, degree),
            GatedNonlinearity(hidden_dim),
            EquivariantLinear(hidden_dim, hidden_dim),
            EquivariantRMSNorm(hidden_dim, degree),
            GatedNonlinearity(hidden_dim),
            EquivariantLinear(hidden_dim, hidden_dim),
            EquivariantRMSNorm(hidden_dim, degree),
            GatedNonlinearity(hidden_dim),
            EquivariantLinear(hidden_dim, feature_dim),
        )
    
    return seq

def cartesian_dipole(n: torch.Tensor) -> torch.Tensor:
    return n

def cartesian_quadrupole(n: torch.Tensor) -> torch.Tensor:
    X, Y, Z = n[:, 0], n[:, 1], n[:, 2]
    Qxx = X * X - 1.0 / 3.0
    Qyy = Y * Y - 1.0 / 3.0
    Qxy = X * Y
    Qxz = X * Z
    Qyz = Y * Z
    return torch.stack([Qxx, Qxy, Qxz, Qyy, Qyz], dim=-1)

def clebsch_gordan_matrix(l: int, k: int) -> Dict[int, torch.Tensor]:
    if (l, k) in _CG_CACHE:
        return _CG_CACHE[(l, k)]

    result: Dict[int, torch.Tensor] = {}
    for J in range(abs(l - k), l + k + 1):
        w3j = o3.wigner_3j(l, k, J) * math.sqrt(2 * J + 1)
        result[J] = w3j.reshape((2 * l + 1) * (2 * k + 1), 2 * J + 1)

    _CG_CACHE[(l, k)] = result
    return result

class RadialNetwork(nn.Module):
    def __init__(
        self,
        num_basis:     int   = 8,
        hidden_dim:    int   = 32,
        cutoff_radius: float = 5.0,
        num_heads:     int = 1,
        edge_feature_dim: int = 0,
        envelope_floor: float = 0.02,
    ):
        super().__init__()
        self.num_basis     = num_basis
        self.cutoff_radius = cutoff_radius
        self.num_heads     = num_heads
        self.edge_feature_dim = edge_feature_dim
        self.envelope_floor = envelope_floor

        centres = torch.linspace(0.0, cutoff_radius, num_basis)
        self.register_buffer("centres", centres)

        width = cutoff_radius / max(num_basis - 1, 1)
        self.register_buffer(
            "inv_width_sq",
            torch.tensor(-1.0 / (2.0 * width ** 2)),
        )
        self.net = nn.Sequential(
            nn.Linear(num_basis + edge_feature_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_heads, bias=False),
        )
        init_sequential_(self.net, "silu")

    def _basis(self, r: torch.Tensor) -> torch.Tensor:
        diff    = r - self.centres.unsqueeze(0)
        rbf     = torch.exp(self.inv_width_sq * diff * diff)

        r_scaled = (r / self.cutoff_radius).clamp(max=1.0)
        envelope = (1.0 - self.envelope_floor) * 0.5 * (torch.cos(math.pi * r_scaled) + 1.0) + self.envelope_floor

        return rbf * envelope

    def forward(self, r: torch.Tensor, edge_feat: Optional[torch.Tensor] = None, *args, **kwargs) -> torch.Tensor:
        basis = self._basis(r)
        if self.edge_feature_dim > 0:
            basis = torch.cat([basis, edge_feat], dim=-1)
        return self.net(basis)
              

def equivariant_weight_matrix(
    x:          torch.Tensor,   
    l:          int,
    k:          int,
    radial_net: nn.Module,
) -> torch.Tensor:               
    device, dtype = x.device, x.dtype
    N = x.shape[0]
    W = torch.zeros(N, 2 * l + 1, 2 * k + 1, device=device, dtype=dtype)

    if N == 0:
        return W

    r  = x.norm(dim=-1, keepdim=True)
    cg = clebsch_gordan_matrix(l, k)

    for J in range(abs(l - k), l + k + 1):
        if J not in cg:
            continue

        if isinstance(radial_net, nn.ModuleDict):
            if str(J) not in radial_net:
                continue
            rnet = radial_net[str(J)]
        else:
            rnet = radial_net

        phi = rnet(r)

        Q_J = cg[J].to(device=device, dtype=dtype)

        Y_J = get_spherical_harmonics(J, x)

        QTY = torch.einsum("ji,ni->nj", Q_J, Y_J)
        had = phi * QTY
        W   = W + had.view(N, 2 * l + 1, 2 * k + 1)

    return W

def apply_direct_sum_W(
    f:           Dict[int, torch.Tensor],
    x:           torch.Tensor,
    radial_nets: nn.ModuleDict,
    max_degree:  int,
    k_max:       int = None,
) -> Dict[int, torch.Tensor]:
    out: Dict[int, torch.Tensor] = {}
    k_hi = max_degree if k_max is None else k_max

    for l in range(max_degree + 1):
        any_k = next(iter(f.values()))
        C     = any_k.shape[-2]

        out[l] = torch.zeros(
            *x.shape[:-1], C, 2 * l + 1,
            device=x.device, dtype=x.dtype,
        )

        for k in range(k_hi + 1):
            key = f"{l}_{k}"
            W   = equivariant_weight_matrix(x, l, k, radial_nets[key])
            contrib = torch.einsum("...ij,...cj->...ci", W, f[k])
            out[l]  = out[l] + contrib

    return out


def direct_sum_vmap(tensors):
    n1, n2    = tensors[0].shape[0], tensors[0].shape[1]
    flattened = [t.flatten(0, 1) for t in tensors]
    out_flat  = torch.vmap(torch.block_diag, in_dims=0)(*flattened)
    return out_flat.view(n1, n2, out_flat.shape[1], out_flat.shape[2])


def direct_sum_inner_product(
    a: Dict[int, torch.Tensor],
    b: Dict[int, torch.Tensor],
) -> torch.Tensor:
    result = None
    for l in a:
        if l not in b:
            continue
        dot    = a[l].permute(0, 1, 3, 2) @ b[l]
        result = dot if result is None else direct_sum_vmap([result, dot])
    if result is None:
        raise ValueError("No shared degrees between query and key.")
    return result


def softmax_over_neighbors(
    scores: torch.Tensor,
    mask:   torch.Tensor = None,
) -> torch.Tensor:
    if mask is not None:
        scores = scores.masked_fill(~mask, float(-1e9))
    attn = torch.nn.functional.softmax(scores + 1e-8, dim=-1)
    return attn

def scatter_softmax(
    scores: torch.Tensor,
    index:  torch.Tensor,
    num_groups: int,
) -> torch.Tensor:
    group_max = torch.full((num_groups,), float("-inf"), device=scores.device, dtype=scores.dtype)
    group_max = group_max.scatter_reduce(0, index, scores, reduce="amax", include_self=True)
    group_max = torch.where(torch.isinf(group_max), torch.zeros_like(group_max), group_max)
    exp_scores = torch.exp(scores - group_max[index] + 1e-8)
    denom = torch.zeros(num_groups, device=scores.device, dtype=scores.dtype).scatter_add(0, index, exp_scores)
    return exp_scores / denom[index].clamp(min=1e-12)

def hadamard_exp_kernel(
    q_i: torch.Tensor,
    k_j: torch.Tensor,
    v_j: torch.Tensor
):
    exp_qi = torch.exp(q_i)
    exp_kj = torch.exp(k_j) 
    


def verify_cg_orthogonality(max_degree: int = 2, tol: float = 1e-5) -> None:
    print("Verifying CG matrix orthogonality...")
    all_ok = True
    for l in range(max_degree + 1):
        for k in range(max_degree + 1):
            cg     = clebsch_gordan_matrix(l, k)
            Q_full = torch.cat([cg[J] for J in sorted(cg.keys())], dim=1)
            QQT    = Q_full @ Q_full.T
            eye    = torch.eye(Q_full.shape[0], dtype=QQT.dtype)
            err    = (QQT - eye).abs().max().item()
            status = "OK" if err < tol else "FAIL"
            if err >= tol:
                all_ok = False
            print(f"  (l={l}, k={k}): max |QQ^T - I| = {err:.2e}  [{status}]")

    if all_ok:
        print("All CG matrices passed orthogonality check.")
    else:
        raise ValueError(
            "One or more CG matrices failed the orthogonality check."
        )

def fit_sh_to_cartesian(l: int, target_fn, n_samples: int = 20000, tol: float = 1e-4) -> torch.Tensor:
    n = torch.randn(n_samples, 3, dtype=torch.float64)
    n = n / n.norm(dim=-1, keepdim=True)

    Y_l = get_spherical_harmonics(l, n)
    T   = target_fn(n)

    M, *_ = torch.linalg.lstsq(Y_l, T)
    resid = (Y_l @ M - T).abs().max().item()
    assert resid < tol, f"SH{l}->Cartesian fit residual too high: {resid:.2e}"

    return M.T

class AggregateFeatures(nn.Module):
    def __init__(self, out_feat_dim, inp_feat_dim, hidden_dim=128):
        super().__init__()
        self.f0_norm = nn.Linear(inp_feat_dim, 1)
        init_linear_(self.f0_norm, "linear")
        self.f1_norm = EquivariantNormReadout(inp_feat_dim)
        self.f2_norm = EquivariantNormReadout(inp_feat_dim)

        self.f0_proj = EquivariantLinear(1, out_feat_dim)
        self.f1_proj = EquivariantLinear(1, out_feat_dim)
        self.f2_proj = EquivariantLinear(1, out_feat_dim)

        self.out = nn.Sequential(
            nn.Linear(3*out_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, out_feat_dim)
        )
        init_sequential_(self.out, "selu")

    def forward(self, f0, f1, f2):
        f0_inv = self.f0_norm(f0.squeeze(-1)).unsqueeze(-1)
        f1_inv = self.f1_norm(f1).unsqueeze(-1).unsqueeze(-1)
        f2_inv = self.f2_norm(f2).unsqueeze(-1).unsqueeze(-1)

        f0_proj = self.f0_proj(f0_inv).squeeze(-1)
        f1_proj = self.f1_proj(f1_inv).squeeze(-1)
        f2_proj = self.f2_proj(f2_inv).squeeze(-1)

        f_out = self.out(torch.cat([f0_proj, f1_proj, f2_proj], dim=-1)).unsqueeze(-1)

        return f_out

class IrrepLinear(nn.Module):
    def __init__(self, channels, nonlinearity: str = "linear"):
        super().__init__()
        self.weight = nn.Linear(channels, channels, bias=False)
        init_linear_(self.weight, nonlinearity)

    def forward(self, x):
        x = x.transpose(-1,-2)
        x = self.weight(x)
        return x.transpose(-1,-2)

class EquivariantReadout(nn.Module):
    def __init__(self, C: int, hidden: int = 64):
        super().__init__()
        self.proj = nn.Linear(C, 1, bias=False)
        init_linear_(self.proj, "linear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(-1, -2)
        x = self.proj(x)             
        return x.squeeze(-1)        

def straight_through_norm(
    x: torch.Tensor, dim: int = -1, keepdim: bool = False, floor: float = 0.5,
) -> torch.Tensor:
    sq = (x ** 2).sum(dim=dim, keepdim=True)
    norm = torch.sqrt(sq + floor ** 2)
    return norm if keepdim else norm.squeeze(dim)

def straight_through_norm_and_direction(
    x: torch.Tensor, dim: int = -1, floor: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    sq = (x ** 2).sum(dim=dim, keepdim=True)
    norm = torch.sqrt(sq + floor ** 2)
    direction = x / norm
    return norm.squeeze(dim), direction

class GatedNonlinearity(nn.Module):
    def __init__(self, num_channels: int, floor: float = 0.5):
        super().__init__()
        self.floor = floor
        self.gate_lin = nn.Linear(num_channels, num_channels)
        init_linear_(self.gate_lin, "tanh")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = straight_through_norm(x, dim=-1, floor=self.floor)
        gate = torch.tanh(self.gate_lin(norm))
        return x * gate.unsqueeze(-1)

class EquivariantNormReadout(nn.Module):
    def __init__(self, C: int, hidden: int = 64, floor: float = 0.5):
        super().__init__()
        self.floor = floor
        self.equiv_proj = nn.Linear(C, hidden, bias=False)
        init_linear_(self.equiv_proj, "linear")
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden, bias=False),
            nn.SiLU(),
            nn.Linear(hidden, 1, bias=False),
        )
        init_sequential_(self.mlp, "silu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self.equiv_proj(x.transpose(-1, -2))
        v = v.transpose(-1, -2)
        norms = straight_through_norm(v, dim=-1, floor=self.floor)
        return self.mlp(norms).squeeze(-1)

class EquivariantLinear(nn.Module):
    def __init__(self, cin, cout, nonlinearity: str = "linear"):
        super().__init__()
        self.weight = nn.Linear(cin, cout, bias=False)
        init_linear_(self.weight, nonlinearity)

    def forward(self, x):
        return self.weight(x.transpose(-1, -2)).transpose(-1, -2)

class EquivariantRMSNorm(nn.Module):
    def __init__(
        self,
        num_channels: int,
        degree: int,
        eps: float = 1e-8,
        floor: float = 0.5,
    ):
        super().__init__()
        self.degree = degree
        self.eps = eps
        self.floor = floor
        self.gamma = nn.Parameter(torch.ones(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.degree == 0:
            v = x.squeeze(-1)
            rms = v.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
            if counters.DIAG_ENABLED and (rms < self.eps).any():
                counters.incr("rmsnorm_near_zero")
            v = (v / rms) * self.gamma
            return v.unsqueeze(-1)
        else:
            if counters.DIAG_ENABLED and ((x ** 2).sum(dim=-1) < self.floor ** 2).any():
                counters.incr("eqln_deg_gt0_near_degenerate") 
            norm, x_dir = straight_through_norm_and_direction(x, dim=-1, floor=self.floor,)
            rms = norm.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
            target = (norm / rms) * self.gamma
            return x_dir * target.unsqueeze(-1)

def cg_for_J(l: int, k: int, J: int) -> torch.Tensor:
    """Return the CG matrix for a single J from the cached dict."""
    return clebsch_gordan_matrix(l, k)[J]

def equivariant_weight_single_J(
    x:   torch.Tensor,  
    l:   int,
    k:   int,
    J:   int,
    phi: torch.Tensor,  
) -> torch.Tensor:     

    device, dtype = x.device, x.dtype

    if x.shape[0] == 0:
        return torch.zeros(0, 2 * l + 1, 2 * k + 1, device=device, dtype=dtype)

    cg    = clebsch_gordan_matrix(l, k)
    Q_J   = cg[J].to(device=device, dtype=dtype)

    Y_J = get_spherical_harmonics(J, x)

    QTY = torch.einsum("ji,ni->nj", Q_J, Y_J)
    had = phi * QTY                               
    return had.view(x.shape[0], 2 * l + 1, 2 * k + 1)

def flat_to_traceless_symmetric(flat: torch.Tensor) -> torch.Tensor:
    Qxx, Qxy, Qxz, Qyy, Qyz = flat.unbind(-1)
    Qzz = -Qxx - Qyy
    B = flat.shape[0]
    T = torch.zeros(B, 3, 3, device=flat.device, dtype=flat.dtype)
    T[:, 0, 0] = Qxx
    T[:, 1, 1] = Qyy
    T[:, 2, 2] = Qzz
    T[:, 0, 1] = T[:, 1, 0] = Qxy
    T[:, 0, 2] = T[:, 2, 0] = Qxz
    T[:, 1, 2] = T[:, 2, 1] = Qyz
    return T