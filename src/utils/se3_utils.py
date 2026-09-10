from math import pi, sqrt
from functools import reduce, lru_cache
from operator import mul
from typing import Dict, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
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


def scalar_ffn(
    feature_dim: int, hidden_dim: int, in_dim: int | None = None, use_norms: bool = True,
) -> nn.Sequential:
    in_dim = feature_dim if in_dim is None else in_dim
    def _norm(dim: int) -> nn.Module:
        return nn.RMSNorm(dim) if use_norms else nn.Identity()
    seq = nn.Sequential(
        _norm(in_dim),
        nn.Linear(in_dim, hidden_dim),
        _norm(hidden_dim),
        nn.LeakyReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        _norm(hidden_dim),
        nn.LeakyReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        _norm(hidden_dim),
        nn.LeakyReLU(),
        nn.Linear(hidden_dim, feature_dim),
    )
    init_sequential_(seq, "leaky_relu")
    return seq

def high_ord_ffn(
    feature_dim, hidden_dim, degree, in_dim: int | None = None, use_norms: bool = True,
) -> nn.Sequential:
    in_dim = feature_dim if in_dim is None else in_dim
    def _norm(dim: int) -> nn.Module:
        return EquivariantRMSNorm(dim, degree) if use_norms else nn.Identity()
    seq = nn.Sequential(
            _norm(in_dim),
            EquivariantLinear(in_dim, hidden_dim),
            _norm(hidden_dim),
            GatedNonlinearity(hidden_dim),
            EquivariantLinear(hidden_dim, hidden_dim),
            _norm(hidden_dim),
            GatedNonlinearity(hidden_dim),
            EquivariantLinear(hidden_dim, hidden_dim),
            _norm(hidden_dim),
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

def clebsch_gordan_matrix(
    l: int, k: int, dtype: torch.dtype = None, device: torch.device = None,
) -> Dict[int, torch.Tensor]:
    if dtype is None:
        dtype = torch.get_default_dtype()
    key = (l, k, dtype, None if device is None else torch.device(device))
    if key in _CG_CACHE:
        return _CG_CACHE[key]

    result: Dict[int, torch.Tensor] = {}
    for J in range(abs(l - k), l + k + 1):
        w3j = o3.wigner_3j(l, k, J, dtype=dtype) * math.sqrt(2 * J + 1)
        w3j = w3j.reshape((2 * l + 1) * (2 * k + 1), 2 * J + 1)
        if device is not None:
            w3j = w3j.to(device=device)
        result[J] = w3j

    _CG_CACHE[key] = result
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
        linear_readout: bool = False,
        use_norms: bool = True,
    ):
        super().__init__()
        self.num_basis     = num_basis
        self.cutoff_radius = cutoff_radius
        self.num_heads     = num_heads
        self.edge_feature_dim = edge_feature_dim
        self.envelope_floor = envelope_floor
        self.linear_readout = linear_readout

        centres = torch.linspace(0.0, cutoff_radius, num_basis)
        self.register_buffer("centres", centres)

        width = cutoff_radius / max(num_basis - 1, 1)
        self.register_buffer(
            "inv_width_sq",
            torch.tensor(-1.0 / (2.0 * width ** 2)),
        )
        if linear_readout:
            self.net = nn.Linear(num_basis + edge_feature_dim, num_heads, bias=False)
        else:
            norm_cls = nn.RMSNorm if use_norms else nn.Identity
            self.net = nn.Sequential(
                nn.Linear(num_basis + edge_feature_dim, hidden_dim, bias=True),
                norm_cls(hidden_dim) if use_norms else norm_cls(),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim, bias=False),
                norm_cls(hidden_dim) if use_norms else norm_cls(),
                nn.SiLU(),
                nn.Linear(hidden_dim, num_heads, bias=False),
            )

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


class BatchedRadialNetwork(nn.Module):
    def __init__(
        self,
        n_copies:      int,
        num_basis:     int = 8,
        hidden_dim:    int = 32,
        cutoff_radius: float = 5.0,
        num_heads:     int = 1,
        envelope_floor: float = 0.02,
        radial_net_cls: type = None,
        use_norms: bool = True,
    ):
        super().__init__()
        radial_net_cls = RadialNetwork if radial_net_cls is None else radial_net_cls
        self.n_copies = n_copies
        self.cutoff_radius = cutoff_radius
        self.envelope_floor = envelope_floor
        self.use_norms = use_norms

        centres = torch.linspace(0.0, cutoff_radius, num_basis)
        self.register_buffer("centres", centres)
        width = cutoff_radius / max(num_basis - 1, 1)
        self.register_buffer("inv_width_sq", torch.tensor(-1.0 / (2.0 * width ** 2)))

        templates = [
            radial_net_cls(num_basis, hidden_dim, cutoff_radius, num_heads, edge_feature_dim=0,
                            envelope_floor=envelope_floor)
            for _ in range(n_copies)
        ]
        self.linear_readout = isinstance(templates[0].net, nn.Linear)
        if self.linear_readout:
            self.W1 = nn.Parameter(torch.stack([t.net.weight.detach().clone() for t in templates]))
        else:
            self.W1 = nn.Parameter(torch.stack([t.net[0].weight.detach().clone() for t in templates]))
            self.b1 = nn.Parameter(torch.stack([t.net[0].bias.detach().clone() for t in templates]))
            self.g1 = nn.Parameter(torch.stack([t.net[1].weight.detach().clone() for t in templates]))
            self.W2 = nn.Parameter(torch.stack([t.net[3].weight.detach().clone() for t in templates]))
            self.g2 = nn.Parameter(torch.stack([t.net[4].weight.detach().clone() for t in templates]))
            self.W3 = nn.Parameter(torch.stack([t.net[6].weight.detach().clone() for t in templates]))

    def _basis(self, r: torch.Tensor) -> torch.Tensor:
        diff = r - self.centres.unsqueeze(0)
        rbf  = torch.exp(self.inv_width_sq * diff * diff)
        r_scaled = (r / self.cutoff_radius).clamp(max=1.0)
        envelope = (1.0 - self.envelope_floor) * 0.5 * (torch.cos(math.pi * r_scaled) + 1.0) + self.envelope_floor
        return rbf * envelope

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        basis = self._basis(r)
        if self.linear_readout:
            return torch.einsum("nb,jhb->jnh", basis, self.W1)
        h1 = torch.einsum("nb,jhb->jnh", basis, self.W1) + self.b1.unsqueeze(1)
        if self.use_norms:
            h1 = F.rms_norm(h1, (h1.shape[-1],), None, None) * self.g1.unsqueeze(1)
        h1 = F.silu(h1)
        h2 = torch.einsum("jnh,jgh->jng", h1, self.W2)
        if self.use_norms:
            h2 = F.rms_norm(h2, (h2.shape[-1],), None, None) * self.g2.unsqueeze(1)
        h2 = F.silu(h2)
        return torch.einsum("jnh,joh->jno", h2, self.W3)


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
    cg = clebsch_gordan_matrix(l, k, dtype=dtype, device=device)
    Js = list(range(abs(l - k), l + k + 1))

    if isinstance(radial_net, BatchedRadialNetwork):
        phi_all = radial_net(r)
        for idx, J in enumerate(Js):
            Q_J = cg[J].to(device=device, dtype=dtype)
            Y_J = get_spherical_harmonics(J, x)
            QTY = torch.einsum("ji,ni->nj", Q_J, Y_J)
            had = phi_all[idx] * QTY
            W   = W + had.view(N, 2 * l + 1, 2 * k + 1)
        return W

    for J in Js:
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
    k_hi = max_degree if k_max is None else k_max

    if str(0) in radial_nets and isinstance(radial_nets[str(0)], BatchedRadialNetwork):
        device, dtype = x.device, x.dtype
        N = x.shape[0]
        r = x.norm(dim=-1, keepdim=True)
        f_concat = torch.cat([f[k] for k in range(k_hi + 1)], dim=-1)

        Y_cache: Dict[int, torch.Tensor] = {}
        def get_Y(J: int) -> torch.Tensor:
            if J not in Y_cache:
                Y_cache[J] = get_spherical_harmonics(J, x)
            return Y_cache[J]

        row_blocks = []
        for l in range(max_degree + 1):
            phi_all = radial_nets[str(l)](r) if N > 0 else None
            blocks = []
            idx = 0
            for k in range(k_hi + 1):
                cg = clebsch_gordan_matrix(l, k, dtype=dtype, device=device)
                W_lk = torch.zeros(N, 2 * l + 1, 2 * k + 1, device=device, dtype=dtype)
                for J in range(abs(l - k), l + k + 1):
                    if N > 0:
                        phi = phi_all[idx]
                        Q_J = cg[J]
                        Y_J = get_Y(J)
                        QTY = torch.einsum("ji,ni->nj", Q_J, Y_J)
                        W_lk = W_lk + (phi * QTY).view(N, 2 * l + 1, 2 * k + 1)
                    idx += 1
                blocks.append(W_lk)
            row_blocks.append(torch.cat(blocks, dim=-1))

        W_full = torch.cat(row_blocks, dim=-2) 
        out_concat = torch.einsum("nij,ncj->nci", W_full, f_concat)  

        out: Dict[int, torch.Tensor] = {}
        row_start = 0
        for l in range(max_degree + 1):
            row_end = row_start + (2 * l + 1)
            out[l] = out_concat[..., row_start:row_end]
            row_start = row_end
        return out

    out: Dict[int, torch.Tensor] = {}

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
            nn.RMSNorm(hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.RMSNorm(hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.RMSNorm(hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, out_feat_dim)
        )
        init_sequential_(self.out, "selu")

    def forward(self, f0, f1, f2, x_rel=None):
        f0_inv = self.f0_norm(f0.squeeze(-1)).unsqueeze(-1)
        f1_inv = self.f1_norm(f1).unsqueeze(-1).unsqueeze(-1)
        f2_inv = self.f2_norm(f2).unsqueeze(-1).unsqueeze(-1)

        f0_proj = self.f0_proj(f0_inv).squeeze(-1)
        f1_proj = self.f1_proj(f1_inv).squeeze(-1)
        f2_proj = self.f2_proj(f2_inv).squeeze(-1)

        f_out = self.out(torch.cat([f0_proj, f1_proj, f2_proj], dim=-1)).unsqueeze(-1)

        return f_out

class DeterministicMeanAggregate(nn.Module):
    def forward(self, f0, f1, f2, x_rel=None):
        f0_inv = f0.squeeze(-1)
        f1_inv = f1.norm(dim=-1)
        f2_inv = f2.norm(dim=-1)
        return ((f0_inv + f1_inv + f2_inv) / 3.0).unsqueeze(-1)

class DirectionAwareMeanAggregate(nn.Module):
    def forward(self, f0, f1, f2, x_rel=None):
        f0_inv = f0.squeeze(-1)
        f1_inv = f1.norm(dim=-1)
        f2_inv = f2.norm(dim=-1)
        if x_rel is None:
            return ((f0_inv + f1_inv + f2_inv) / 3.0).unsqueeze(-1)

        has_dir = (x_rel.norm(dim=-1, keepdim=True) > 1e-7).to(x_rel.dtype)
        Y1 = get_spherical_harmonics(1, x_rel) * has_dir
        Y2 = get_spherical_harmonics(2, x_rel) * has_dir
        f1_dot = torch.einsum("...ci,...i->...c", f1, Y1)
        f2_dot = torch.einsum("...cj,...j->...c", f2, Y2)
        return ((f0_inv + f1_inv + f1_dot + f2_inv + f2_dot) / 5.0).unsqueeze(-1)

class AttentionPoolExpand(nn.Module):
    def __init__(self, feature_dim: int, n_master_nodes: int):
        super().__init__()
        self.n_master_nodes = n_master_nodes
        self.queries = nn.Parameter(torch.randn(n_master_nodes, feature_dim) * (feature_dim ** -0.5))
        self.scale = feature_dim ** 0.5

    def forward(self, agg_per_atom: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        B = agg_per_atom.shape[0]
        Q = self.queries.unsqueeze(0).expand(B, -1, -1)
        scores = torch.einsum("bac,bnc->ban", Q, agg_per_atom) / self.scale
        scores = scores.masked_fill(~node_mask.unsqueeze(1), float("-inf"))
        alpha = torch.softmax(scores, dim=-1)
        return torch.einsum("ban,bnc->bac", alpha, agg_per_atom)

class ContentWeightedCOM(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.RMSNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        init_sequential_(self.score, "silu")

    def forward(
        self, f0_raw: torch.Tensor, x: torch.Tensor, node_mask: torch.Tensor,
        exclude_idx: torch.Tensor = None,
    ) -> torch.Tensor:
        logits = self.score(f0_raw).squeeze(-1)
        logits = logits.masked_fill(~node_mask, float("-inf"))
        if exclude_idx is not None:
            logits = logits.scatter(1, exclude_idx.view(-1, 1), float("-inf"))
        weights = torch.softmax(logits, dim=-1)
        return torch.einsum("bn,bni->bi", weights, x)


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


class BatchedGatedNonlinearity(nn.Module):
    def __init__(self, num_channels: int, degrees: list, floor: float = 0.5):
        super().__init__()
        self.degrees = list(degrees)
        self.floor = floor
        templates = [GatedNonlinearity(num_channels, floor=floor) for _ in self.degrees]
        self.W_gate = nn.Parameter(torch.stack([t.gate_lin.weight.detach().clone() for t in templates]))
        self.b_gate = nn.Parameter(torch.stack([t.gate_lin.bias.detach().clone() for t in templates]))

    def forward(self, x: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        norms = [straight_through_norm(x[d], dim=-1, floor=self.floor) for d in self.degrees]
        norm_stack = torch.stack(norms, dim=0)
        lead_shape = norm_stack.shape[1:-1]
        flat = norm_stack.reshape(len(self.degrees), -1, norm_stack.shape[-1])
        gate_flat = torch.einsum("doc,dnc->dno", self.W_gate, flat) + self.b_gate.unsqueeze(1)
        gate = torch.tanh(gate_flat).reshape(len(self.degrees), *lead_shape, -1)
        return {d: x[d] * gate[i].unsqueeze(-1) for i, d in enumerate(self.degrees)}

class EquivariantNormReadout(nn.Module):
    def __init__(self, C: int, hidden: int = 64, floor: float = 0.5, out_dim: int | None = None):
        super().__init__()
        self.floor = floor
        self.equiv_proj = nn.Linear(C, hidden, bias=False)
        init_linear_(self.equiv_proj, "linear")
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden, bias=False),
            nn.SiLU(),
            nn.Linear(hidden, 1 if out_dim is None else out_dim, bias=False),
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


class BatchedEquivariantRMSNorm(nn.Module):
    def __init__(self, num_channels: int, degrees: list, eps: float = 1e-8, floor: float = 0.5):
        super().__init__()
        self.degrees = list(degrees)
        self.eps = eps
        self.floor = floor
        templates = [EquivariantRMSNorm(num_channels, degree=d, eps=eps, floor=floor) for d in self.degrees]
        self.gamma = nn.Parameter(torch.stack([t.gamma.detach().clone() for t in templates]))

    def forward(self, x: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        norms, dirs = [], []
        for d in self.degrees:
            floor_d = 0.0 if d == 0 else self.floor
            norm_d, dir_d = straight_through_norm_and_direction(x[d], dim=-1, floor=floor_d)
            norms.append(norm_d)
            dirs.append(dir_d)

        norm_stack = torch.stack(norms, dim=0)
        rms = norm_stack.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        gamma_view = self.gamma.view(len(self.degrees), *([1] * (norm_stack.dim() - 2)), -1)
        target = (norm_stack / rms) * gamma_view

        return {d: dirs[i] * target[i].unsqueeze(-1) for i, d in enumerate(self.degrees)}


class BatchedHighOrdFFN(nn.Module):
    def __init__(
        self, feature_dim: int, hidden_dim: int, degree: int, n_copies: int,
        floor: float = 0.5, eps: float = 1e-8, in_dim: int | None = None, use_norms: bool = True,
    ):
        super().__init__()
        self.degree = degree
        self.floor = floor
        self.eps = eps
        self.n_copies = n_copies
        self.use_norms = use_norms

        templates = [high_ord_ffn(feature_dim, hidden_dim, degree, in_dim=in_dim) for _ in range(n_copies)]

        def stack_rms(idx):
            return nn.Parameter(torch.stack([t[idx].gamma.detach().clone() for t in templates]))

        def stack_lin(idx):
            return nn.Parameter(torch.stack([t[idx].weight.weight.detach().clone() for t in templates]))

        def stack_gate(idx):
            w = nn.Parameter(torch.stack([t[idx].gate_lin.weight.detach().clone() for t in templates]))
            b = nn.Parameter(torch.stack([t[idx].gate_lin.bias.detach().clone() for t in templates]))
            return w, b

        self.gamma0 = stack_rms(0)
        self.W1 = stack_lin(1)
        self.gamma1 = stack_rms(2)
        self.gate_w1, self.gate_b1 = stack_gate(3)
        self.W2 = stack_lin(4)
        self.gamma2 = stack_rms(5)
        self.gate_w2, self.gate_b2 = stack_gate(6)
        self.W3 = stack_lin(7)
        self.gamma3 = stack_rms(8)
        self.gate_w3, self.gate_b3 = stack_gate(9)
        self.W4 = stack_lin(10)

    def _rms(self, x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        if not self.use_norms:
            return x
        floor = 0.0 if self.degree == 0 else self.floor
        norm, x_dir = straight_through_norm_and_direction(x, dim=-1, floor=floor)
        rms = norm.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        target = (norm / rms) * gamma.view(1, self.n_copies, 1, -1)
        return x_dir * target.unsqueeze(-1)

    def _lin(self, x: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        return torch.einsum("doi,bdnim->bdnom", W, x)

    def _gate(self, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        norm = straight_through_norm(x, dim=-1, floor=self.floor)
        gate_pre = torch.einsum("doi,bdni->bdno", w, norm) + b.view(1, self.n_copies, 1, -1)
        gate = torch.tanh(gate_pre)
        return x * gate.unsqueeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_copies, M, feature_dim, 2*degree+1)."""
        x = self._rms(x, self.gamma0)
        x = self._lin(x, self.W1)
        x = self._rms(x, self.gamma1)
        x = self._gate(x, self.gate_w1, self.gate_b1)
        x = self._lin(x, self.W2)
        x = self._rms(x, self.gamma2)
        x = self._gate(x, self.gate_w2, self.gate_b2)
        x = self._lin(x, self.W3)
        x = self._rms(x, self.gamma3)
        x = self._gate(x, self.gate_w3, self.gate_b3)
        x = self._lin(x, self.W4)
        return x


def cg_for_J(l: int, k: int, J: int) -> torch.Tensor:
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

    cg    = clebsch_gordan_matrix(l, k, dtype=dtype)
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