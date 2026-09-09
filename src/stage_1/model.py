import os
import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import Dict, Optional, Tuple

from stage_1.attention import SRAttention, MasterNodeAttention, MasterFusionAttention, SRCrossAttention
from utils.se3_utils import *
from utils.pw_bases import PWProjection, minimum_image, make_wavevectors
from utils.calc_props import _MOL_FEAT_DIM

if os.path.exists("sh1_cartesian.pt"):
    SH1_TO_CARTESIAN = torch.load("sh1_cartesian.pt")
else:
    SH1_TO_CARTESIAN = fit_sh_to_cartesian(1, cartesian_dipole)
    torch.save(SH1_TO_CARTESIAN, "sh1_cartesian.pt")

if os.path.exists("sh2_cartesian.pt"):
    SH2_TO_CARTESIAN = torch.load("sh2_cartesian.pt")
else:
    SH2_TO_CARTESIAN = fit_sh_to_cartesian(2, cartesian_quadrupole)
    torch.save(SH2_TO_CARTESIAN, "sh2_cartesian.pt")

def masked_weighted_com(
    x:    torch.Tensor,   
    mass: torch.Tensor,  
    mask: torch.Tensor,  
) -> Tuple[torch.Tensor, torch.Tensor]:
    w = mass * mask.to(mass.dtype)
    has_atoms = mask.any(dim=1)
    w_sum = w.sum(dim=1, keepdim=True).clamp(min=1e-8)
    com = (w.unsqueeze(-1) * x).sum(dim=1) / w_sum
    return com, has_atoms

def masked_mean(feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    extra_dims = feat.dim() - mask.dim()
    m = mask.to(feat.dtype)
    for _ in range(extra_dims):
        m = m.unsqueeze(-1)
    count = mask.sum(dim=1).clamp(min=1).to(feat.dtype)
    for _ in range(extra_dims):
        count = count.unsqueeze(-1)
    return (feat * m).sum(dim=1) / count

def safe_minimum_image_cutoff(V: torch.Tensor, cutoff: float) -> torch.Tensor:
    a1, a2, a3 = V[..., 0], V[..., 1], V[..., 2]
    volume = torch.linalg.det(V).abs()
    d1 = volume / torch.cross(a2, a3, dim=-1).norm(dim=-1).clamp(min=1e-8)
    d2 = volume / torch.cross(a1, a3, dim=-1).norm(dim=-1).clamp(min=1e-8)
    d3 = volume / torch.cross(a1, a2, dim=-1).norm(dim=-1).clamp(min=1e-8)
    min_width = torch.minimum(torch.minimum(d1, d2), d3)
    return torch.minimum(torch.full_like(min_width, cutoff), 0.5 * min_width)

def seed_equivariant_features(
    combined_scalar: torch.Tensor,
    radial_net:      nn.Module,
    x:             torch.Tensor,
    atomic_masses: torch.Tensor,
    node_mask:     torch.Tensor,
    V:             torch.Tensor,
    diag_sink:     Optional[dict] = None,
    x_wrapped:     Optional[torch.Tensor] = None,
) -> Dict[int, torch.Tensor]:
    if x_wrapped is None:
        x_wrapped = minimum_image(x, V)
    center, _ = masked_weighted_com(x_wrapped, atomic_masses, node_mask)

    x_rel   = x_wrapped - center.unsqueeze(1)
    r_true  = x_rel.norm(dim=-1)
    r       = r_true.clamp(min=1e-8)
    has_dir = (r_true > 1e-7).unsqueeze(-1)

    Y1 = get_spherical_harmonics(1, x_rel) * has_dir
    Y2 = get_spherical_harmonics(2, x_rel) * has_dir

    B, N = r.shape
    radial = radial_net(r.reshape(-1, 1)).view(B, N, 1)
    if diag_sink is not None:
        masked_radial = radial.masked_fill(~node_mask.unsqueeze(-1), float("inf"))
        diag_sink["min_abs_radial"] = float(masked_radial.detach().abs().min())
    scalar = combined_scalar.unsqueeze(-1)

    f1 = scalar * (radial.unsqueeze(-1) * Y1.unsqueeze(2))
    f2 = scalar * (radial.unsqueeze(-1) * Y2.unsqueeze(2))

    return {1: f1, 2: f2}

class Readout(nn.Module):
    def __init__(self, feature_dim, hidden_dim=128):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        self.ffn0 = scalar_ffn(feature_dim, hidden_dim)
        self.ffn1 = high_ord_ffn(feature_dim, hidden_dim, 1)
        self.ffn2 = high_ord_ffn(feature_dim, hidden_dim, 2)

        self.ffn = [self.ffn0, self.ffn1, self.ffn2]

    def forward(self, f):
        for l in f.keys():
            if l == 0:
                f[l] = self.ffn[l](f[l].squeeze(-1)).unsqueeze(-1)
            else:
                f[l] = self.ffn[l](f[l])

        return f

class EnergyLayer(nn.Module):
    def __init__(
        self, radial_net, max_degree: int, feature_dim: int, hidden_dim: int = 128,
        edge_cutoff: float = 5.0, n_max: int = 2, pw_e_cutoff: Optional[float] = None,
        k_max: int = 0, edge_feat_dim: int = 8, ablate_max_degree: Optional[int] = None,
        ablate_lr: bool = False, normalize_z_by_n_real: bool = False,
        decouple_ri_norm: bool = False,
        imaginary_only_ri: bool = False,
    ):
        super().__init__()
        self.radial_net = radial_net
        self.max_degree  = max_degree
        self.feature_dim = feature_dim
        self.edge_cutoff = edge_cutoff
        self.k_max = k_max
        self.ablate_max_degree = ablate_max_degree
        self.ablate_lr = ablate_lr
        self.normalize_z_by_n_real = normalize_z_by_n_real
        self.decouple_ri_norm = decouple_ri_norm
        self.imaginary_only_ri = imaginary_only_ri
        self.sr_attn = SRAttention(
            radial_net, max_degree, feature_dim, hidden_dim, k_max=k_max, edge_feat_dim=edge_feat_dim,
        )

        self.W_V_self_linear = nn.ModuleDict({
            str(l): EquivariantLinear(feature_dim, feature_dim)
            for l in range(max_degree + 1)
        })
        self.W_V_self_norm = BatchedEquivariantRMSNorm(feature_dim, degrees=list(range(max_degree + 1)))
        self.sigma_gate = BatchedGatedNonlinearity(feature_dim, degrees=list(range(max_degree + 1)))

        self.W_master_self = IrrepLinear(feature_dim)
        self.master_attn = MasterNodeAttention(feature_dim, n_max, pw_e_cutoff, hidden_dim)
        self.master_ln = EquivariantRMSNorm(feature_dim, degree=0)

        self.layer_norm_in = BatchedEquivariantRMSNorm(feature_dim, degrees=list(range(max_degree + 1)))
        self.layer_norm_mid = BatchedEquivariantRMSNorm(feature_dim, degrees=list(range(max_degree + 1)))
        self.layer_norm_out = BatchedEquivariantRMSNorm(feature_dim, degrees=list(range(max_degree + 1)))

        self.readout = Readout(feature_dim, hidden_dim)

        self.diag_enabled = False
        self._diag_g_a_norm: Optional[float] = None

    def forward(
        self,
        f_in:        Dict[int, torch.Tensor],
        f0:          Dict[int, torch.Tensor],
        g_a_in:      torch.Tensor,
        x_wrapped:   torch.Tensor,
        x_com:       torch.Tensor,
        node_mask:   torch.Tensor,
        edge_b:      torch.Tensor,
        edge_i:      torch.Tensor,
        edge_j:      torch.Tensor,
        x_rel_e:     torch.Tensor,
        edge_feat_e: torch.Tensor,
        kvecs:       torch.Tensor,
        mode_mask:   torch.Tensor,
        return_scale_feat: bool = False,
    ) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
        B, N, _ = x_wrapped.shape

        f_in = self.layer_norm_in(f_in)

        f_in_i_e = {k: f_in[k][edge_b, edge_i] for k in range(self.k_max + 1)}
        f0_j_e   = {k: f0[k][edge_b, edge_j]   for k in range(self.k_max + 1)}

        alpha_e, Wv_j_sr_flat = self.sr_attn(
            f_in_i_e, f0_j_e, x_rel_e, edge_feat_e, edge_b, edge_i, batch_size=B, num_atoms=N,
        )

        receiver_flat = edge_b * N + edge_i

        # Final local features
        self_terms_raw = {l: self.W_V_self_linear[str(l)](f_in[l]) for l in range(self.max_degree + 1)}
        self_terms = self.W_V_self_norm(self_terms_raw)

        neighbor_raws: Dict[int, torch.Tensor] = {}
        for l in range(self.max_degree + 1):
            C = f_in[l].shape[2]
            contrib_e    = alpha_e.view(-1, 1, 1) * Wv_j_sr_flat[l]
            neighbor_raws[l] = torch.zeros(
                B * N, C, 2 * l + 1, device=x_wrapped.device, dtype=x_wrapped.dtype,
            ).scatter_add(
                0, receiver_flat.view(-1, 1, 1).expand(-1, C, 2 * l + 1), contrib_e,
            ).view(B, N, C, 2 * l + 1)
        neighbor_terms = self.sigma_gate(neighbor_raws)

        pre_mid = {l: f_in[l] + self_terms[l] + neighbor_terms[l] for l in range(self.max_degree + 1)}
        f_out = self.layer_norm_mid(pre_mid)
        for l in range(self.max_degree + 1):
            if self.ablate_max_degree is not None and l > self.ablate_max_degree:
                f_out[l] = torch.zeros_like(f_out[l])

        f = self.readout(f_out)
        pre_out = {l: f[l] + f_out[l] for l in range(self.max_degree + 1)}
        f = self.layer_norm_out(pre_out)

        master_attn_result = self.master_attn(
            g_a_in, f_in[0], x_wrapped, x_com, node_mask,
            normalize_z_by_n_real=self.normalize_z_by_n_real,
            return_scale_feat=return_scale_feat,
            decouple_ri_norm=self.decouple_ri_norm,
            imaginary_only_ri=self.imaginary_only_ri,
            kvecs=kvecs, mode_mask=mode_mask,
        )
        scale_feat = None
        if return_scale_feat:
            master_attn_out, scale_feat = master_attn_result
        else:
            master_attn_out = master_attn_result
        if self.ablate_lr:
            master_attn_out = torch.zeros_like(master_attn_out)
        g_a_out = self.master_ln(self.W_master_self(g_a_in) + master_attn_out)
        if self.diag_enabled:
            self._diag_g_a_norm = float(g_a_out.detach().norm())

        if return_scale_feat:
            return f, g_a_out, scale_feat
        return f, g_a_out

def _run_energy_layer(
    layer, f_in, f0, g_a_in, x_wrapped, x_com, node_mask,
    edge_b, edge_i, edge_j, x_rel_e, edge_feat_e, kvecs, mode_mask,
):
    return layer(
        f_in=f_in, f0=f0, g_a_in=g_a_in, x_wrapped=x_wrapped, x_com=x_com, node_mask=node_mask,
        edge_b=edge_b, edge_i=edge_i, edge_j=edge_j, x_rel_e=x_rel_e, edge_feat_e=edge_feat_e,
        kvecs=kvecs, mode_mask=mode_mask,
    )

def _compute_k_max(m: int, max_degree: int) -> int:
    return m if m <= max_degree else m % max_degree

class EnergyModel(nn.Module):
    def __init__(
        self,
        radial_net,
        num_atom_types: int,
        max_degree:       int = 2,
        num_layers:       int = 4,
        feature_dim:      int = 32,
        hidden_dim:       int = 128,
        task:             int = 0,
        edge_cutoff:      float = 5.0,
        n_max:            int = 2,
        pw_e_cutoff:      Optional[float] = None,
        n_master_nodes:   int = 4,
        num_fusion_heads: int = 4,
        edge_feat_dim:    int = 8,
        ablate_max_degree: Optional[int] = None,
        ablate_lr: bool = False,
        grad_checkpoint: bool = False,
        normalize_z_by_n_real: bool = False,
        decouple_ri_norm: bool = False,
        imaginary_only_ri: bool = False,
        deterministic_pool_agg: bool = False,
        attention_pool: bool = False,
        skip_master_expand_norm: bool = False,
        static_box: bool = False,
        content_weighted_com: bool = False,
    ):
        super().__init__()
        self.radial_net = radial_net
        self.edge_cutoff = edge_cutoff
        self.n_max = n_max
        self.pw_e_cutoff = pw_e_cutoff
        self.static_box = static_box
        self._cached_effective_cutoff: Optional[torch.Tensor] = None
        self._cached_kvecs: Optional[torch.Tensor] = None
        self._cached_mode_mask: Optional[torch.Tensor] = None
        self.content_weighted_com = content_weighted_com
        self.com_weighting = ContentWeightedCOM(feature_dim, hidden_dim) if content_weighted_com else None

        self.max_degree = max_degree
        self.task       = task
        self.n_master_nodes = n_master_nodes
        self.ablate_max_degree = ablate_max_degree
        self.ablate_lr = ablate_lr
        self.grad_checkpoint = grad_checkpoint
        self.normalize_z_by_n_real = normalize_z_by_n_real
        self.decouple_ri_norm = decouple_ri_norm
        self.imaginary_only_ri = imaginary_only_ri
        self.attention_pool = attention_pool

        self.input_embedding = nn.Embedding(num_atom_types, feature_dim)
        self.ln1             = nn.RMSNorm(feature_dim)
        self.mol_prop_emb    = nn.Sequential(
                            scalar_ffn(_MOL_FEAT_DIM, hidden_dim),
                            nn.RMSNorm(_MOL_FEAT_DIM),
                            nn.LeakyReLU(),
                            nn.Linear(_MOL_FEAT_DIM, feature_dim)
                        )
        init_sequential_(self.mol_prop_emb, "leaky_relu")
        self.ln2             = nn.RMSNorm(feature_dim)
        self.f0_ffn          = scalar_ffn(feature_dim, hidden_dim)
        self.mol_ffn         = scalar_ffn(feature_dim, hidden_dim)
        
        self.seed_proj       = scalar_ffn(2 * feature_dim, hidden_dim)
        self.seed_proj_out   = nn.Linear(2 * feature_dim, feature_dim)
        init_linear_(self.seed_proj_out, "linear")
        self.seed_proj_norm  = nn.RMSNorm(feature_dim)
       
        self.seed_radial_net = radial_net(num_basis=1, hidden_dim=hidden_dim)

        self.master_pool_agg = (
            DeterministicMeanAggregate() if deterministic_pool_agg else
            AggregateFeatures(out_feat_dim=feature_dim, inp_feat_dim=feature_dim, hidden_dim=hidden_dim)
        )
        if attention_pool:
            self.pool_expand = AttentionPoolExpand(feature_dim, n_master_nodes)
        else:
            self.master_pool_ffn = scalar_ffn(feature_dim, hidden_dim)
            self.master_expand   = nn.Sequential(
                nn.RMSNorm(feature_dim),
                nn.LeakyReLU(),
                nn.Linear(feature_dim, n_master_nodes * feature_dim)
            )
            init_sequential_(self.master_expand, "leaky_relu")
        self.master_expand_norm = nn.Identity() if skip_master_expand_norm else EquivariantRMSNorm(feature_dim, degree=0)

        self.layers = nn.ModuleList([
            EnergyLayer(
                radial_net, max_degree, feature_dim, hidden_dim, edge_cutoff, n_max, pw_e_cutoff,
                k_max=_compute_k_max(m, max_degree), edge_feat_dim=edge_feat_dim,
                ablate_max_degree=ablate_max_degree, ablate_lr=ablate_lr,
                normalize_z_by_n_real=normalize_z_by_n_real,
                decouple_ri_norm=decouple_ri_norm,
                imaginary_only_ri=imaginary_only_ri,
            )
            for m in range(num_layers)
        ])

        self.master_fusion = MasterFusionAttention(
            radial_net, max_degree, feature_dim, hidden_dim, num_heads=num_fusion_heads,
        )
        self.layer_norm_fusion = nn.ModuleDict({
            str(l): EquivariantRMSNorm(feature_dim, degree=l)
            for l in range(max_degree + 1)
        })

        self.readout = Readout(feature_dim, hidden_dim)

        self.diag_enabled = False

    def _box_derived_quantities(self, V: torch.Tensor, B: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.static_box:
            effective_cutoff = safe_minimum_image_cutoff(V, self.edge_cutoff).view(B, 1, 1)
            kvecs, _, mode_mask = make_wavevectors(V, self.n_max, self.pw_e_cutoff)
            return effective_cutoff, kvecs, mode_mask

        if self._cached_kvecs is None:
            if not torch.equal(V, V[:1].expand_as(V)):
                raise ValueError
            effective_cutoff_full = safe_minimum_image_cutoff(V[:1], self.edge_cutoff).view(1, 1, 1)
            kvecs_full, _, mode_mask_full = make_wavevectors(V[:1], self.n_max, self.pw_e_cutoff)
            self._cached_effective_cutoff = effective_cutoff_full.detach()
            self._cached_kvecs = kvecs_full.detach()
            self._cached_mode_mask = mode_mask_full.detach()

        effective_cutoff = self._cached_effective_cutoff.expand(B, 1, 1)
        kvecs = self._cached_kvecs.expand(B, -1, -1)
        mode_mask = self._cached_mode_mask.expand(B, -1)
        return effective_cutoff, kvecs, mode_mask
        self._diag_stats: Dict[str, float] = {}

    def forward(
        self,
        x, node_types, atomic_masses, node_mask,
        V, mol_props, edge_feats,
        return_layer_states: bool = False,
        seed_masses=None,
        collect_scale_feats: bool = False,
        target_idx: torch.Tensor = None,
    ) -> torch.Tensor:
        node_mask = node_mask.bool()
        atomic_masses = atomic_masses * node_mask.to(atomic_masses.dtype)
        seed_masses = atomic_masses if seed_masses is None else seed_masses * node_mask.to(seed_masses.dtype)
        B, N = node_types.shape

        f0_raw = self.input_embedding(node_types)
        f0_ffn_out = self.f0_ffn(self.ln1(f0_raw))

        mol_ctx = self.mol_prop_emb(mol_props).unsqueeze(1).expand(-1, N, -1)
        mol_ffn_out = self.mol_ffn(self.ln2(mol_ctx))

        combined_scalar = (self.seed_proj(torch.cat([f0_ffn_out, mol_ffn_out], dim=-1)))
        combined_scalar = self.seed_proj_norm(self.seed_proj_out(combined_scalar))

        x_wrapped = minimum_image(x, V)

        seeded = seed_equivariant_features(
            combined_scalar, self.seed_radial_net, x, seed_masses, node_mask, V,
            diag_sink=(self._diag_stats if self.diag_enabled else None),
            x_wrapped=x_wrapped,
        )
        if self.ablate_max_degree is not None:
            seeded = {
                l: (v if l <= self.ablate_max_degree else torch.zeros_like(v))
                for l, v in seeded.items()
            }
        f0: Dict[int, torch.Tensor] = {0: f0_raw.unsqueeze(-1), 1: seeded[1], 2: seeded[2]}
        f: Dict[int, torch.Tensor] = dict(f0)

        if self.content_weighted_com:
            x_com = self.com_weighting(f0_raw, x_wrapped, node_mask, exclude_idx=target_idx)
        else:
            x_com, _ = masked_weighted_com(x_wrapped, atomic_masses, node_mask)

        effective_cutoff, kvecs, mode_mask = self._box_derived_quantities(V, B)

        x_i = x_wrapped.unsqueeze(2).expand(B, N, N, 3)
        x_j = x_wrapped.unsqueeze(1).expand(B, N, N, 3)
        dist = (x_j - x_i).norm(dim=-1)
        self_mask = ~torch.eye(N, dtype=torch.bool, device=x.device).unsqueeze(0)
        pair_mask = node_mask.unsqueeze(2) & node_mask.unsqueeze(1) & (dist <= effective_cutoff) & self_mask
        edge_b, edge_i, edge_j = pair_mask.nonzero(as_tuple=True)
        x_rel_e = x_wrapped[edge_b, edge_j] - x_wrapped[edge_b, edge_i]
        edge_feat_e = edge_feats[edge_b, edge_i, edge_j]

        agg_per_atom = self.master_pool_agg(f0[0], f0[1], f0[2]).squeeze(-1)
        if self.attention_pool:
            g_a = self.pool_expand(agg_per_atom, node_mask).unsqueeze(-1)
        else:
            pooled = self.master_pool_ffn(masked_mean(agg_per_atom, node_mask))
            g_a = self.master_expand(pooled).view(B, self.n_master_nodes, -1).unsqueeze(-1)
        g_a = self.master_expand_norm(g_a)

        layer_states = [] if return_layer_states else None
        scale_feats = [] if collect_scale_feats else None
        for layer in self.layers:
            if self.grad_checkpoint and self.training:
                f, g_a = checkpoint(
                    _run_energy_layer, layer, f, f0, g_a, x_wrapped, x_com, node_mask,
                    edge_b, edge_i, edge_j, x_rel_e, edge_feat_e, kvecs, mode_mask,
                    use_reentrant=False,
                )
            elif collect_scale_feats:
                f, g_a, scale_feat = layer(
                    f_in=f, f0=f0, g_a_in=g_a, x_wrapped=x_wrapped, x_com=x_com, node_mask=node_mask,
                    edge_b=edge_b, edge_i=edge_i, edge_j=edge_j, x_rel_e=x_rel_e, edge_feat_e=edge_feat_e,
                    kvecs=kvecs, mode_mask=mode_mask,
                    return_scale_feat=True,
                )
                scale_feats.append(scale_feat)
            else:
                f, g_a = layer(
                    f_in=f, f0=f0, g_a_in=g_a, x_wrapped=x_wrapped, x_com=x_com, node_mask=node_mask,
                    edge_b=edge_b, edge_i=edge_i, edge_j=edge_j, x_rel_e=x_rel_e, edge_feat_e=edge_feat_e,
                    kvecs=kvecs, mode_mask=mode_mask,
                )
            if return_layer_states:
                layer_states.append(f)

        fusion_out = self.master_fusion(f_atoms=f, g_a=g_a, x=x_wrapped, x_com=x_com)
        if self.ablate_max_degree is not None:
            fusion_out = {
                l: (v if l <= self.ablate_max_degree else torch.zeros_like(v))
                for l, v in fusion_out.items()
            }
        if self.ablate_lr:
            fusion_out = {l: torch.zeros_like(v) for l, v in fusion_out.items()}
        f = {l: self.layer_norm_fusion[str(l)](f[l] + fusion_out[l]) for l in f}

        f = self.readout(f)

        if return_layer_states and collect_scale_feats:
            return f, layer_states, scale_feats
        if return_layer_states:
            return f, layer_states
        if collect_scale_feats:
            return f, scale_feats
        return f

class SolventModel(nn.Module):
    def __init__(self, radial_net, feature_dim):
        super().__init__()

    def forward(self, 
        lw_x, lw_types, lw_masses, lw_mask,
        pw_x, pw_types, pw_masses, pw_mask, 
        wl_x, wl_types, wl_masses, wl_mask, 
        wp_x, wp_types, wp_masses, wp_mask,
        V,
    ):
        x_i = torch.cat([lw_x, pw_x], dim=1)
        node_types_i = torch.cat([lw_types, pw_types], dim=1)
        masses_i = torch.cat([lw_masses, pw_masses], dim=1)
        node_mask_i = torch.cat([lw_mask, pw_mask], dim=1).bool()

        x_j = torch.cat([wl_x, wp_x], dim=1)
        node_types_j = torch.cat([wl_types, wp_types], dim=1)
        masses_j = torch.cat([wl_masses, wp_masses], dim=1)
        node_mask_j = torch.cat([wl_mask, wp_mask], dim=1).bool()        

class SolventLayer(nn.Module):
    def __init__(
        self, radial_net, max_degree, feature_dim, hidden_dim: int = 64,
        edge_cutoff: float = 5.0, n_max: int = 2, pw_e_cutoff: Optional[float] = None,
    ):
        super().__init__()

        self.max_degree  = max_degree
        self.feature_dim = feature_dim
        self.edge_cutoff = edge_cutoff
        self.n_max       = n_max

        # Self interaction modules
        self.W_V_self_i = nn.ModuleDict({
            str(l): IrrepLinear(feature_dim)
            for l in range(max_degree + 1)
        })

        self.W_V_self_j = nn.ModuleDict({
            str(l): IrrepLinear(feature_dim)
            for l in range(max_degree + 1)
        })

        self.rnet = radial_net

        self.eqn1 = EquivariantNormReadout(feature_dim)
        self.eqn2 = EquivariantNormReadout(feature_dim)
        self.eqn3 = EquivariantNormReadout(feature_dim)
        self.eqn4 = EquivariantNormReadout(feature_dim)
        self.eqn5 = EquivariantNormReadout(feature_dim)
        self.eqn6 = EquivariantNormReadout(feature_dim)

        self.z_1  = EquivariantLinear(feature_dim, feature_dim)

        # Dipole-dipole interaction modules
        self.sr_attn = SRCrossAttention(radial_net, max_degree, feature_dim, hidden_dim)

        self.radial_V_sr = nn.ModuleDict()
        for l in range(max_degree + 1):
            for k in range(max_degree + 1):
                key = f"{l}_{k}"
                self.radial_V_sr[key] = nn.ModuleDict({
                    str(J): radial_net(num_basis=(2 * l + 1), hidden_dim=hidden_dim)
                    for J in range(abs(l - k), l + k + 1)
                })

        self.aggregate_features = AggregateFeatures(
            out_feat_dim=feature_dim, inp_feat_dim=feature_dim, hidden_dim=hidden_dim,
        )
        self.pw_projection = PWProjection(
            n_max=n_max, h_global_dim=feature_dim, e_cutoff=pw_e_cutoff,
        )

        self.sr_attn_w = SRCrossAttention(radial_net, max_degree, feature_dim, hidden_dim)

        self.radial_V_sr_w = nn.ModuleDict()
        for l in range(max_degree + 1):
            for k in range(max_degree + 1):
                key = f"{l}_{k}"
                self.radial_V_sr_w[key] = nn.ModuleDict({
                    str(J): radial_net(num_basis=(2 * l + 1), hidden_dim=hidden_dim)
                    for J in range(abs(l - k), l + k + 1)
                })

        self.pw_projection_w = PWProjection(
            n_max=n_max, h_global_dim=feature_dim, e_cutoff=pw_e_cutoff,
        )

        # Implicit interaction modules
        self.cav_rad_i = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.RMSNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.RMSNorm(2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, 2 * hidden_dim),
            nn.RMSNorm(2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, 1),
        )

        self.cav_rad_j = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.RMSNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.RMSNorm(2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, 2 * hidden_dim),
            nn.RMSNorm(2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, 1)
        )

        self.eq_lin0i = EquivariantLinear(feature_dim, 1)
        self.eq_lin0j = EquivariantLinear(feature_dim, 1)

        self.eq_lin1i = EquivariantLinear(feature_dim, 1)
        self.eq_lin1j = EquivariantLinear(feature_dim, 1)

    def _calc_cavity_radius(self, f_i0, f_j0):
        B = f_i0.shape[0]

        proj_i = self.cav_rad_i(f_i0.squeeze(-1)).mean(dim=1).squeeze(-1)
        proj_j = self.cav_rad_j(f_j0.squeeze(-1)).mean(dim=1).squeeze(-1)

        cavity = proj_i + proj_j
        scaled = (3 / (4 * torch.pi)) * cavity
        a = torch.sign(scaled) * scaled.abs().pow(1 / 3)

        return a

    def forward(
        self,
        f_in_i:      Dict[int, torch.Tensor],
        x_i:         torch.Tensor,
        node_mask_i: torch.Tensor,
        f_in_j:      torch.Tensor,
        x_j:         torch.Tensor,
        node_mask_j: torch.Tensor,
        V:         torch.Tensor,
    ) -> Dict[int, torch.Tensor]:
        B, N1, _ = x_i.shape 
        B, N2, _ = x_j.shape 

        # Dipole-dipole interactions
        x_i_wrapped = minimum_image(x_i, V)
        x_j_wrapped = minimum_image(x_j, V)

        z_2 = self.W_V_self_i['2'](f_in_i[2])

        sr_x_rel = x_j_wrapped.unsqueeze(1) - x_i_wrapped.unsqueeze(2)  
        sr_dist  = sr_x_rel.norm(dim=-1)

        effective_cutoff = safe_minimum_image_cutoff(V, self.edge_cutoff).view(B, 1, 1)
        pair_mask = node_mask_i.unsqueeze(2) & node_mask_j.unsqueeze(1) & (sr_dist <= effective_cutoff)

        alpha = self.sr_attn(f_in_i, f_in_j, sr_x_rel, pair_mask)   

        sr_x_rel_flat = sr_x_rel.reshape(B * N1 * N2, 3)
        f_j_flat = {
            l: f_in_j[l].unsqueeze(1).expand(B, N1, N2, f_in_j[l].shape[2], 2 * l + 1)
               .reshape(B * N1 * N2, f_in_j[l].shape[2], 2 * l + 1)
            for l in f_in_j
        }
        Wv_j_sr_flat = apply_direct_sum_W(
            f=f_j_flat, x=sr_x_rel_flat,
            radial_nets=self.radial_V_sr, max_degree=self.max_degree,
        )

        pair_mask_grid = pair_mask.unsqueeze(-1).unsqueeze(-1)

        agg_j     = self.aggregate_features(f_in_j[0], f_in_j[1], f_in_j[2]).squeeze(-1)  
        agg_j_exp = agg_j.unsqueeze(1).expand(B, N1, N2, self.feature_dim)
        h         = (agg_j_exp * pair_mask.unsqueeze(-1)).sum(dim=2)                       

        sr_x_rel_pw = sr_x_rel.view(B * N1, N2, 3)
        h_pw        = h.reshape(B * N1, self.feature_dim)
        V_pw        = V.unsqueeze(1).expand(B, N1, 3, 3).reshape(B * N1, 3, 3)

        pw_out    = self.pw_projection(sr_x_rel_pw, h_pw, V_pw).view(B, N1, N2, 2)
        pw_scalar = pw_out[..., 0]

        combined_weight     = alpha * pw_scalar
        combined_weight_exp = combined_weight.unsqueeze(-1).unsqueeze(-1)

        sr_term: Dict[int, torch.Tensor] = {}
        for l in range(self.max_degree + 1):
            C    = f_in_j[l].shape[2]
            self_term = self.W_V_self_i[str(l)](f_in_i[l])
            Wv_j = Wv_j_sr_flat[l].view(B, N1, N2, C, 2 * l + 1)
            Wv_j = torch.where(pair_mask_grid, Wv_j, torch.zeros_like(Wv_j))
            sr_term[l] = self_term + (combined_weight_exp * Wv_j).sum(dim=2)

        z_0 = sr_term[0]
        z_1 = sr_term[1]
        z_2 = sr_term[2]

        # Dipole-field interactions
        eps_j = self.W_V_self_i["0"](f_in_j[0]) * self.eqn2(f_in_j[1]).reshape(B, N2, 1, 1)
        eps_j = eps_j.mean(dim=1, keepdim=True)
        z_1_df = self.z_1(f_in_i[1] * eps_j)
        z_1 = z_1 + z_1_df

        # Dipole-dipole interactions solvent
        x_kv    = torch.cat([x_i_wrapped, x_j_wrapped], dim=1)          
        mask_kv = torch.cat([node_mask_i, node_mask_j], dim=1)        
        f_kv    = {l: torch.cat([f_in_i[l], f_in_j[l]], dim=1) for l in range(self.max_degree + 1)}

        w_x_rel = x_kv.unsqueeze(1) - x_j_wrapped.unsqueeze(2)       
        w_dist  = w_x_rel.norm(dim=-1)

        water_eye    = ~torch.eye(N2, dtype=torch.bool, device=x_j.device)
        kv_self_mask = torch.cat(
            [torch.ones(N2, N1, dtype=torch.bool, device=x_j.device), water_eye], dim=1
        ).unsqueeze(0)                                                  

        w_pair_mask = (
            node_mask_j.unsqueeze(2) & mask_kv.unsqueeze(1)
            & (w_dist <= effective_cutoff) & kv_self_mask
        )                                                              

        w_alpha = self.sr_attn_w(f_in_j, f_kv, w_x_rel, w_pair_mask)    

        Nkv = N1 + N2
        w_x_rel_flat = w_x_rel.reshape(B * N2 * Nkv, 3)
        f_kv_flat = {
            l: f_kv[l].unsqueeze(1).expand(B, N2, Nkv, f_kv[l].shape[2], 2 * l + 1)
               .reshape(B * N2 * Nkv, f_kv[l].shape[2], 2 * l + 1)
            for l in f_kv
        }
        Wv_kv_flat = apply_direct_sum_W(
            f=f_kv_flat, x=w_x_rel_flat, radial_nets=self.radial_V_sr_w, max_degree=self.max_degree,
        )

        w_pair_mask_grid = w_pair_mask.unsqueeze(-1).unsqueeze(-1)

        agg_kv     = self.aggregate_features(f_kv[0], f_kv[1], f_kv[2]).squeeze(-1)  
        agg_kv_exp = agg_kv.unsqueeze(1).expand(B, N2, Nkv, self.feature_dim)
        w_h        = (agg_kv_exp * w_pair_mask.unsqueeze(-1)).sum(dim=2)              

        w_x_rel_pw = w_x_rel.view(B * N2, Nkv, 3)
        w_h_pw     = w_h.reshape(B * N2, self.feature_dim)
        V_pw_w     = V.unsqueeze(1).expand(B, N2, 3, 3).reshape(B * N2, 3, 3)

        w_pw_out    = self.pw_projection_w(w_x_rel_pw, w_h_pw, V_pw_w).view(B, N2, Nkv, 2)
        w_pw_scalar = w_pw_out[..., 0]

        w_combined_weight     = w_alpha * w_pw_scalar
        w_combined_weight_exp = w_combined_weight.unsqueeze(-1).unsqueeze(-1)

        cross_term_w: Dict[int, torch.Tensor] = {}
        for l in range(self.max_degree + 1):
            C     = f_kv[l].shape[2]
            Wv_kv = Wv_kv_flat[l].view(B, N2, Nkv, C, 2 * l + 1)
            Wv_kv = torch.where(w_pair_mask_grid, Wv_kv, torch.zeros_like(Wv_kv))
            cross_term_w[l] = (w_combined_weight_exp * Wv_kv).sum(dim=2)

        f_0_j = self.W_V_self_j["0"](f_in_j[0]) + cross_term_w[0]
        f_1_j = self.W_V_self_j["1"](f_in_j[1]) + cross_term_w[1]
        f_2_j = self.W_V_self_j["2"](f_in_j[2]) + cross_term_w[2]

        # Implicit long-range interactions
        cav_rad = self._calc_cavity_radius(z_0, f_0_j).reshape(B, 1, 1, 1)
        eps_j   = eps_j

        dip_i = self.eq_lin0i(z_0) * self.eq_lin1i(z_1)
        dip_j = self.eq_lin0j(f_0_j) * self.eq_lin1j(f_1_j)

        reac  = (2 * (eps_j + torch.ones_like(eps_j, device=V.device))) / (2 * eps_j + torch.ones_like(eps_j, device=V.device))

        z_1   = z_1 + reac * dip_i * cav_rad
        f_1_j = f_1_j + reac * dip_j * cav_rad

        return z_0, z_1, z_2, f_0_j, f_1_j, f_2_j

class FeatureProjHead(nn.Module):
    def __init__(self, feature_dim: int, type_2: bool = True, iso: bool = True, hidden_dim: int = 64):
            super().__init__()
            self.feature_dim = feature_dim
            self.iso = iso
            self.type2 = type_2

            self.eq_lin0 = EquivariantLinear(feature_dim, 1)

            if not iso:
                self.register_buffer("sh1_to_cartesian", SH1_TO_CARTESIAN.clone())
                self.iso_linear = nn.Linear(2 * feature_dim, 1)
                self.eq_lin11 = EquivariantLinear(feature_dim, 1)

            if type_2:
                self.register_buffer("identity3", torch.eye(3))
                self.eq_lin2 = EquivariantLinear(feature_dim, 1)
                self.aniso_readout = EquivariantReadout(feature_dim, hidden_dim)
                self.register_buffer("sh2_to_cartesian", SH2_TO_CARTESIAN.clone())

    def f1_to_skew(self, f1_cart):
        vx, vy, vz = f1_cart[..., 0], f1_cart[..., 1], f1_cart[..., 2]
        zeros = torch.zeros_like(vx)
        out = torch.stack(
            [
                torch.stack([zeros, -vz, vy], dim=-1),
                torch.stack([vz, zeros, -vx], dim=-1),
                torch.stack([-vy, vx, zeros], dim=-1),
            ],
            dim=-2,
        )
        return out

    def forward(self, f0: torch.Tensor, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        B = f0.shape[0]

        if self.type2:
            f0m = self.eq_lin0(f0)
            iso_tensor = f0m.view(B, -1, 1) * self.identity3.unsqueeze(0)

            f2m = self.eq_lin2(f2)
            out2_sh = self.aniso_readout(f2m)
            aniso_flat = torch.einsum("ij,bnj->bni", self.sh2_to_cartesian, out2_sh).mean(dim=1).squeeze(1)
            aniso_tensor = _flat_to_traceless_symmetric(aniso_flat)

            if not self.iso:
                f1_skew_emb = self.eq_lin11(f1)
                v1_skew = torch.einsum("ij,bcj->bci", self.sh1_to_cartesian, f1_skew_emb)
                skew_tensor = self.f1_to_skew(v1_skew)

                return iso_tensor + skew_tensor + aniso_tensor
        
            return iso_tensor + aniso_tensor
        
        elif not self.iso:
            f0m = self.eq_lin0(f0).reshape(B, -1, 1, 1)
            f1_skew_emb = self.eq_lin11(f1)
            v1_skew = torch.einsum("ij,bncj->bnci", self.sh1_to_cartesian, f1_skew_emb)

            return f0m * v1_skew

        else:
            f0m = self.eq_lin0(f0).reshape(B, -1, 1, 1)

            return f0m