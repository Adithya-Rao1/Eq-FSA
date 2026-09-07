import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union

from utils.se3_utils import *
from utils.pw_bases import make_wavevectors

def _rms_scalar_norm(t: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    rms = t.pow(2).mean(dim=-1, keepdim=True).clamp(min=eps).sqrt()
    return t / rms * gamma

class QKProjection(nn.Module):
    def __init__(self, radial_net, max_degree: int, feature_dim: int, hidden_dim: int, batch_size=32):
        super().__init__()
        self.max_degree = max_degree
        self.W_Q = nn.ModuleDict({
            str(l): EquivariantLinear(cin=feature_dim, cout=feature_dim)
            for l in range(max_degree + 1)
        })

        self.radial_K: Dict[str, nn.ModuleDict] = nn.ModuleDict()
        for l in range(max_degree + 1):
            for k in range(max_degree + 1):
                key = f"{l}_{k}"
                j_nets = nn.ModuleDict({
                    str(J): radial_net(num_basis=2*l+1, hidden_dim=hidden_dim)
                    for J in range(abs(l - k), l + k + 1)
                })
                self.radial_K[key] = j_nets

    def query(self, f: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        q: Dict[int, torch.Tensor] = {}
        for l in range(self.max_degree + 1):
            if l not in f:
                continue
            q[l] = self.W_Q[str(l)](f[l])   
        return q

    def key(
        self,
        f: Dict[int, torch.Tensor],
        x_rel: torch.Tensor,
    ) -> Dict[int, torch.Tensor]:
        return apply_direct_sum_W(
            f=f,
            x=x_rel,
            radial_nets=self.radial_K,
            max_degree=self.max_degree,
        )

def _equivariant_inner_product(
    a: Dict[int, torch.Tensor],
    b: Dict[int, torch.Tensor],
) -> torch.Tensor:
    result = None
    for l in a:
        if l not in b:
            continue
        contrib = (a[l] * b[l]).sum(dim=(-2, -1))   
        result = contrib if result is None else result + contrib
    if result is None:
        raise ValueError("No shared degrees between query and key.")
    return result

class NormedPairReadout(nn.Module):
    def __init__(self, C: int, hidden: int, floor: float = 0.5):
        super().__init__()
        self.floor = floor
        self.equiv_proj = nn.Linear(C, hidden, bias=False)
        init_linear_(self.equiv_proj, "linear")
        self.hidden_ffn = nn.Sequential(
            nn.Linear(hidden, hidden, bias=False),
            nn.RMSNorm(hidden),
            nn.SiLU(),
        )
        init_sequential_(self.hidden_ffn, "silu")
        self.out = nn.Linear(hidden, 1, bias=False)
        init_linear_(self.out, "linear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self.equiv_proj(x.transpose(-1, -2)).transpose(-1, -2)
        norms = straight_through_norm(v, dim=-1, floor=self.floor)
        h = self.hidden_ffn(norms)
        return self.out(h).squeeze(-1)

class SRAttention(nn.Module):
    def __init__(
        self, radial_net, max_degree: int, feature_dim: int, hidden_dim: int,
        k_max: int, edge_feat_dim: int,
    ):
        super().__init__()
        self.max_degree = max_degree
        self.k_max = k_max

        self.radial_V_sr = nn.ModuleDict()
        for l in range(max_degree + 1):
            for k in range(k_max + 1):
                key = f"{l}_{k}"
                self.radial_V_sr[key] = nn.ModuleDict({
                    str(J): radial_net(num_basis=(2 * l + 1), hidden_dim=hidden_dim)
                    for J in range(abs(l - k), l + k + 1)
                })

        self.pair_readout = nn.ModuleDict({
            str(k): NormedPairReadout(2 * feature_dim, hidden_dim)
            for k in range(k_max + 1)
        })
        
        self.edge_input_norm = nn.RMSNorm(1 + edge_feat_dim)
        self.psi_e = scalar_ffn(1 + edge_feat_dim, hidden_dim)
        self.psi_e_out = nn.Sequential(
            nn.RMSNorm(1 + edge_feat_dim),
            nn.LeakyReLU(),
            nn.Linear(1 + edge_feat_dim, 1),
        )
        init_sequential_(self.psi_e_out, "leaky_relu")

    def forward(
        self,
        f_in_i_e:   Dict[int, torch.Tensor],
        f0_j_e:     Dict[int, torch.Tensor],
        x_rel_e:    torch.Tensor,
        edge_feat_e: torch.Tensor,
        edge_b:     torch.Tensor,
        edge_i:     torch.Tensor,
        batch_size: int,
        num_atoms:  int,
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        score = None
        for k in range(self.k_max + 1):
            cat_k = torch.cat([f_in_i_e[k], f0_j_e[k]], dim=-2)
            contrib = self.pair_readout[str(k)](cat_k)
            score = contrib if score is None else score + contrib

        dist = x_rel_e.norm(dim=-1, keepdim=True)
        edge_feats = self.edge_input_norm(torch.cat([dist, edge_feat_e], dim=-1))
        psi = self.psi_e_out(self.psi_e(edge_feats)).squeeze(-1)

        scores_e = psi * score

        receiver_flat = edge_b * num_atoms + edge_i
        num_receivers = batch_size * num_atoms
        alpha_e = scatter_softmax(scores_e, receiver_flat, num_receivers)

        Wv_j_sr_flat = apply_direct_sum_W(
            f=f0_j_e, x=x_rel_e,
            radial_nets=self.radial_V_sr, max_degree=self.max_degree, k_max=self.k_max,
        )

        return alpha_e, Wv_j_sr_flat

class MasterNodeAttention(nn.Module):
    def __init__(
        self, feature_dim: int, n_max: int, pw_e_cutoff: Optional[float] = None, hidden_dim: int = 128,
        alpha_floor: float = 1e-4,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_max = n_max
        self.pw_e_cutoff = pw_e_cutoff
        self.alpha_floor = alpha_floor

        self.W_Q = high_ord_ffn(feature_dim, hidden_dim, 1)
        self.w_vq_norm = EquivariantRMSNorm(feature_dim, 1)

        self.W_K = high_ord_ffn(feature_dim, hidden_dim, 1)
        self.w_vk_norm = EquivariantRMSNorm(feature_dim, 1)

        self.W_V0 = nn.Sequential(
            scalar_ffn(feature_dim, hidden_dim),
            nn.RMSNorm(feature_dim),
        )
        self.s_ri_norm = nn.RMSNorm(2 * feature_dim)
        self.s_re_norm = nn.RMSNorm(feature_dim)
        self.s_im_norm = nn.RMSNorm(feature_dim)
        self.cos_com_gamma = nn.Parameter(torch.tensor(1.0))
        self.sin_com_gamma = nn.Parameter(torch.tensor(1.0))

        self.spatial_radial = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        init_sequential_(self.spatial_radial, "silu")
        self.spatial_query_proj = nn.Linear(feature_dim, hidden_dim, bias=False)
        init_linear_(self.spatial_query_proj, "linear")

        self.node_k_gate = nn.Linear(feature_dim, feature_dim)
        init_linear_(self.node_k_gate, "linear")

        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.diag_enabled = False
        self._diag_damping_sum_min: Optional[float] = None
        self._diag_Z_min: Optional[float] = None
        self._diag_ipr: Optional[torch.Tensor] = None

    def forward(
        self,
        g_a:       torch.Tensor,
        f_in0:     torch.Tensor,
        x:         torch.Tensor,
        x_com:     torch.Tensor,
        node_mask: torch.Tensor,
        V:         torch.Tensor,
        return_scale_feat: bool = False,
        per_node_com: bool = False,
        node_conditioned_k: bool = False,
        normalize_z_by_n_real: bool = False,
        decouple_ri_norm: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, N_a, C, _ = g_a.shape
        m = node_mask.to(x.dtype)

        Q = self.w_vq_norm(self.W_Q(g_a)).squeeze(-1)
        K = self.w_vk_norm(self.W_K(f_in0)).squeeze(-1)
        phi_Q = F.elu(Q) + 1
        phi_K = (F.elu(K) + 1) * m.unsqueeze(-1)

        if node_conditioned_k:
            node_gate = torch.sigmoid(self.node_k_gate(g_a.squeeze(-1)))
            phi_K_use = phi_K.unsqueeze(1) * node_gate.unsqueeze(2)
            phi_K_sum = phi_K_use.sum(dim=2)
            Z_raw = torch.einsum("bac,bac->ba", phi_Q, phi_K_sum)
        else:
            phi_K_use = phi_K
            phi_K_sum = phi_K.sum(dim=1)
            Z_raw = torch.einsum("bac,bc->ba", phi_Q, phi_K_sum)
        if normalize_z_by_n_real:
            n_real = node_mask.sum(dim=1, keepdim=True).clamp(min=1).to(Z_raw.dtype)
            Z_raw = Z_raw / n_real
        Z = Z_raw.clamp(min=1e-8)
        if self.diag_enabled:
            self._diag_Z_min = float(Z.detach().min())
            with torch.no_grad():
                if node_conditioned_k:
                    w_diag = torch.einsum("bac,bagc->bag", phi_Q, phi_K_use)
                else:
                    w_diag = torch.einsum("bac,bgc->bag", phi_Q, phi_K_use)
                w_diag_norm = w_diag / w_diag.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                self._diag_ipr = 1.0 / w_diag_norm.pow(2).sum(dim=-1).clamp(min=1e-12)

        if per_node_com:
            dist_to_com = (x - x_com.unsqueeze(1)).norm(dim=-1, keepdim=True)
            dist_feat = self.spatial_radial(dist_to_com)
            spatial_q = self.spatial_query_proj(g_a.squeeze(-1))
            spatial_logit = torch.einsum("bah,bgh->bag", spatial_q, dist_feat) / (dist_feat.shape[-1] ** 0.5)
            spatial_logit = spatial_logit.masked_fill(~m.bool().unsqueeze(1), -1e9)
            spatial_w = torch.softmax(spatial_logit, dim=-1)
            x_com_use = torch.einsum("bag,bgi->bai", spatial_w, x)
        else:
            x_com_use = x_com.unsqueeze(1).expand(B, N_a, -1)

        kvecs, _, mode_mask = make_wavevectors(V, self.n_max, self.pw_e_cutoff)
        mode_mask_f = mode_mask.to(x.dtype)
        if mode_mask_f.dim() == 1:
            mode_mask_f = mode_mask_f.unsqueeze(0).expand(B, -1)

        alpha = F.softplus(self.alpha) + self.alpha_floor
        k2 = (kvecs ** 2).sum(dim=-1)
        damping = torch.exp(-k2 / (4.0 * alpha)) * mode_mask_f
        damping_sum = damping.sum(dim=-1) # min=1.0 for k=0
        if self.diag_enabled:
            self._diag_damping_sum_min = float(damping_sum.detach().min())

        phase_g = -torch.einsum("bgi,bmi->bgm", x, kvecs)
        cos_g = torch.cos(phase_g) * mode_mask_f.unsqueeze(1)
        sin_g = torch.sin(phase_g) * mode_mask_f.unsqueeze(1)

        V_proj0 = self.W_V0(f_in0.squeeze(-1)).unsqueeze(-1)

        if node_conditioned_k:
            raw_S_re0 = torch.einsum("bagc,bgnd,bgm->bamcnd", phi_K_use, V_proj0, cos_g)
            raw_S_im0 = torch.einsum("bagc,bgnd,bgm->bamcnd", phi_K_use, V_proj0, sin_g)
        else:
            raw_S_re0 = torch.einsum("bgc,bgnd,bgm->bmcnd", phi_K_use, V_proj0, cos_g)
            raw_S_im0 = torch.einsum("bgc,bgnd,bgm->bmcnd", phi_K_use, V_proj0, sin_g)
        raw_S_ri = torch.cat([raw_S_re0, raw_S_im0], dim=-2)
        if return_scale_feat:
            raw_S_ri_mag = raw_S_ri.norm(dim=-2)
        if decouple_ri_norm:
            S_re0 = self.s_re_norm(raw_S_re0.transpose(-1, -2)).transpose(-1, -2)
            S_im0 = self.s_im_norm(raw_S_im0.transpose(-1, -2)).transpose(-1, -2)
        else:
            S_ri = self.s_ri_norm(raw_S_ri.transpose(-1, -2)).transpose(-1, -2)
            S_re0, S_im0 = S_ri.split(self.feature_dim, dim=-2)

        phase_com = torch.einsum("bai,bmi->bam", x_com_use, kvecs)
        cos_com = torch.cos(phase_com) * damping.unsqueeze(1)
        sin_com = torch.sin(phase_com) * damping.unsqueeze(1)
        if decouple_ri_norm:
            cos_com = _rms_scalar_norm(cos_com, self.cos_com_gamma)
            sin_com = _rms_scalar_norm(sin_com, self.sin_com_gamma)

        if node_conditioned_k:
            QS_re0 = torch.einsum("bac,bamcnd->bamnd", phi_Q, S_re0)
            QS_im0 = torch.einsum("bac,bamcnd->bamnd", phi_Q, S_im0)
        else:
            QS_re0 = torch.einsum("bac,bmcnd->bamnd", phi_Q, S_re0)
            QS_im0 = torch.einsum("bac,bmcnd->bamnd", phi_Q, S_im0)

        cos_b = cos_com.unsqueeze(-1).unsqueeze(-1)
        sin_b = sin_com.unsqueeze(-1).unsqueeze(-1)
        lr_master = (QS_re0 * cos_b - QS_im0 * sin_b).sum(dim=2)
        lr_master = lr_master / Z.view(B, N_a, 1, 1)
        lr_master = lr_master / damping_sum.view(B, 1, 1, 1)

        if not return_scale_feat:
            return lr_master

        if node_conditioned_k:
            Q_mag = torch.einsum("bac,bamcd->bamd", phi_Q, raw_S_ri_mag)
        else:
            Q_mag = torch.einsum("bac,bmcd->bamd", phi_Q, raw_S_ri_mag)
        mag_pooled = torch.einsum("bamd,bm->bad", Q_mag, damping)
        mag_pooled = mag_pooled / Z.view(B, N_a, 1)
        mag_pooled = mag_pooled / damping_sum.view(B, 1, 1)
        scale_feat = torch.log1p(mag_pooled.squeeze(-1))
        return lr_master, scale_feat

class EquivariantLinearRMSNorm(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.lin = EquivariantLinear(feature_dim, feature_dim)
        self.norm = nn.RMSNorm(feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.lin(x)
        return self.norm(y.transpose(-1, -2)).transpose(-1, -2)

class MasterFusionAttention(nn.Module):
    def __init__(self, radial_net, max_degree: int, feature_dim: int, hidden_dim: int, num_heads: int):
        super().__init__()
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_fusion_heads"
        self.max_degree = max_degree
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.scale = self.head_dim ** 0.5

        self.W_Q = nn.ModuleDict({
            str(l): EquivariantLinear(feature_dim, feature_dim) for l in range(max_degree + 1)
        })
        self.readout_K = nn.ModuleDict({
            str(l): EquivariantLinearRMSNorm(feature_dim) for l in range(max_degree + 1)
        })
        self.readout_V = nn.ModuleDict({
            str(l): EquivariantLinearRMSNorm(feature_dim) for l in range(max_degree + 1)
        })
        self.radial_K = nn.ModuleDict({
            f"{l}_0": nn.ModuleDict({str(l): radial_net(num_basis=2 * l + 1, hidden_dim=hidden_dim)})
            for l in range(max_degree + 1)
        })
        self.radial_V = nn.ModuleDict({
            f"{l}_0": nn.ModuleDict({str(l): radial_net(num_basis=2 * l + 1, hidden_dim=hidden_dim)})
            for l in range(max_degree + 1)
        })

    def forward(
        self,
        f_atoms: Dict[int, torch.Tensor],
        g_a:     torch.Tensor,
        x:       torch.Tensor,
        x_com:   torch.Tensor,
    ) -> Dict[int, torch.Tensor]:
        B, N, C, _ = f_atoms[0].shape
        N_a = g_a.shape[1]
        H, Ch = self.num_heads, self.head_dim

        x_rel = x - x_com.unsqueeze(1)
        x_rel_flat = x_rel.reshape(B * N, 3)
        has_dir = (x_rel.norm(dim=-1) > 1e-7).to(x.dtype).view(B, N, 1, 1)

        Q = {l: self.W_Q[str(l)](f_atoms[l]) for l in range(self.max_degree + 1)}

        K_lifted: Dict[int, torch.Tensor] = {}
        V_lifted: Dict[int, torch.Tensor] = {}
        for l in range(self.max_degree + 1):
            K_content = self.readout_K[str(l)](g_a)
            V_content = self.readout_V[str(l)](g_a)

            W_K = equivariant_weight_matrix(x_rel_flat, l, 0, self.radial_K[f"{l}_0"]).view(B, N, 2 * l + 1, 1)
            W_V = equivariant_weight_matrix(x_rel_flat, l, 0, self.radial_V[f"{l}_0"]).view(B, N, 2 * l + 1, 1)
            if l > 0:
                W_K = W_K * has_dir
                W_V = W_V * has_dir

            K_lifted[l] = torch.einsum("bnij,bacj->bnaci", W_K, K_content)
            V_lifted[l] = torch.einsum("bnij,bacj->bnaci", W_V, V_content)

        score = None
        for l in range(self.max_degree + 1):
            Qh = Q[l].view(B, N, H, Ch, 2 * l + 1)
            Kh = K_lifted[l].view(B, N, N_a, H, Ch, 2 * l + 1)
            contrib = torch.einsum("bnhcd,bnahcd->bnah", Qh, Kh)
            score = contrib if score is None else score + contrib
        score = score / self.scale

        alpha = softmax_over_neighbors(score.permute(0, 1, 3, 2))
        alpha = alpha.permute(0, 1, 3, 2)

        out: Dict[int, torch.Tensor] = {}
        for l in range(self.max_degree + 1):
            Vh = V_lifted[l].view(B, N, N_a, H, Ch, 2 * l + 1)
            weighted = torch.einsum("bnah,bnahcd->bnhcd", alpha, Vh)
            out[l] = weighted.reshape(B, N, C, 2 * l + 1)

        return out

class SRCrossAttention(nn.Module):
    def __init__(self, radial_net, max_degree: int, feature_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.max_degree = max_degree
        self.qk = QKProjection(radial_net, max_degree, feature_dim, hidden_dim)
        d = sum(2 * l + 1 for l in range(max_degree + 1))
        self.scale = d ** 0.5

    def forward(
        self,
        f_in_i:    Dict[int, torch.Tensor],   
        f_in_j:    Dict[int, torch.Tensor],   
        x_rel:     torch.Tensor,              
        pair_mask: torch.Tensor,             
    ) -> torch.Tensor:
        B, N1, N2, _ = x_rel.shape
        C = next(iter(f_in_i.values())).shape[2]

        q = self.qk.query(f_in_i)

        x_rel_flat = x_rel.reshape(B * N1 * N2, 3)

        f_j_flat = {
            l: f_in_j[l].unsqueeze(1).expand(B, N1, N2, C, 2 * l + 1).reshape(B * N1 * N2, C, 2 * l + 1)
            for l in f_in_j
        }
        k_flat = self.qk.key(f_j_flat, x_rel_flat)

        q_expanded = {
            l: q[l].unsqueeze(2).expand(B, N1, N2, C, 2 * l + 1).reshape(B * N1 * N2, C, 2 * l + 1)
            for l in q
        }

        scores = _equivariant_inner_product(q_expanded, k_flat) / self.scale
        scores = scores.view(B, N1, N2)

        return softmax_over_neighbors(scores, pair_mask)
