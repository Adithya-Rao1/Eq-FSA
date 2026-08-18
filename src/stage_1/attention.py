import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from utils.se3_utils import *
from utils.pw_bases import make_wavevectors

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
            str(k): EquivariantNormReadout(2 * feature_dim, hidden_dim)
            for k in range(k_max + 1)
        })
        self.score_act = nn.SiLU()

        self.edge_input_norm = nn.LayerNorm(1 + edge_feat_dim)
        self.psi_e = scalar_ffn(1 + edge_feat_dim, hidden_dim)
        self.psi_e_out = nn.Sequential(
            nn.LayerNorm(1 + edge_feat_dim),
            nn.LeakyReLU(),
            nn.Linear(1 + edge_feat_dim, 1)
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
        score = self.score_act(score)

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
    def __init__(self, feature_dim: int, n_max: int, pw_e_cutoff: Optional[float] = None, hidden_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_max = n_max
        self.pw_e_cutoff = pw_e_cutoff

        self.W_Q = high_ord_ffn(feature_dim, hidden_dim, 1)
        self.w_vq_norm = EquivariantLayerNorm(feature_dim, 1)

        self.W_K = high_ord_ffn(feature_dim, hidden_dim, 1)
        self.w_vk_norm = EquivariantLayerNorm(feature_dim, 1)

        self.W_V0 = high_ord_ffn(feature_dim, hidden_dim, 1)
        self.w_v0_norm = EquivariantLayerNorm(feature_dim, 1)

        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.diag_enabled = False
        self._diag_damping_sum_min: Optional[float] = None
        self._diag_Z_min: Optional[float] = None

    def forward(
        self,
        g_a:       torch.Tensor,
        f_in0:     torch.Tensor,
        x:         torch.Tensor,
        x_com:     torch.Tensor,
        node_mask: torch.Tensor,
        V:         torch.Tensor,
    ) -> torch.Tensor:
        B, N_a, C, _ = g_a.shape
        m = node_mask.to(x.dtype)

        Q = self.w_vq_norm(self.W_Q(g_a)).squeeze(-1)
        K = self.w_vk_norm(self.W_K(f_in0)).squeeze(-1)
        phi_Q = F.elu(Q) + 1
        phi_K = (F.elu(K) + 1) * m.unsqueeze(-1)
        phi_K_sum = phi_K.sum(dim=1)
        Z_raw = torch.einsum("bac,bc->ba", phi_Q, phi_K_sum)
        Z = Z_raw.clamp(min=1e-8)
        if self.diag_enabled:
            self._diag_Z_min = float(Z.detach().min())

        kvecs, _, mode_mask = make_wavevectors(V, self.n_max, self.pw_e_cutoff)
        mode_mask_f = mode_mask.to(x.dtype)
        if mode_mask_f.dim() == 1:
            mode_mask_f = mode_mask_f.unsqueeze(0).expand(B, -1)

        alpha = F.softplus(self.alpha) + 1e-4
        k2 = (kvecs ** 2).sum(dim=-1)
        damping = torch.exp(-k2 / (4.0 * alpha)) * mode_mask_f
        damping_sum = damping.sum(dim=-1) # min=1.0 for k=0
        if self.diag_enabled:
            self._diag_damping_sum_min = float(damping_sum.detach().min())

        phase_g = -torch.einsum("bgi,bmi->bgm", x, kvecs)
        cos_g = torch.cos(phase_g) * mode_mask_f.unsqueeze(1)
        sin_g = torch.sin(phase_g) * mode_mask_f.unsqueeze(1)

        V_proj0 = self.w_v0_norm(self.W_V0(f_in0))

        S_re0 = torch.einsum("bgc,bgnd,bgm->bmcnd", phi_K, V_proj0, cos_g)
        S_im0 = torch.einsum("bgc,bgnd,bgm->bmcnd", phi_K, V_proj0, sin_g)

        phase_com = torch.einsum("bi,bmi->bm", x_com, kvecs)
        cos_com = torch.cos(phase_com) * damping
        sin_com = torch.sin(phase_com) * damping

        QS_re0 = torch.einsum("bac,bmcnd->bamnd", phi_Q, S_re0)
        QS_im0 = torch.einsum("bac,bmcnd->bamnd", phi_Q, S_im0)

        cos_b = cos_com.view(B, 1, -1, 1, 1)
        sin_b = sin_com.view(B, 1, -1, 1, 1)
        lr_master = (QS_re0 * cos_b - QS_im0 * sin_b).sum(dim=2)
        lr_master = lr_master / Z.view(B, N_a, 1, 1)
        lr_master = lr_master / damping_sum.view(B, 1, 1, 1)

        return lr_master

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
            str(l): EquivariantLinear(feature_dim, feature_dim) for l in range(max_degree + 1)
        })
        self.readout_V = nn.ModuleDict({
            str(l): EquivariantLinear(feature_dim, feature_dim) for l in range(max_degree + 1)
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
