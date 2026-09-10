from __future__ import annotations

import torch
import torch.nn as nn

from stage_1.model import EnergyModel
from utils.se3_utils import EquivariantLinear, EquivariantNormReadout, EquivariantRMSNorm


class EqFSAClassifier(nn.Module):
    def __init__(
        self,
        radial_net,
        num_atom_types: int,
        max_degree: int,
        num_layers: int,
        feature_dim: int,
        hidden_dim: int,
        edge_cutoff: float,
        n_max: int,
        classes: int,
        pw_e_cutoff: float | None = None,
        n_master_nodes: int = 4,
        num_fusion_heads: int = 4,
        edge_feat_dim: int = 1,
        ablate_max_degree: int | None = None,
        grad_checkpoint: bool = False,
        multi_layer_readout: bool = True,
        normalize_z_by_n_real: bool = True,
        decouple_ri_norm: bool = True,
        imaginary_only_ri: bool = False,
        deterministic_pool_agg: bool = True,
        direction_aware_pool_agg: bool = False,
        attention_pool: bool = True,
        skip_master_expand_norm: bool = True,
        static_box: bool = True,
        content_weighted_com: bool = False,
        per_layer_fusion: bool = True,
        reduce_norms: bool = True,
        dist_weighted_content: bool = False,
        dist_rbf_num_basis: int = 12,
        dist_rbf_min: float = 0.3,
        dist_rbf_cutoff: float = 25.0,
        linear_com: bool = False,
        use_norms: bool = True,
        qk_pre_elu_scale: float = 1.0,
    ):
        super().__init__()
        self.multi_layer_readout = multi_layer_readout
        self.energy_model = EnergyModel(
            radial_net=radial_net,
            num_atom_types=num_atom_types,
            max_degree=max_degree,
            num_layers=num_layers,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            edge_cutoff=edge_cutoff,
            n_max=n_max,
            pw_e_cutoff=pw_e_cutoff,
            n_master_nodes=n_master_nodes,
            num_fusion_heads=num_fusion_heads,
            edge_feat_dim=edge_feat_dim,
            ablate_max_degree=ablate_max_degree,
            normalize_z_by_n_real=normalize_z_by_n_real,
            grad_checkpoint=grad_checkpoint,
            decouple_ri_norm=decouple_ri_norm,
            imaginary_only_ri=imaginary_only_ri,
            deterministic_pool_agg=deterministic_pool_agg,
            direction_aware_pool_agg=direction_aware_pool_agg,
            attention_pool=attention_pool,
            skip_master_expand_norm=skip_master_expand_norm,
            static_box=static_box,
            content_weighted_com=content_weighted_com,
            per_layer_fusion=per_layer_fusion,
            reduce_norms=reduce_norms,
            dist_weighted_content=dist_weighted_content,
            dist_rbf_num_basis=dist_rbf_num_basis,
            dist_rbf_min=dist_rbf_min,
            dist_rbf_cutoff=dist_rbf_cutoff,
            linear_com=linear_com,
            use_norms=use_norms,
            qk_pre_elu_scale=qk_pre_elu_scale,
        )
        self.norm_readout1 = EquivariantNormReadout(C=feature_dim, hidden=hidden_dim, out_dim=feature_dim)
        self.norm_readout2 = EquivariantNormReadout(C=feature_dim, hidden=hidden_dim, out_dim=feature_dim)
        def _head_norm(dim: int) -> nn.Module:
            return EquivariantRMSNorm(dim, degree=0) if use_norms else nn.Identity()
        self.classifier_head = nn.Sequential(
            _head_norm(feature_dim),
            EquivariantLinear(feature_dim, hidden_dim, nonlinearity="leaky_relu"),
            _head_norm(hidden_dim),
            nn.LeakyReLU(),
            EquivariantLinear(hidden_dim, hidden_dim, nonlinearity="leaky_relu"),
            _head_norm(hidden_dim),
            nn.LeakyReLU(),
            EquivariantLinear(hidden_dim, hidden_dim, nonlinearity="leaky_relu"),
            _head_norm(hidden_dim),
            nn.LeakyReLU(),
            EquivariantLinear(hidden_dim, classes),
        )

    @property
    def input_embedding_module(self) -> nn.Embedding:
        return self.energy_model.input_embedding

    def _readout_logits(self, f: dict, batch_ar: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
        f_target = {l: f[l][batch_ar, target_idx] for l in f}
        u_vec = self.norm_readout1(f_target[1])
        u_ten = self.norm_readout2(f_target[2])
        full_vec = f_target[0].squeeze(-1) + u_vec + u_ten
        logits = self.classifier_head(full_vec.unsqueeze(-1))
        return logits.squeeze(-1)

    def forward(
        self,
        x: torch.Tensor,
        types: torch.Tensor,
        masses: torch.Tensor,
        mask: torch.Tensor,
        V: torch.Tensor,
        mol_props: torch.Tensor,
        edge_feats: torch.Tensor,
        target_idx: torch.Tensor,
        seed_masses: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = x.shape[0]
        batch_ar = torch.arange(B, device=x.device)

        if self.multi_layer_readout:
            f, layer_states = self.energy_model(
                x, types, masses, mask, V, mol_props, edge_feats, return_layer_states=True,
                seed_masses=seed_masses, target_idx=target_idx,
            )
            logits = self._readout_logits(f, batch_ar, target_idx)
            for layer_f in layer_states:
                logits = logits + self._readout_logits(layer_f, batch_ar, target_idx)
            return logits

        f = self.energy_model(
            x, types, masses, mask, V, mol_props, edge_feats,
            seed_masses=seed_masses, target_idx=target_idx,
        )
        return self._readout_logits(f, batch_ar, target_idx)
