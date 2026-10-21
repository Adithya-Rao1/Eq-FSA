from __future__ import annotations

import argparse
import collections
import csv
import math
import os
import time
from datetime import datetime, timezone

import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.utils.data import DataLoader, random_split

from stage_1.model import EnergyModel, masked_mean, masked_weighted_com, SH1_TO_CARTESIAN
from utils.pw_bases import minimum_image
from stage_1.lr_restructure.model import LREnergyModel
from utils.se3_utils import RadialNetwork, EquivariantNormReadout, EquivariantRMSNorm, EquivariantLinear
from stage_1.spice_dataset import SPICEDataset, spice_collate_fn, _EDGE_FEAT_DIM
from log_setup.utils import logger
from diagnostics import counters
from diagnostics.hooks import attach_hooks, attach_grad_hooks
from diagnostics.provenance import ProvenanceLogger
from diagnostics.drift import DriftDetector


@torch._dynamo.disable()
def _solve_torque_alpha(I_com: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    return (torch.linalg.pinv(I_com, hermitian=True) @ tau.unsqueeze(-1)).squeeze(-1)


class SPICEEnergyForceModel(nn.Module):
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
        pw_e_cutoff: float | None = None,
        n_master_nodes: int = 4,
        backbone: str = "lr_restructure",
        edge_feat_dim: int = _EDGE_FEAT_DIM,
        # production (EnergyModel) only
        num_fusion_heads: int = 4,
        ablate_max_degree: int | None = None,
        grad_checkpoint: bool = False,
        # lr_restructure (LREnergyModel) only
        num_broadcast_heads: int = 4,
        rbf_num_basis: int = 12,
        rbf_min: float = 0.3,
        rbf_cutoff: float = 25.0,
        sigma: float | None = None,
        alpha_floor: float = 1e-4,
        use_long_range: bool = True,
        # (lr_restructure backbone only -- EnergyModel/production doesn't take this yet)
        use_norms: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        if backbone == "lr_restructure":
            self.energy_model = LREnergyModel(
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
                num_broadcast_heads=num_broadcast_heads,
                edge_feat_dim=edge_feat_dim,
                rbf_num_basis=rbf_num_basis,
                rbf_min=rbf_min,
                rbf_cutoff=rbf_cutoff,
                sigma=sigma,
                alpha_floor=alpha_floor,
                use_norms=use_norms,
                use_long_range=use_long_range,
            )
        elif backbone == "production":
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
                grad_checkpoint=grad_checkpoint,
            )
        else:
            raise ValueError(f"Unknown backbone: {backbone!r} (expected 'lr_restructure' or 'production')")
        self.norm_readout1 = EquivariantNormReadout(C=feature_dim, hidden=hidden_dim)
        self.norm_readout2 = EquivariantNormReadout(C=feature_dim, hidden=hidden_dim)
        self.out = nn.Sequential(
            EquivariantRMSNorm(feature_dim, degree=0) if use_norms else nn.Identity(),
            EquivariantLinear(feature_dim, hidden_dim, nonlinearity="leaky_relu"),
            EquivariantRMSNorm(hidden_dim, degree=0) if use_norms else nn.Identity(),
            nn.LeakyReLU(),
            EquivariantLinear(hidden_dim, hidden_dim, nonlinearity="leaky_relu"),
            EquivariantRMSNorm(hidden_dim, degree=0) if use_norms else nn.Identity(),
            nn.LeakyReLU(),
            EquivariantLinear(hidden_dim, hidden_dim, nonlinearity="leaky_relu"),
            EquivariantRMSNorm(hidden_dim, degree=0) if use_norms else nn.Identity(),
            nn.LeakyReLU(),
            EquivariantLinear(hidden_dim, 1),
        )

        self.direct_force_proj = EquivariantLinear(feature_dim, 1)
        self.register_buffer("sh1_to_cartesian", SH1_TO_CARTESIAN.clone())

    def _compute_direct_forces(
        self,
        f: dict,
        x: torch.Tensor,
        masses: torch.Tensor,
        mask: torch.Tensor,
        V: torch.Tensor,
    ) -> torch.Tensor:
        m = mask.to(x.dtype)

        f1_sh = self.direct_force_proj(f[1]).squeeze(-2)  
        sh1_to_cartesian = self.sh1_to_cartesian.to(device=x.device, dtype=f1_sh.dtype)
        F_raw = torch.einsum("ij,bnj->bni", sh1_to_cartesian, f1_sh)  # Cartesian

        n_real = m.sum(dim=1, keepdim=True).clamp(min=1.0).unsqueeze(-1)
        F_mean = (F_raw * m.unsqueeze(-1)).sum(dim=1, keepdim=True) / n_real
        F1 = F_raw - F_mean

        ref_idx = m.argmax(dim=1)
        x_ref = x[torch.arange(x.shape[0], device=x.device), ref_idx]
        x_tilde = x_ref.unsqueeze(1) + minimum_image(x - x_ref.unsqueeze(1), V)
        r_com, _ = masked_weighted_com(x_tilde, masses, mask, V=None)
        r_rel = (x_tilde - r_com.unsqueeze(1)) * m.unsqueeze(-1)
        masses_m = masses * m

        tau = (torch.cross(r_rel, F1, dim=-1) * m.unsqueeze(-1)).sum(dim=1)  # (B, 3)

        r_outer = torch.einsum("bni,bnj->bnij", r_rel, r_rel)
        r_sq = (r_rel ** 2).sum(dim=-1)
        eye3 = torch.eye(3, device=x.device, dtype=x.dtype)
        I_terms = masses_m.unsqueeze(-1).unsqueeze(-1) * (
            r_sq.unsqueeze(-1).unsqueeze(-1) * eye3 - r_outer
        )
        I_com = I_terms.sum(dim=1)
        alpha = _solve_torque_alpha(I_com, tau)
        correction = masses_m.unsqueeze(-1) * torch.cross(
            alpha.unsqueeze(1).expand_as(r_rel), r_rel, dim=-1,
        )
        return F1 - correction

    def forward(
        self,
        x: torch.Tensor,
        types_: torch.Tensor,
        masses: torch.Tensor,
        mask: torch.Tensor,
        V: torch.Tensor,
        mol_props: torch.Tensor,
        edge_feats: torch.Tensor,
        force_mode: str = "energy_only",
    ):
        f = self.energy_model(
            x, types_, masses, mask, V, mol_props, edge_feats,
        )
        f_pooled = {l: masked_mean(f[l], mask) for l in f}

        u_vec = self.norm_readout1(f_pooled[1])
        u_ten = self.norm_readout2(f_pooled[2])
        full_vec = f_pooled[0].squeeze(-1) + u_vec.unsqueeze(-1) + u_ten.unsqueeze(-1)
        out = self.out(full_vec.unsqueeze(-1))
        energy = out.squeeze(-1).squeeze(-1)

        if force_mode == "energy_only":
            return energy
        if force_mode in ("direct", "both"):
            direct_forces = self._compute_direct_forces(f, x, masses, mask, V)
            return energy, direct_forces
        raise ValueError(
            f"Unknown force_mode: {force_mode!r} (expected 'energy_only', 'direct', or 'both')"
        )

def compute_energy_and_forces(
    model: SPICEEnergyForceModel,
    x: torch.Tensor,
    types_: torch.Tensor,
    masses: torch.Tensor,
    mask: torch.Tensor,
    V: torch.Tensor,
    mol_props: torch.Tensor,
    edge_feats: torch.Tensor,
    force_mode: str = "conservative",
    model_fn=None,
):
    call = model_fn if model_fn is not None else model
    if force_mode == "conservative":
        x = x.clone().requires_grad_(True)
        U = call(x, types_, masses, mask, V, mol_props, edge_feats)
        (grad_x,) = torch.autograd.grad(U.sum(), x, create_graph=model.training)
        F = -grad_x
        return U, F
    if force_mode == "direct":
        U, F = call(x, types_, masses, mask, V, mol_props, edge_feats, force_mode="direct")
        return U, F
    if force_mode == "both":
        x = x.clone().requires_grad_(True)
        U, F_direct = call(x, types_, masses, mask, V, mol_props, edge_feats, force_mode="direct")
        (grad_x,) = torch.autograd.grad(U.sum(), x, create_graph=model.training)
        F_cons = -grad_x
        return U, F_direct, F_cons
    raise ValueError(f"Unknown force_mode: {force_mode!r} (expected 'conservative', 'direct', or 'both')")


def spice_loss(
    U: torch.Tensor,
    U_hat: torch.Tensor,
    F: torch.Tensor,
    F_hat: torch.Tensor,
    mask: torch.Tensor,
    lambda1: float,
    lambda2: float,
    F_cons: torch.Tensor | None = None,
    F_cons_hat: torch.Tensor | None = None,
    lambda2_conservative: float | None = None,
    lambda_force_match: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    energy_term = lambda1 * (U - U_hat) ** 2
    force_sq = ((F - F_hat) ** 2).sum(dim=-1) * mask
    n_atoms = mask.sum(dim=1).clamp(min=1)
    force_term = lambda2 * (force_sq.sum(dim=1) / n_atoms)
    total = energy_term + force_term

    if F_cons is not None and F_cons_hat is not None and lambda2_conservative:
        cons_sq = ((F_cons - F_cons_hat) ** 2).sum(dim=-1) * mask
        total = total + lambda2_conservative * (cons_sq.sum(dim=1) / n_atoms)

    if F_cons is not None and lambda_force_match:
        match_sq = ((F - F_cons) ** 2).sum(dim=-1) * mask
        total = total + lambda_force_match * (match_sq.sum(dim=1) / n_atoms)

    return total.mean(), energy_term.mean(), force_term.mean()


_GRAD_BUCKET_NAMES = [
    "sr_psi", "sr_pairread", "sr_radial", "sr_selfnorm", "sr_selflin",
    "master", "fusion",
    "other_seedradial", "other_molpropemb", "other_f0ffn", "other_molffn",
    "other_seedprojout", "other_seedprojnorm", "other_seedproj", "other_inputembed",
    "other_layernorm", "other_gate", "other_readout", "other_final", "other",
]
_CSV_FIELDS = [
    "timestamp", "epoch", "batch_idx", "global_step", "phase", "loss", "grad_norm",
    *_GRAD_BUCKET_NAMES,
    *[f"weight_{b}" for b in _GRAD_BUCKET_NAMES],
    "energy_term", "force_term", "lr",
    "alpha", "damping_sum_min", "Z_min", "ln_clamp_hits", "safe_inv_retries",
    "max_force_mag", "max_abs_U",
    "min_abs_radial", "eqln_deg_gt0_near_degenerate", "g_a_norm_max",
    "skipped",
]


class SPICETrainer:
    def __init__(
        self,
        model: SPICEEnergyForceModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.global_step = 0
        self.diag = bool(config.get("diagnostics", False))
        self._last_batch_stats: dict = {}
        self._setup_csv_logging()
        self._setup_training()
        self._setup_diagnostics()
        self._setup_explosion_skip()

    def _setup_diagnostics(self):
        if not self.diag:
            self.mag_tracker = None
            self.grad_tracker = None
            self.provenance = None
            self.drift = None
            return
            
        self._diag_skip_hooks = bool(self.config.get("diag_skip_hooks", False))
        self._diag_skip_buckets = bool(self.config.get("diag_skip_buckets", False))

        counters.set_enabled(True)
        if self._diag_skip_hooks:
            self.mag_tracker = None
            self.grad_tracker = None
        else:
            self.mag_tracker = attach_hooks(self.model)
            self.grad_tracker = attach_grad_hooks(self.model)
        self.model.energy_model.diag_enabled = True
        for layer in self.model.energy_model.layers:
            layer.diag_enabled = True
            if hasattr(layer, "master_attn"):
                layer.master_attn.diag_enabled = True
        if self.model.backbone != "production":
            logger.info(
                "Diagnostics: alpha/damping_sum_min/Z_min/g_a_norm_max risk scalars are "
                "production-backbone instrumentation (MasterNodeAttention-specific) and will "
                f"read empty for backbone={self.model.backbone!r}; grad/weight-norm bucketing, "
                "hooks, provenance, and drift detection still apply."
            )

        ckpt_dir = self.config.get("ckpt_dir", ".")
        self.provenance = ProvenanceLogger(os.path.join(ckpt_dir, "provenance.jsonl"))
        self.drift = DriftDetector(
            _GRAD_BUCKET_NAMES,
            decay=self.config.get("diag_ema_decay", 0.98),
            k=self.config.get("diag_drift_k", 6.0),
        )
        self._diag_ring = collections.deque(maxlen=self.config.get("diag_ring_k", 50))
        self._diag_dump_dir = self.config.get("diag_dump_dir") or os.path.join(ckpt_dir, "diag_dumps")
        self._diag_ckpt_every = self.config.get("diag_ckpt_every", 200)
        self._diag_ckpt_keep = self.config.get("diag_ckpt_keep", 3)
        logger.info(
            f"Diagnostics enabled. Provenance: {self.provenance.path}, dumps: {self._diag_dump_dir}"
        )

    def _setup_explosion_skip(self):
        self._n_skipped_nonfinite = 0

    def _setup_csv_logging(self):
        csv_path = self.config.get("csv_path") or os.path.join(self.config.get("ckpt_dir", "."), "loss_log.csv")
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        self.csv_path = csv_path
        write_header = not (os.path.exists(csv_path) and os.path.getsize(csv_path) > 0)
        self._csv_file = open(csv_path, "a", newline="")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=_CSV_FIELDS)
        if write_header:
            self._csv_writer.writeheader()
            self._csv_file.flush()
        logger.info(f"Loss CSV: {csv_path}")

    def _log_csv_row(
        self, phase: str, epoch: int, batch_idx: int | None = None, loss: float | None = None,
        grad_norm: float | None = None, buckets: dict[str, float] | None = None,
        weight_buckets: dict[str, float] | None = None,
        energy_term: float | None = None, force_term: float | None = None, lr: float | None = None,
        alpha: float | None = None, damping_sum_min: float | None = None, Z_min: float | None = None,
        ln_clamp_hits: int | None = None, safe_inv_retries: int | None = None,
        max_force_mag: float | None = None, max_abs_U: float | None = None,
        min_abs_radial: float | None = None, eqln_deg_gt0_near_degenerate: int | None = None,
        g_a_norm_max: float | None = None, skipped: int | None = None,
    ):
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "epoch": epoch,
            "batch_idx": batch_idx,
            "global_step": self.global_step,
            "phase": phase,
            "loss": loss,
            "grad_norm": grad_norm,
            **{b: (buckets or {}).get(b) for b in _GRAD_BUCKET_NAMES},
            **{f"weight_{b}": (weight_buckets or {}).get(b) for b in _GRAD_BUCKET_NAMES},
            "energy_term": energy_term,
            "force_term": force_term,
            "lr": lr,
            "alpha": alpha,
            "damping_sum_min": damping_sum_min,
            "Z_min": Z_min,
            "ln_clamp_hits": ln_clamp_hits,
            "safe_inv_retries": safe_inv_retries,
            "max_force_mag": max_force_mag,
            "max_abs_U": max_abs_U,
            "min_abs_radial": min_abs_radial,
            "eqln_deg_gt0_near_degenerate": eqln_deg_gt0_near_degenerate,
            "g_a_norm_max": g_a_norm_max,
            "skipped": skipped,
        }
        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def close(self):
        if getattr(self, "_csv_file", None) is not None and not self._csv_file.closed:
            self._csv_file.close()
        if getattr(self, "mag_tracker", None) is not None:
            self.mag_tracker.remove()
        if getattr(self, "grad_tracker", None) is not None:
            self.grad_tracker.remove()
        if getattr(self, "provenance", None) is not None:
            self.provenance.close()

    def _setup_training(self):
        config = self.config
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config["lr"],
            weight_decay=config.get("weight_decay", 1e-4),
            fused=(self.device.type == "cuda"),
        )

        self.force_mode = config.get("force_mode", "conservative")
        self.train_model = self.model
        if config.get("torch_compile"):
            if self.force_mode == "direct":
                compile_kwargs = {}
                if config.get("compile_mode"):
                    compile_kwargs["mode"] = config["compile_mode"]
                self.train_model = torch.compile(self.model, **compile_kwargs)
            else:
                logger.warning(
                    f"--torch-compile was requested but force_mode={self.force_mode!r} still "
                    "needs the autograd-forces double-backward graph (create_graph=True) -- "
                    "compiling only helps in force_mode='direct'. Falling back to eager."
                )

        steps_per_epoch = max(len(self.train_loader), 1)
        self.warmup_steps = max(config.get("warmup_epochs", 1) * steps_per_epoch, 1)
        self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=self.warmup_steps,
        )
        self.plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min",
            factor=config.get("plateau_factor", 0.9),
            patience=config.get("plateau_patience", 50),
            min_lr=config.get("min_lr", 1e-6),
        )

        logger.info("SPICE stage-1 training setup:")
        logger.info(f"  Device:              {self.device}")
        logger.info(f"  lr:                  {config['lr']:.2e}")
        logger.info(f"  lambda1 / lambda2:   {config['lambda1']} / {config['lambda2']}")
        logger.info(
            f"  force_mode:          {self.force_mode}  "
            f"(torch.compile active: {self.train_model is not self.model})"
        )
        logger.info(f"  Total training steps:{steps_per_epoch * config['n_epochs']}  (warmup_steps={self.warmup_steps})")
        logger.info(
            f"  Model config:        num_layers={config.get('num_layers')} max_degree={config.get('max_degree')} "
            f"feature_dim={config.get('feature_dim')} hidden_dim={config.get('hidden_dim')} "
            f"edge_cutoff={config.get('edge_cutoff')} n_max={config.get('n_max')} "
            f"pw_e_cutoff={config.get('pw_e_cutoff')} grad_clip_norm={config.get('grad_clip_norm')}"
        )

    def _to_device(self, batch: dict) -> dict:
        out = {}
        for k, v in batch.items():
            out[k] = v.to(self.device) if torch.is_tensor(v) else v
        return out

    def _step(self, batch: dict, use_compiled: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = self._to_device(batch)
        result = compute_energy_and_forces(
            self.model,
            batch["all_solute_x"], batch["all_solute_types"],
            batch["all_solute_masses"], batch["all_solute_mask"], batch["V"],
            batch['mol_props'], batch['all_solute_edge_feats'],
            force_mode=self.force_mode,
            model_fn=self.train_model if use_compiled else self.model,
        )
        F_cons = None
        if self.force_mode == "both":
            U, F, F_cons = result
        else:
            U, F = result

        energy_mean = self.config["energy_mean"]
        energy_std = self.config["energy_std"]
        force_rms = self.config["force_rms"]

        U_n = (U - energy_mean) / energy_std
        U_target_n = (batch["energy"] - energy_mean) / energy_std
        F_n = F / force_rms
        F_target_n = batch["solute_forces"] / force_rms
        F_cons_n = F_cons / force_rms if F_cons is not None else None

        if self.diag:
            mask = batch["all_solute_mask"].bool()
            force_mag = F.detach().norm(dim=-1)
            self._last_batch_stats = {
                "max_force_mag": float(force_mag[mask].max().item()) if mask.any() else 0.0,
                "max_abs_U": float(U.detach().abs().max().item()),
                "idx": batch.get("idx", []),
                "smiles": batch.get("smiles", []),
            }

        return spice_loss(
            U_n, U_target_n, F_n, F_target_n, batch["all_solute_mask"],
            self.config["lambda1"], self.config["lambda2"],
            F_cons=F_cons_n, F_cons_hat=F_target_n if F_cons_n is not None else None,
            lambda2_conservative=self.config.get("lambda2_conservative"),
            lambda_force_match=self.config.get("lambda_force_match", 0.0),
        )

    @staticmethod
    def _bucket_name(name: str) -> str:
        # "sr" split by submodule
        if "sr_attn.psi_e" in name:
            return "sr_psi"
        if "sr_attn.pair_readout" in name:
            return "sr_pairread"
        if "sr_attn.direct_sum_w" in name or "radial_V_sr" in name:
            return "sr_radial"
        if "W_V_self_norm" in name:
            return "sr_selfnorm"
        if "W_V_self_linear" in name:
            return "sr_selflin"
        if "sr_attn" in name:
            return "sr_radial"  
        if ("master_attn" in name or "W_master_self" in name or "master_pool" in name
            or "master_expand" in name
            or ".aggregator." in name or "master_ln" in name
        ):
            return "master"
        if (
            "master_fusion" in name
            or ".broadcast." in name or ".update." in name
        ):
            return "fusion"

        if "seed_radial_net" in name:
            return "other_seedradial"
        if "mol_prop_emb" in name:
            return "other_molpropemb"
        if "f0_ffn" in name:
            return "other_f0ffn"
        if "mol_ffn" in name:
            return "other_molffn"
        if "seed_proj_out" in name:
            return "other_seedprojout"
        if "seed_proj_norm" in name:
            return "other_seedprojnorm"
        if "seed_proj" in name:
            return "other_seedproj"

        if "input_embedding" in name:
            return "other_inputembed"
        if "layer_norm_in" in name:
            return "other_layernorm"
        if "sigma_gate" in name:
            return "other_gate"
        if ".readout." in name:
            return "other_readout"
        if "norm_readout" in name or name.startswith("out."):
            return "other_final"
        return "other"

    def _grad_norm_buckets(self) -> dict[str, float]:
        sums = dict(self._EMPTY_BUCKETS)
        for name, p in self.model.named_parameters():
            if p.grad is None:
                continue
            sums[self._bucket_name(name)] += float((p.grad ** 2).sum().item())
        return {k: v ** 0.5 for k, v in sums.items()}

    def _weight_norm_buckets(self) -> dict[str, float]:
        sums = dict(self._EMPTY_BUCKETS)
        for name, p in self.model.named_parameters():
            sums[self._bucket_name(name)] += float((p.data ** 2).sum().item())
        return {k: v ** 0.5 for k, v in sums.items()}

    def _risk_scalars(self) -> dict:
        alphas, damping_mins, Z_mins, g_a_norms = [], [], [], []
        if self.model.backbone == "production":
            for layer in self.model.energy_model.layers:
                ma = layer.master_attn
                alphas.append(float(nnf.softplus(ma.alpha).item() + 1e-4))
                if ma._diag_damping_sum_min is not None:
                    damping_mins.append(ma._diag_damping_sum_min)
                if ma._diag_Z_min is not None:
                    Z_mins.append(ma._diag_Z_min)
                if layer._diag_g_a_norm is not None:
                    g_a_norms.append(layer._diag_g_a_norm)
        counter_snap = counters.snapshot_and_reset()
        min_abs_radial = getattr(self.model.energy_model, "_diag_stats", {}).get("min_abs_radial")
        return {
            "alpha": min(alphas) if alphas else None,
            "damping_sum_min": min(damping_mins) if damping_mins else None,
            "Z_min": min(Z_mins) if Z_mins else None,
            "ln_clamp_hits": counter_snap.get("ln_clamp", 0),
            "safe_inv_retries": counter_snap.get("safe_inv_retry", 0),
            "min_abs_radial": min_abs_radial,
            "eqln_deg_gt0_near_degenerate": counter_snap.get("eqln_deg_gt0_near_degenerate", 0),
            "g_a_norm_max": max(g_a_norms) if g_a_norms else None,
        }

    _EMPTY_BUCKETS = {b: 0.0 for b in _GRAD_BUCKET_NAMES}
    _EMPTY_RISK_SCALARS = {
        "alpha": None, "damping_sum_min": None, "Z_min": None, "ln_clamp_hits": 0,
        "safe_inv_retries": 0, "min_abs_radial": None, "eqln_deg_gt0_near_degenerate": 0,
        "g_a_norm_max": None,
    }

    def _diag_step_record(self, epoch: int, batch_idx: int) -> dict:
        stats = self._last_batch_stats
        if self._diag_skip_buckets:
            grad_buckets, weight_buckets, risk = dict(self._EMPTY_BUCKETS), dict(self._EMPTY_BUCKETS), dict(self._EMPTY_RISK_SCALARS)
        else:
            grad_buckets, weight_buckets, risk = self._grad_norm_buckets(), self._weight_norm_buckets(), self._risk_scalars()
        return {
            "epoch": epoch,
            "batch_idx": batch_idx,
            "grad_buckets": grad_buckets,
            "weight_buckets": weight_buckets,
            **risk,
            "max_force_mag": stats.get("max_force_mag"),
            "max_abs_U": stats.get("max_abs_U"),
            "idx": stats.get("idx", []),
            "smiles": stats.get("smiles", []),
            "mag_stats": dict(self.mag_tracker.stats) if self.mag_tracker else {},
            "grad_hook_stats": dict(self.grad_tracker.stats) if self.grad_tracker else {},
        }

    def _save_ring_checkpoint(self):
        os.makedirs(self._diag_dump_dir, exist_ok=True)
        slot = (self.global_step // self._diag_ckpt_every) % self._diag_ckpt_keep
        path = os.path.join(self._diag_dump_dir, f"ring_ckpt_{slot}.pt")
        torch.save({
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "warmup_scheduler_state_dict": self.warmup_scheduler.state_dict(),
            "plateau_scheduler_state_dict": self.plateau_scheduler.state_dict(),
            "config": self.config,
        }, path)
        logger.info(f"[DIAG] Ring checkpoint saved: {path}")

    def _dump_explosion(self, epoch: int, batch_idx: int, batch: dict, diag_record: dict, reasons: dict):
        os.makedirs(self._diag_dump_dir, exist_ok=True)
        path = os.path.join(self._diag_dump_dir, f"explosion_step{self.global_step}.pt")
        torch.save({
            "global_step": self.global_step,
            "epoch": epoch,
            "batch_idx": batch_idx,
            "reasons": reasons,
            "ring_history": list(self._diag_ring),
            "batch": {k: v for k, v in batch.items() if torch.is_tensor(v) or isinstance(v, list)},
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "hook_stats": diag_record.get("mag_stats", {}),
            "grad_stats": diag_record.get("grad_hook_stats", {}),
            "config": self.config,
        }, path)
        logger.warning(f"[DIAG] Drift/explosion trigger at step {self.global_step}: {reasons}. Dumped to {path}")
        logger.warning(
            f"Replay with: python -m diagnostics.replay_batch --path {path}"
        )

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        logger.info(f"Epoch {epoch}")
        epoch_loss, epoch_start = 0.0, time.time()
        epoch_skipped_nonfinite = 0

        for batch_idx, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad(set_to_none=True)
            if self.diag:
                if self.mag_tracker is not None:
                    self.mag_tracker.reset()
                if self.grad_tracker is not None:
                    self.grad_tracker.reset()

            loss, energy_term, force_term = self._step(batch)
            loss.backward()

            grads = [p.grad for p in self.model.parameters() if p.grad is not None]
            total_norm_sq = torch.zeros((), device=self.device)
            if grads:
                norms = torch._foreach_norm(grads)
                total_norm_sq = torch.stack(norms).pow(2).sum()
            is_finite = bool(torch.isfinite(total_norm_sq).item())

            should_log = batch_idx % self.config.get("log_interval", 10) == 0

            diag_record = None
            if self.diag:
                diag_record = self._diag_step_record(epoch, batch_idx)
                self._diag_ring.append(diag_record)
                triggered, reasons = self.drift.update_and_check(
                    diag_record["grad_buckets"], self.config.get("diag_explode_threshold", 50.0),
                )
                if triggered:
                    self._dump_explosion(epoch, batch_idx, batch, diag_record, reasons)
                if self.global_step > 0 and self.global_step % self._diag_ckpt_every == 0:
                    self._save_ring_checkpoint()

            if diag_record is not None:
                buckets = diag_record["grad_buckets"]
            elif is_finite and should_log:
                buckets = self._grad_norm_buckets()
            else:
                buckets = dict(self._EMPTY_BUCKETS)

            if not is_finite:
                bad_params = [
                    n for n, p in self.model.named_parameters()
                    if p.grad is not None and not torch.isfinite(p.grad).all()
                ]
                logger.warning(f"  [WARN] Non-finite gradients at step {self.global_step}, skipping. Bad params: {bad_params[:10]}")
                self.optimizer.zero_grad(set_to_none=True)
                self._n_skipped_nonfinite += 1
                epoch_skipped_nonfinite += 1
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get("grad_clip_norm", 10.0))
                self.optimizer.step()
                if self.global_step < self.warmup_steps:
                    self.warmup_scheduler.step()

            loss_value = float(loss.item())
            epoch_loss += loss_value
            self.global_step += 1

            max_steps = self.config.get("max_steps")
            hit_max_steps = bool(max_steps) and self.global_step >= max_steps

            batch_skipped = not is_finite
            if should_log or batch_skipped:
                extra = {
                    "energy_term": float(energy_term.item()),
                    "force_term": float(force_term.item()),
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "skipped": int(batch_skipped),
                }
                weight_buckets = None
                if diag_record is not None:
                    weight_buckets = diag_record["weight_buckets"]
                    extra.update({
                        "alpha": diag_record["alpha"], "damping_sum_min": diag_record["damping_sum_min"],
                        "Z_min": diag_record["Z_min"],
                        "ln_clamp_hits": diag_record["ln_clamp_hits"], "safe_inv_retries": diag_record["safe_inv_retries"],
                        "max_force_mag": diag_record["max_force_mag"], "max_abs_U": diag_record["max_abs_U"],
                        "min_abs_radial": diag_record["min_abs_radial"],
                        "eqln_deg_gt0_near_degenerate": diag_record["eqln_deg_gt0_near_degenerate"],
                        "g_a_norm_max": diag_record["g_a_norm_max"],
                    })
                self._log_csv_row(
                    "train_batch", epoch, batch_idx=batch_idx, loss=loss_value,
                    grad_norm=math.sqrt(total_norm_sq.item()), buckets=buckets, weight_buckets=weight_buckets, **extra,
                )
                if self.provenance is not None:
                    stats = self._last_batch_stats
                    self.provenance.log(self.global_step, epoch, batch_idx, stats.get("idx", []), stats.get("smiles", []))

            if hit_max_steps:
                logger.info(f"Reached --max-steps={max_steps}, stopping epoch {epoch} early at batch_idx={batch_idx}.")
                break

        avg_loss = epoch_loss / max(batch_idx + 1, 1)
        if epoch_skipped_nonfinite:
            logger.info(
                f"Epoch {epoch}: skipped {epoch_skipped_nonfinite} non-finite-gradient batches "
                f"out of {batch_idx + 1} (cumulative: {self._n_skipped_nonfinite})."
            )
        self._log_csv_row("train_epoch_avg", epoch, loss=avg_loss)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        n_steps = batch_idx + 1
        elapsed = time.time() - epoch_start
        logger.info(
            f"Epoch {epoch}: {elapsed:.2f}s total, {1000 * elapsed / n_steps:.1f}ms/step "
            f"({n_steps} steps)"
        )
        return avg_loss

    def validate(self, epoch: int) -> float:
        self.model.eval()
        val_start = time.time()
        val_loss = 0.0
        batch_times = []
        grad_ctx = torch.no_grad() if self.force_mode == "direct" else torch.enable_grad()
        with grad_ctx:
            for batch in self.val_loader:
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                batch_start = time.time()
                loss, _, _ = self._step(batch, use_compiled=False)
                val_loss += float(loss.item())
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                batch_times.append(time.time() - batch_start)
        avg = val_loss / max(len(self.val_loader), 1)
        self._log_csv_row("validation", epoch, loss=avg)
        if self.global_step >= self.warmup_steps:
            self.plateau_scheduler.step(avg)
        self.model.train()
        if self.diag:
            counters.snapshot_and_reset()
        elapsed = time.time() - val_start
        n_batches = max(len(self.val_loader), 1)
        logger.info(
            f"Validation (epoch {epoch}): {elapsed:.2f}s total, "
            f"{1000 * elapsed / n_batches:.1f}ms/batch ({n_batches} batches)"
        )
        if batch_times:
            sorted_times = sorted(batch_times)
            median = sorted_times[len(sorted_times) // 2]
            outliers = [t for t in batch_times if t > 3 * median]
            worst = sorted_times[-5:]
            logger.info(
                f"Validation per-batch breakdown: median={1000*median:.1f}ms  "
                f"min={1000*sorted_times[0]:.1f}ms  max={1000*sorted_times[-1]:.1f}ms  "
                f"{len(outliers)}/{n_batches} batches >3x median  "
                f"worst-5(ms)={[round(1000*t, 1) for t in worst]}"
            )
        return avg

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        ckpt_dir = self.config["ckpt_dir"]
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "warmup_scheduler_state_dict": self.warmup_scheduler.state_dict(),
            "plateau_scheduler_state_dict": self.plateau_scheduler.state_dict(),
            "val_loss": val_loss,
            "config": self.config,
        }
        path = os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch:04d}.pt")
        torch.save(ckpt, path)
        logger.info(f"Saved: {path}")
        if is_best:
            torch.save(ckpt, os.path.join(ckpt_dir, "best_checkpoint.pt"))
            logger.info("Saved best checkpoint.")

    def load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.warmup_scheduler.load_state_dict(ckpt["warmup_scheduler_state_dict"])
        self.plateau_scheduler.load_state_dict(ckpt["plateau_scheduler_state_dict"])
        self.global_step = ckpt.get("global_step", 0)
        epoch = ckpt.get("epoch", 0)
        logger.info(f"Resumed from {path}: global_step={self.global_step}, epoch={epoch}")
        logger.info(
            "Note: resume continues training from this point but does not replay the exact "
            "same batch order (DataLoader RandomSampler state isn't serialized)."
        )
        return epoch


def detect_device(requested: str | None) -> torch.device:
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


_DATASET_SPLIT_SEED = 0


def compute_split_sizes(n_total: int, val_fraction: float, test_fraction: float | None) -> tuple[int, int, int]:
    n_val = max(int(n_total * val_fraction), 1)
    n_test = max(int(n_total * test_fraction), 1) if test_fraction is not None else 0
    n_train = max(n_total - n_val - n_test, 1)
    return n_train, n_val, n_test


def _save_dataset_split(config: dict, dataset: "SPICEDataset", test_fraction: float, train_idx: list[int], val_idx: list[int], test_idx: list[int]) -> None:
    ckpt_dir = config.get("ckpt_dir", ".")
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, "dataset_split.pt")
    split = {
        "seed": _DATASET_SPLIT_SEED,
        "val_fraction": config["val_fraction"],
        "test_fraction": test_fraction,
        "hdf5_paths": config["hdf5_paths"],
        "subsets": config.get("subsets"),
        "box_padding": config["box_padding"],
        "edge_cutoff": config["edge_cutoff"],
        "n_total": len(dataset),
        "train_identities": [dataset.identity_for(i) for i in train_idx],
        "val_identities": [dataset.identity_for(i) for i in val_idx],
        "test_identities": [dataset.identity_for(i) for i in test_idx],
    }
    torch.save(split, path)
    logger.info(
        f"Saved deterministic dataset split -> {path} "
        f"(train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)})"
    )


def build_dataloaders(config: dict, device: torch.device | None = None) -> tuple[DataLoader, DataLoader, float, float, float]:
    dataset = SPICEDataset(
        hdf5_paths=config["hdf5_paths"],
        subsets=config.get("subsets"),
        exclude_solvated=True,
        box_padding=config["box_padding"],
        edge_cutoff=config["edge_cutoff"],
    )
    test_fraction = config.get("test_fraction")  # None for configs saved before this split existed
    n_train, n_val, n_test = compute_split_sizes(len(dataset), config["val_fraction"], test_fraction)

    if n_test > 0:
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(_DATASET_SPLIT_SEED),
        )
        _save_dataset_split(config, dataset, test_fraction, train_dataset.indices, val_dataset.indices, test_dataset.indices)
    else:
        train_dataset, val_dataset = random_split(
            dataset, [n_train, n_val], generator=torch.Generator().manual_seed(_DATASET_SPLIT_SEED),
        )

    energy_mean, energy_std, force_rms = dataset.energy_force_stats(train_dataset.indices)

    num_workers = config.get("num_workers", 0)
    pin_memory = device.type == "cuda" if device is not None else torch.cuda.is_available()
    loader_kwargs: dict = dict(
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=num_workers > 0,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = config.get("prefetch_factor", 2)

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True,
        collate_fn=spice_collate_fn, **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False,
        collate_fn=spice_collate_fn, **loader_kwargs,
    )
    return train_loader, val_loader, energy_mean, energy_std, force_rms


def run_training(config: dict, device: torch.device | None = None):
    device = device or detect_device(config.get("device"))
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    train_loader, val_loader, energy_mean, energy_std, force_rms = build_dataloaders(config, device=device)
    config["energy_mean"] = energy_mean
    config["energy_std"] = energy_std
    config["force_rms"] = force_rms
    logger.info(
        f"Dataset stats (train split): energy_mean={energy_mean:.4f} eV, "
        f"energy_std={energy_std:.4f} eV, force_rms={force_rms:.4f} eV/A"
    )

    model = SPICEEnergyForceModel(
        radial_net=RadialNetwork,
        num_atom_types=config["num_atom_types"],
        max_degree=config["max_degree"],
        num_layers=config["num_layers"],
        feature_dim=config["feature_dim"],
        hidden_dim=config["hidden_dim"],
        edge_cutoff=config["edge_cutoff"],
        n_max=config["n_max"],
        pw_e_cutoff=config.get("pw_e_cutoff"),
        n_master_nodes=config.get("n_master_nodes", 4),
        backbone=config.get("backbone", "lr_restructure"),
        num_fusion_heads=config.get("num_fusion_heads", 4),
        ablate_max_degree=config.get("ablate_max_degree"),
        grad_checkpoint=config.get("grad_checkpoint", False),
        num_broadcast_heads=config.get("num_broadcast_heads", 4),
        rbf_num_basis=config.get("rbf_num_basis", 12),
        rbf_min=config.get("rbf_min", 0.3),
        rbf_cutoff=config.get("rbf_cutoff", 25.0),
        sigma=config.get("sigma"),
        alpha_floor=config.get("alpha_floor", 1e-4),
        use_norms=config.get("use_norms", True),
        use_long_range=config.get("use_long_range", True),
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")

    trainer = SPICETrainer(model, train_loader, val_loader, config, device)

    start_epoch = 0
    if config.get("resume"):
        start_epoch = trainer.load_checkpoint(config["resume"]) + 1

    try:
        best_val_loss = float("inf")
        for epoch in range(start_epoch, config["n_epochs"]):
            trainer.train_epoch(epoch)
            if (epoch + 1) % config.get("val_every", 1) == 0:
                val_loss = trainer.validate(epoch)
                is_best = val_loss < best_val_loss
                best_val_loss = min(best_val_loss, val_loss)
                if (epoch + 1) % config.get("save_every", 1) == 0:
                    trainer.save_checkpoint(epoch, val_loss, is_best)
            max_steps = config.get("max_steps")
            if max_steps and trainer.global_step >= max_steps:
                logger.info(f"Reached --max-steps={max_steps} (global_step={trainer.global_step}), stopping training.")
                break
    finally:
        trainer.close()

    return trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5-paths", type=str, nargs="+", required=True, help="SPICE .hdf5 file(s)/dir(s)/glob(s)")
    parser.add_argument("--subsets", type=str, nargs="+", default=["pubchem"], help="Friendly or literal subset names; defaults to the SPICE PubChem subsets")
    parser.add_argument("--ckpt-dir", type=str, default="/home/ubuntu/metis-v1-storage/spice_stage1")
    parser.add_argument("--csv-path", type=str, default='/home/ubuntu/metis-v1-storage/spice_loss.csv', help="Loss/grad-norm CSV path (default: <ckpt-dir>/loss_log.csv); the log file is reserved for non-finite-gradient alerts only")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-epochs", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=None, help="Max steps for phase 2 experiment runs")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--warmup-epochs", type=int, default=100, help="Epochs of linear LR warmup (start_factor=0.01) before ReduceLROnPlateau takes over")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="ReduceLROnPlateau's LR floor")
    parser.add_argument("--plateau-factor", type=float, default=0.9, help="ReduceLROnPlateau's multiplicative LR reduction on a detected plateau")
    parser.add_argument("--plateau-patience", type=int, default=50, help="Validation events (not epochs -- see --val-every) with no improvement before ReduceLROnPlateau reduces LR")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--lambda1", type=float, default=1.0, help="Energy loss weight")
    parser.add_argument("--lambda2", type=float, default=1000.0, help="Force loss weight")
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--max-degree", type=int, default=2)
    parser.add_argument("--ablate-max-degree", type=int, default=None, choices=[0, 1], help="Feature-degree ablation study: zero out degree>l tensors at every point they're produced (seed features, each layer's SR/self output, master-fusion output), holding architecture and parameter count fixed at --max-degree=2. Omit for the full equivariant model.")
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--edge-cutoff", type=float, default=5.0)
    parser.add_argument("--box-padding", type=float, default=12.0, help="Vacuum spacing (Angstrom) around each molecule's auxiliary supercell")
    parser.add_argument("--grad-clip-norm", type=float, default=10.0)
    parser.add_argument("--grad-checkpoint", action="store_true", help="Wrap each EnergyLayer in torch.utils.checkpoint (use_reentrant=False) during training to cut activation memory at the same batch size, at the cost of recomputing each layer's forward during backward (~20-30%% slower per step). No numerical-precision tradeoff, unlike --bf16 (removed -- verified to corrupt forces on this model, see git history).")
    parser.add_argument("--n-max", type=int, default=1)
    parser.add_argument("--pw-e-cutoff", type=float, default=None, help="Long-range k-space energy cutoff (k_max = sqrt(2*e_cutoff)); caps |k| independent of box size. Not a reliable substitute for keeping --n-max small -- filtering high-k modes did not consistently fix instability at n_max=4 in testing")
    parser.add_argument("--num-atom-types", type=int, default=23, help="mappings.py padding + 22 element types (incl. Li)")
    parser.add_argument("--n-master-nodes", type=int, default=8, help="Number of virtual master nodes accumulating long-range SE(3)-invariant context, co-located at the molecular center of mass")
    parser.add_argument("--backbone", type=str, default="lr_restructure", choices=["lr_restructure", "production"], help="Energy+force backbone: 'lr_restructure' (default) is stage_1/lr_restructure's bounded-softmax, moving-master-node architecture that beat DimeNet/Nequip-Allegro on oversquashing and RSA on lr_sr; 'production' is the original stage_1/model.py EnergyModel, kept for comparison/rollback")
    parser.add_argument("--num-fusion-heads", type=int, default=4, help="[--backbone production only] Attention heads for the one-shot master-node-to-atom fusion cross-attention; must divide --feature-dim")
    parser.add_argument("--num-broadcast-heads", type=int, default=4, help="[--backbone lr_restructure only] Attention heads for the per-layer master-node<->atom broadcast attention; must divide --feature-dim")
    parser.add_argument("--rbf-num-basis", type=int, default=12, help="[--backbone lr_restructure only] Number of RBF basis functions for the per-master-node distance-weighted attention")
    parser.add_argument("--rbf-min", type=float, default=0.3, help="[--backbone lr_restructure only] Minimum RBF center spacing (Angstrom)")
    parser.add_argument("--rbf-cutoff", type=float, default=25.0, help="[--backbone lr_restructure only] Maximum RBF center distance (Angstrom)")
    parser.add_argument("--sigma", type=float, default=None, help="[--backbone lr_restructure only] Fixed physical width for the Ewald damping envelope (alpha = 1/(2*sigma^2)); default None leaves alpha a learned parameter, matching production EnergyModel's behavior -- SPICE molecules have no single canonical sigma the way the synthetic lr_sr Coulomb benchmark does")
    parser.add_argument("--alpha-floor", type=float, default=1e-4, help="[--backbone lr_restructure only] Floor added to the learned Ewald damping alpha (softplus(alpha_param) + alpha_floor); ignored when --sigma is set")
    parser.add_argument("--use-norms", action=argparse.BooleanOptionalAction, default=True, help="[--backbone lr_restructure only] Use RMSNorm throughout the backbone; --no-use-norms strips them (ablation)")
    parser.add_argument("--use-long-range", action=argparse.BooleanOptionalAction, default=True, help="[--backbone lr_restructure only] Run the master-node/long-range channel (aggregator+update+broadcast); --no-use-long-range skips it entirely (not constructed, not called) for a local-only ablation")
    parser.add_argument("--force-mode", type=str, default="conservative", choices=["conservative", "direct", "both"], help="'conservative' (default): forces via autograd.grad on positions, unchanged. 'direct': forces from SPICEEnergyForceModel's direct force head, no autograd through positions at all -- the torch.compile-friendly path, intended for pretraining. 'both': one forward pass computes direct forces and energy together, one autograd.grad call (reusing that graph) also recovers conservative forces -- PET-M-style joint training; see --lambda2-conservative/--lambda-force-match.")
    parser.add_argument("--lambda2-conservative", type=float, default=None, help="[force_mode=both only] Loss weight for the conservative force term against the DFT-labeled target. Defaults to None (term disabled) -- pass a value to enable PET-M-style joint training of both heads against the ground truth.")
    parser.add_argument("--lambda-force-match", type=float, default=0.0, help="[force_mode=both only] Loss weight for ||F_direct - F_conservative||^2, encouraging the direct head to approximate the true energy gradient (Bigi et al. 2412.11569 Sec 3.5's recommended cheap alternative to a direct Jacobian-symmetry penalty). Default 0 (off).")
    parser.add_argument("--torch-compile", action=argparse.BooleanOptionalAction, default=False, help="Wrap the model in torch.compile for the forward pass. Only has an effect when --force-mode direct (the only mode without the create_graph=True double-backward pattern that breaks compile); a warning is logged and this is ignored otherwise.")
    parser.add_argument("--compile-mode", type=str, default=None, choices=[None, "default", "reduce-overhead", "max-autotune"], help="torch.compile mode, passed through when --torch-compile is set.")
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--test-fraction", type=float, default=0.05, help="Held-out test split, carved out alongside val (never trained or validated against) -- its identity is persisted to <ckpt-dir>/dataset_split.pt for later use by e.g. diagnostics/test_set_confidence_interval.py")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2, help="Batches prefetched per DataLoader worker; only applies when --num-workers > 0")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume model/optimizer/scheduler/global_step from; continues training, does not replay the exact same batch order")
    parser.add_argument("--diagnostics", action="store_true", help="Attach forward+backward hooks, per-module weight/grad-norm and risk-scalar CSV logging, batch provenance, drift detection, and ring-buffer checkpointing")
    parser.add_argument("--diag-skip-hooks", action="store_true", help="Only with --diagnostics: skip attaching the forward/backward magnitude+gradient hooks (attach_hooks/attach_grad_hooks), isolating the per-step named_parameters() bucket/risk-scalar loop's cost for throughput A/B testing")
    parser.add_argument("--diag-skip-buckets", action="store_true", help="Only with --diagnostics: skip the per-step _grad_norm_buckets/_weight_norm_buckets/_risk_scalars computation, isolating the forward/backward hooks' cost for throughput A/B testing. Drift detection and bucket/risk CSV columns will read as zero/empty while this is set.")
    parser.add_argument("--diag-explode-threshold", type=float, default=50.0, help="Per-bucket grad-norm value that immediately counts as an explosion, regardless of drift history")
    parser.add_argument("--diag-ema-decay", type=float, default=0.98, help="EMA decay for the rolling drift detector")
    parser.add_argument("--diag-drift-k", type=float, default=6.0, help="Number of EMA-stddevs above the rolling mean that counts as a drift-triggered explosion")
    parser.add_argument("--diag-ring-k", type=int, default=50, help="Number of recent steps of telemetry to keep and dump when an explosion/drift event triggers")
    parser.add_argument("--diag-ckpt-every", type=int, default=200, help="Save a ring-buffer checkpoint every N steps when --diagnostics is on")
    parser.add_argument("--diag-ckpt-keep", type=int, default=3, help="Number of ring-buffer checkpoint slots to round-robin over")
    parser.add_argument("--diag-dump-dir", type=str, default=None, help="Directory for diagnostic dumps (default: <ckpt-dir>/diag_dumps)")
    args = parser.parse_args()

    config = vars(args)
    config["hdf5_paths"] = args.hdf5_paths
    run_training(config)


if __name__ == "__main__":
    main()
