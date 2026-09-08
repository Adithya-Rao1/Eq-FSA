from __future__ import annotations

import argparse
import sys

import torch
import torch.nn.functional as F

from stage_1.attention import MasterNodeAttention
from stage_1.experiments.neurreps.lr_sr.invariant_energy_data import generate_batch, make_wavevector_grid
from stage_1.experiments.neurreps.lr_sr.invariant_energy_model import EqFSAEncoder, RSAEncoder
from stage_1.experiments.neurreps.lr_sr.rsa_attention import RSAAttention
from stage_1.experiments.neurreps.lr_sr.train_invariant_energy import build_model, build_optimizer, detect_device
from stage_1.model import masked_mean, masked_weighted_com
from utils.pw_bases import make_wavevectors

def cross_example_variation(t: torch.Tensor) -> dict:
    per_example = t.reshape(t.shape[0], -1).norm(dim=1)
    mean_, std_ = per_example.mean(), per_example.std(unbiased=False)
    return {
        "mean_norm": mean_.item(), "std_norm": std_.item(),
        "relative_cv": (std_ / mean_.clamp(min=1e-12)).item(), "n": t.shape[0],
    }


def mean_pairwise_cosine_similarity(t: torch.Tensor) -> float:
    flat = t.reshape(t.shape[0], -1)
    B = flat.shape[0]
    if B < 2:
        return float("nan")
    normed = F.normalize(flat, dim=-1, eps=1e-12)
    sim = normed @ normed.T
    off_diag_sum = sim.sum() - sim.diagonal().sum()
    return (off_diag_sum / (B * (B - 1))).item()

def instrumented_master_node_attn_forward(
    module: MasterNodeAttention, g_a: torch.Tensor, f_in0: torch.Tensor, x: torch.Tensor,
    x_com: torch.Tensor, node_mask: torch.Tensor, V: torch.Tensor,
    normalize_z_by_n_real: bool = False,
) -> tuple[torch.Tensor, dict]:
    diag: dict = {}
    B, N_a, C, _ = g_a.shape
    m = node_mask.to(x.dtype)

    Q_pre_norm, K_pre_norm = module.qk_pre_norm(g_a, f_in0)
    Q_post_norm = module.w_vq_norm(Q_pre_norm)
    Q = Q_post_norm.squeeze(-1)
    diag["Q_pre_norm"], diag["Q_post_norm"] = Q_pre_norm.detach(), Q_post_norm.detach()

    K_post_norm = module.w_vk_norm(K_pre_norm)
    K = K_post_norm.squeeze(-1)
    diag["K_pre_norm"], diag["K_post_norm"] = K_pre_norm.detach(), K_post_norm.detach()

    diag["w_vq_norm_gamma"] = module.w_vq_norm.gamma.detach().clone()
    diag["w_vk_norm_gamma"] = module.w_vk_norm.gamma.detach().clone()

    with torch.no_grad():
        w_dbg = torch.einsum("bac,bgc->bag", F.elu(Q) + 1, (F.elu(K) + 1) * m.unsqueeze(-1))
        w_dbg_norm = w_dbg / w_dbg.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        diag["ipr"] = 1.0 / w_dbg_norm.pow(2).sum(dim=-1).clamp(min=1e-12)

    phi_Q = F.elu(Q) + 1
    phi_K = (F.elu(K) + 1) * m.unsqueeze(-1)
    phi_K_sum = phi_K.sum(dim=1)
    Z_raw = torch.einsum("bac,bc->ba", phi_Q, phi_K_sum)
    if normalize_z_by_n_real:
        n_real = node_mask.sum(dim=1, keepdim=True).clamp(min=1).to(Z_raw.dtype)
        Z_raw = Z_raw / n_real
    Z = Z_raw.clamp(min=1e-8)
    diag["Z"] = Z.detach()

    kvecs, indices, mode_mask = make_wavevectors(V, module.n_max, module.pw_e_cutoff)
    mode_mask_f = mode_mask.to(x.dtype)
    if mode_mask_f.dim() == 1:
        mode_mask_f = mode_mask_f.unsqueeze(0).expand(B, -1)

    alpha = F.softplus(module.alpha) + 1e-4
    diag["alpha_raw"] = module.alpha.detach().clone()
    diag["alpha_softplus"] = alpha.detach().clone()
    k2 = (kvecs ** 2).sum(dim=-1)
    damping = torch.exp(-k2 / (4.0 * alpha)) * mode_mask_f
    damping_sum = damping.sum(dim=-1)
    diag["damping_sum"] = damping_sum.detach()
    is_dc = indices.abs().sum(dim=-1) == 0
    dc_contrib = damping[:, is_dc].sum(dim=-1)
    diag["dc_mode_fraction"] = (dc_contrib / damping_sum.clamp(min=1e-12)).detach()

    phase_g = -torch.einsum("bgi,bmi->bgm", x, kvecs)
    cos_g = torch.cos(phase_g) * mode_mask_f.unsqueeze(1)
    sin_g = torch.sin(phase_g) * mode_mask_f.unsqueeze(1)

    V_proj0 = module.W_V0(f_in0.squeeze(-1)).unsqueeze(-1)

    raw_S_re0 = torch.einsum("bgc,bgnd,bgm->bmcnd", phi_K, V_proj0, cos_g)
    raw_S_im0 = torch.einsum("bgc,bgnd,bgm->bmcnd", phi_K, V_proj0, sin_g)
    raw_S_ri = torch.cat([raw_S_re0, raw_S_im0], dim=-2)
    diag["raw_S_ri"] = raw_S_ri.detach()
    raw_S_ri_mag = raw_S_ri.norm(dim=-2)

    S_ri = module.s_ri_norm(raw_S_ri.transpose(-1, -2)).transpose(-1, -2)
    diag["S_ri"] = S_ri.detach()
    S_re0, S_im0 = S_ri.split(module.feature_dim, dim=-2)

    dist_to_com = (x - x_com.unsqueeze(1)).norm(dim=-1, keepdim=True)
    dist_feat = module.spatial_radial(dist_to_com)
    spatial_q = module.spatial_query_proj(g_a.squeeze(-1))
    spatial_logit = torch.einsum("bah,bgh->bag", spatial_q, dist_feat) / (dist_feat.shape[-1] ** 0.5)
    spatial_logit = spatial_logit.masked_fill(~m.bool().unsqueeze(1), -1e9)
    spatial_w = torch.softmax(spatial_logit, dim=-1)
    x_com_use = torch.einsum("bag,bgi->bai", spatial_w, x)
    diag["x_com_per_node"] = x_com_use.detach()
    diag["spatial_w_entropy"] = (-(spatial_w.clamp(min=1e-12) * spatial_w.clamp(min=1e-12).log()).sum(dim=-1))

    phase_com = torch.einsum("bai,bmi->bam", x_com_use, kvecs)
    cos_com = torch.cos(phase_com) * damping.unsqueeze(1)
    sin_com = torch.sin(phase_com) * damping.unsqueeze(1)
    diag["x_com"] = x_com.detach()

    QS_re0 = torch.einsum("bac,bmcnd->bamnd", phi_Q, S_re0)
    QS_im0 = torch.einsum("bac,bmcnd->bamnd", phi_Q, S_im0)
    diag["QS_re0"] = QS_re0.detach()

    cos_b = cos_com.unsqueeze(-1).unsqueeze(-1)
    sin_b = sin_com.unsqueeze(-1).unsqueeze(-1)
    lr_master_pre_div = (QS_re0 * cos_b - QS_im0 * sin_b).sum(dim=2)
    diag["lr_master_pre_div"] = lr_master_pre_div.detach()

    lr_master = lr_master_pre_div / Z.view(B, N_a, 1, 1)
    lr_master = lr_master / damping_sum.view(B, 1, 1, 1)
    diag["lr_master"] = lr_master.detach()

    Q_mag = torch.einsum("bac,bmcd->bamd", phi_Q, raw_S_ri_mag)
    mag_pooled = torch.einsum("bamd,bm->bad", Q_mag, damping)
    mag_pooled = mag_pooled / Z.view(B, N_a, 1)
    mag_pooled = mag_pooled / damping_sum.view(B, 1, 1)
    scale_feat = torch.log1p(mag_pooled.squeeze(-1))
    diag["scale_feat"] = scale_feat.detach()

    return lr_master, diag


def instrumented_rsa_forward(
    module: RSAAttention, f_in0: torch.Tensor, x: torch.Tensor, node_mask: torch.Tensor, V: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    diag: dict = {}
    B, N, C, _ = f_in0.shape
    m = node_mask.to(x.dtype)
    h = f_in0.squeeze(-1)

    q, k, v = module.qkv(h).chunk(3, dim=-1)
    q = module.act(q) * m.unsqueeze(-1)
    k = module.act(k) * m.unsqueeze(-1)

    kvecs, indices, mode_mask = make_wavevectors(V, module.n_max, e_cutoff=None)
    mode_mask_f = mode_mask.to(x.dtype)
    if mode_mask_f.dim() == 1:
        mode_mask_f = mode_mask_f.unsqueeze(0).expand(B, -1)
    nonzero = (indices.abs().sum(dim=-1) > 0).unsqueeze(0).to(x.device)
    active = (nonzero.expand(B, -1) & mode_mask_f.bool()).to(x.dtype)

    k2 = (kvecs ** 2).sum(dim=-1)
    w_k = torch.exp(-k2 * (module.sigma ** 2) / 2.0) / k2.clamp(min=1e-12) * active
    w_k = w_k / w_k.sum(dim=-1, keepdim=True).clamp(min=1e-12)

    phase = torch.einsum("bni,bmi->bnm", x, kvecs)
    cos, sin = phase.cos(), phase.sin()

    q_rot = module._rope(q, cos, sin) * module.scale_q
    k_rot = module._rope(k, cos, sin)

    kv = torch.einsum("bnmc,bnd,bn->bmcd", k_rot, v, m)
    half = kv.shape[-2] // 2
    diag["S_re"], diag["S_im"] = kv[..., :half, :].detach(), kv[..., half:, :].detach()

    beta = torch.einsum("bnmc,bmcd->bnmd", q_rot, kv)
    diag["out_pre_div"] = beta.detach()

    update = torch.einsum("bnmd,bm->bnd", beta, w_k) * m.unsqueeze(-1)
    update = module.out_norm(update) * m.unsqueeze(-1)
    diag["out"] = update.detach()

    return update.unsqueeze(-1), diag

def probe_eqfsa(model: EqFSAEncoder, charges, positions, mask, V) -> tuple[torch.Tensor, dict]:
    B = charges.shape[0]
    f_in0 = model.embed(charges)
    masses = mask.float()
    x_com, _ = masked_weighted_com(positions, masses, mask)
    if model.use_input_dependent_g_a:
        dist_to_com = (positions - x_com.unsqueeze(1)).norm(dim=-1, keepdim=True)
        pool_in_feat = f_in0.squeeze(-1) + model.dist_to_com_embed(dist_to_com)
        pooled_in = model.master_pool_ffn(masked_mean(pool_in_feat, mask))
        g_a = model.master_expand(pooled_in).view(B, model.num_master_nodes, -1).unsqueeze(-1)
        g_a = model.master_expand_norm(g_a)
    else:
        g_a = model.g_a_init.unsqueeze(0).expand(B, -1, -1).unsqueeze(-1).contiguous()

    lr_master, diag = instrumented_master_node_attn_forward(
        model.master_attn, g_a, f_in0, positions, x_com, mask, V,
        normalize_z_by_n_real=getattr(model, "use_normalize_z_by_n_real", False),
    )
    master_out = model.master_attn_norm(lr_master.squeeze(-1))
    g_a_const = g_a.squeeze(-1)
    g_a_out = g_a_const + master_out
    decoder_in = torch.cat([g_a_out.mean(dim=1), diag["scale_feat"].mean(dim=1, keepdim=True)], dim=-1)

    diag["decoder_in"] = decoder_in.detach()
    diag["master_out_normed"] = master_out.detach()
    master_norm = master_out.reshape(B, -1).norm(dim=1)
    g_a_norm = g_a_const.reshape(B, -1).norm(dim=1)
    diag["master_vs_ga_ratio"] = (master_norm / g_a_norm.clamp(min=1e-12)).detach()
    diag["rescale_log_scale"] = model.master_attn_norm.log_scale.detach().clone()

    pred = model.decoder(decoder_in)
    return pred, diag


def probe_rsa(model: RSAEncoder, charges, positions, mask, V) -> tuple[torch.Tensor, dict]:
    f_in0 = model.embed(charges)
    rsa_out, diag = instrumented_rsa_forward(model.rsa_attn, f_in0, positions, mask, V)
    per_atom = rsa_out.squeeze(-1)
    pooled = (per_atom * mask.unsqueeze(-1)).sum(dim=1)
    diag["decoder_in"] = pooled.detach()
    pred = model.decoder(pooled)
    return pred, diag

def train_short(encoder: str, n_atoms: int, config: dict, seed: int, train_steps: int) -> torch.nn.Module:
    device = config["_device"]
    generator = torch.Generator().manual_seed(seed)
    kvecs = make_wavevector_grid(config["n_max"], box=config["box"]).to(device)
    sigmas = torch.tensor(config["sigmas"], device=device)
    num_outputs = sigmas.shape[0]

    torch.manual_seed(seed)
    model = build_model(encoder, num_outputs, config).to(device)
    optimizer = build_optimizer(model, config["lr"], config.get("qk_gamma_wd", 0.0))

    with torch.no_grad():
        _, _, _, _, E_probe = generate_batch(
            config["eval_batches"] * config["batch_size"], n_atoms, kvecs.cpu(), sigmas.cpu(),
            box=config["box"], generator=torch.Generator().manual_seed(seed + 999),
        )
        target_std = E_probe.var(dim=0).sqrt().clamp(min=1e-12).to(device)
        target_mean = E_probe.mean(dim=0).to(device)
        model.decoder.net[-1].weight.zero_()
        model.decoder.net[-1].bias.copy_(target_mean)

    model.train()
    for _ in range(train_steps):
        charges, positions, V, mask, E_LR = generate_batch(
            config["batch_size"], n_atoms, kvecs.cpu(), sigmas.cpu(), box=config["box"], generator=generator,
        )
        charges, positions, mask, V, E_LR = (
            charges.to(device), positions.to(device), mask.to(device), V.to(device), E_LR.to(device),
        )
        optimizer.zero_grad()
        pred = model(charges, positions, mask, V)
        loss = F.mse_loss((pred - target_mean) / target_std, (E_LR - target_mean) / target_std)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def build_probe_batch(n_atoms: int, config: dict, seed: int):
    kvecs = make_wavevector_grid(config["n_max"], box=config["box"])
    sigmas = torch.tensor(config["sigmas"])
    charges, positions, V, mask, _ = generate_batch(
        config["batch_size"], n_atoms, kvecs, sigmas, box=config["box"],
        generator=torch.Generator().manual_seed(seed + 54321),
    )
    return charges, positions, V, mask

EQFSA_STAGES = [
    ("0  decoder_in  [PRIMARY go/no-go]", "decoder_in"),
    ("0a scale_feat (parallel channel -- tests #2 fix)", "scale_feat"),
    ("0b master_attn_out (raw, pre-norm)", "lr_master"),
    ("0c master_attn_out (post master_attn_norm)", "master_out_normed"),
    ("2  lr_master_pre_div (before /Z,/damping_sum)", "lr_master_pre_div"),
    ("3  QS_re0 (post phi_Q contraction, pre COM-phase)", "QS_re0"),
    ("4  S_ri (post s_ri_norm)  [tests #2]", "S_ri"),
    ("5  raw_S_ri (pre s_ri_norm -- untouched baseline)", "raw_S_ri"),
    ("6  Q_pre_norm (pre EquivariantRMSNorm)  [tests #3]", "Q_pre_norm"),
    ("6  Q_post_norm (post EquivariantRMSNorm)", "Q_post_norm"),
    ("6  K_pre_norm (pre EquivariantRMSNorm)", "K_pre_norm"),
    ("6  K_post_norm (post EquivariantRMSNorm)", "K_post_norm"),
    ("   x_com  [tests #6]", "x_com"),
    ("   x_com_per_node (per-master-node weighted COM)  [tests architecture fix]", "x_com_per_node"),
]

RSA_STAGES = [
    ("0  decoder_in  [REFERENCE]", "decoder_in"),
    ("   out (final rsa_attn output, analog of 0b)", "out"),
    ("   out_pre_div (analog of stage 2)", "out_pre_div"),
    ("   S_re (analog of stage 4/5 -- no forced-magnitude norm exists here)", "S_re"),
    ("   S_im (analog of stage 4/5 -- no forced-magnitude norm exists here)", "S_im"),
]


def print_stage_table(label: str, diag: dict, stages: list[tuple[str, str]]) -> None:
    print(f"  -- {label} --")
    for name, key in stages:
        if key not in diag:
            continue
        s = cross_example_variation(diag[key])
        print(f"    {name:52s} mean_norm={s['mean_norm']:11.4g}  std_norm={s['std_norm']:11.4g}  "
              f"relative_cv={s['relative_cv']:.4f}")


def print_scalar_diagnostics(diag: dict) -> None:
    if "Z" in diag:
        z = diag["Z"]
        print(f"    Z: within-batch mean={z.mean().item():.6g} std={z.std().item():.6g}  "
              f"(expect std~0 -- batch-constant per hypothesis #5)")
    if "damping_sum" in diag:
        ds = diag["damping_sum"]
        print(f"    damping_sum: within-batch mean={ds.mean().item():.6g} std={ds.std().item():.6g}  "
              f"(expect std~0 -- never touches atom data)")
    if "alpha_raw" in diag:
        print(f"    alpha (raw param)={diag['alpha_raw'].item():.6g}  "
              f"softplus(alpha)+1e-4={diag['alpha_softplus'].item():.6g}")
    if "dc_mode_fraction" in diag:
        dcf = diag["dc_mode_fraction"]
        print(f"    k=0 (DC, position-blind) mode's fraction of damping_sum: "
              f"mean={dcf.mean().item():.4f} std={dcf.std().item():.4f}  [tests #4]")
    if "master_vs_ga_ratio" in diag:
        r = diag["master_vs_ga_ratio"]
        print(f"    ||master_attn_out|| / ||g_a_init|| per example: "
              f"mean={r.mean().item():.4f} std={r.std().item():.4f}  [tests #1 -- residual dominance]")
    if "rescale_log_scale" in diag:
        ls = diag["rescale_log_scale"]
        print(f"    master_attn_norm (_LearnableRescale) exp(log_scale): "
              f"mean={ls.exp().mean().item():.4f} min={ls.exp().min().item():.4f} "
              f"max={ls.exp().max().item():.4f}  [init=500 -- shrinking this dilutes the g_a_init "
              f"residual's relative weight in g_a_out, damping decoder_in's cv growth]")
    if "w_vq_norm_gamma" in diag:
        print(f"    ||w_vq_norm.gamma||={diag['w_vq_norm_gamma'].norm().item():.4f}  "
              f"||w_vk_norm.gamma||={diag['w_vk_norm_gamma'].norm().item():.4f}  "
              f"[root-cause metric -- unconstrained gain, no weight decay unless --qk-gamma-wd>0]")
    if "ipr" in diag:
        ipr = diag["ipr"]
        print(f"    master-node attention IPR (of N real atoms): mean={ipr.mean().item():.2f} "
              f"std={ipr.std().item():.2f}  [lower = more concentrated attention -- "
              f"the mechanism linking gamma growth to decoder_in cv shrinkage]")
    if "spatial_w_entropy" in diag:
        se = diag["spatial_w_entropy"]
        print(f"    per_node_com spatial_w entropy (per master node): mean={se.mean().item():.3f} "
              f"std={se.std().item():.3f}  [lower = more spatially concentrated per node -- "
              f"tests whether the new position-aware weighting actually differentiates atoms, "
              f"unlike phi_K's charge-only IPR above]")


def print_direction_check(diag: dict) -> None:
    if "raw_S_ri" in diag and "S_ri" in diag:
        raw_cos = mean_pairwise_cosine_similarity(diag["raw_S_ri"])
        post_cos = mean_pairwise_cosine_similarity(diag["S_ri"])
        print(f"    mean pairwise cosine similarity across examples: "
              f"raw_S_ri={raw_cos:.4f}  S_ri (post s_ri_norm)={post_cos:.4f}")
        print("      (near 1.0 = examples point in nearly the same direction -- barely "
              "distinguishable even in 'direction', not just magnitude)  [decisive test of #2]")
    if "decoder_in" in diag:
        decoder_in_cos = mean_pairwise_cosine_similarity(diag["decoder_in"])
        print(f"    mean pairwise cosine similarity across examples: decoder_in={decoder_in_cos:.4f}")
        print("      (relative_cv only measures norm -- master_attn_norm now correctly strips "
              "magnitude variation by design, per RMSNorm's definition, so this direction check is "
              "the real signal-preservation test for decoder_in, not relative_cv alone)")

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-values", type=int, nargs="+", default=[32, 512])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-steps-short", type=int, default=300)
    parser.add_argument("--skip-training-probe", action="store_true",
                         help="Frozen-init probe only, skip the short-training condition (fast iteration).")
    parser.add_argument("--num-master-nodes", type=int, default=4)
    parser.add_argument("--rsa-sigma", type=float, default=5.0)
    parser.add_argument("--n-max", type=int, default=2)
    parser.add_argument("--sigmas", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0, 16.0])
    parser.add_argument("--box", type=float, default=20.0)
    parser.add_argument("--feature-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--qk-gamma-wd", type=float, default=0.0)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    config = vars(args)
    config["_device"] = detect_device(config["device"])
    print(f"Using device: {config['_device']}", flush=True)

    conditions = [(0, "frozen")]
    if not config["skip_training_probe"]:
        conditions.append((config["train_steps_short"], "trained"))

    summary_rows = []
    for n_atoms in config["n_values"]:
        for seed in config["seeds"]:
            probe_charges, probe_positions, probe_V, probe_mask = build_probe_batch(n_atoms, config, seed)
            device = config["_device"]
            pc, pp, pV, pm = (
                probe_charges.to(device), probe_positions.to(device),
                probe_V.to(device), probe_mask.to(device),
            )

            for train_steps, condition in conditions:
                print(f"\n=== N={n_atoms} seed={seed} condition={condition} "
                      f"(train_steps={train_steps}) ===", flush=True)

                eqfsa_model = train_short("eqfsa", n_atoms, config, seed, train_steps)
                rsa_model = train_short("rsa", n_atoms, config, seed, train_steps)

                with torch.no_grad():
                    _, eqfsa_diag = probe_eqfsa(eqfsa_model, pc, pp, pm, pV)
                    _, rsa_diag = probe_rsa(rsa_model, pc, pp, pm, pV)

                print_stage_table("EqFSA", eqfsa_diag, EQFSA_STAGES)
                print_scalar_diagnostics(eqfsa_diag)
                print_direction_check(eqfsa_diag)
                print_stage_table("RSA (reference)", rsa_diag, RSA_STAGES)

                eqfsa_cv0 = cross_example_variation(eqfsa_diag["decoder_in"])["relative_cv"]
                rsa_cv0 = cross_example_variation(rsa_diag["decoder_in"])["relative_cv"]
                ratio = eqfsa_cv0 / max(rsa_cv0, 1e-12)
                print(f"  >>> decoder-input relative_cv: eqfsa={eqfsa_cv0:.4f}  rsa={rsa_cv0:.4f}  "
                      f"ratio={ratio:.4f}", flush=True)
                summary_rows.append((n_atoms, seed, condition, eqfsa_cv0, rsa_cv0, ratio))

    print("\n=== SUMMARY: decoder-input relative_cv (eqfsa vs rsa) ===")
    print(f"  {'N':>6} {'seed':>6} {'condition':>10} {'eqfsa_cv':>10} {'rsa_cv':>10} {'ratio':>10}")
    for n_atoms, seed, condition, eqfsa_cv0, rsa_cv0, ratio in summary_rows:
        flag = "  <-- EqFSA's decoder input barely varies across examples" if ratio < 0.05 else ""
        print(f"  {n_atoms:6d} {seed:6d} {condition:>10} {eqfsa_cv0:10.4f} {rsa_cv0:10.4f} "
              f"{ratio:10.4f}{flag}")


if __name__ == "__main__":
    sys.exit(main())
