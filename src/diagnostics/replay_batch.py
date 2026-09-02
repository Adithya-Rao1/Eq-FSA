import argparse
import sys

import torch

from stage_1.model import EnergyLayer, seed_equivariant_features, masked_weighted_com
from stage_1.train_spice import SPICEEnergyForceModel, compute_energy_and_forces, spice_loss
from diagnostics.live_hook_training import attach_hooks, attach_grad_hooks, format_hook_report
from utils.se3_utils import RadialNetwork, EquivariantRMSNorm
from utils.spherical_harm import get_spherical_harmonics
from utils.pw_bases import minimum_image
from utils.calc_props import _MOL_FEAT_NAMES


def rebuild_model(saved_args: dict, device: torch.device, dtype: torch.dtype) -> SPICEEnergyForceModel:
    model = SPICEEnergyForceModel(
        radial_net=RadialNetwork,
        num_atom_types=saved_args["num_atom_types"],
        max_degree=saved_args["max_degree"],
        num_layers=saved_args["num_layers"],
        feature_dim=saved_args["feature_dim"],
        hidden_dim=saved_args["hidden_dim"],
        edge_cutoff=saved_args["edge_cutoff"],
        n_max=saved_args["n_max"],
        n_master_nodes=saved_args["n_master_nodes"],
        num_fusion_heads=saved_args["num_fusion_heads"],
    ).to(device=device, dtype=dtype)
    return model


def load_state_dict_allow_new_params(model: SPICEEnergyForceModel, state_dict: dict):
    result = model.load_state_dict(state_dict, strict=False)
    unexpected_missing = [k for k in result.missing_keys if not k.endswith(".alpha")]
    if unexpected_missing:
        raise RuntimeError(f"Unexpected missing keys beyond the new alpha params: {unexpected_missing}")
    if result.unexpected_keys:
        raise RuntimeError(f"Unexpected keys in checkpoint not present in current model: {result.unexpected_keys}")
    print(f"  Loaded checkpoint with strict=False; new params left at init: {result.missing_keys}")


def describe_geometry(batch: dict):
    x = batch["all_solute_x"]
    mask = batch["all_solute_mask"].bool()
    masses = batch["all_solute_masses"]
    types_ = batch["all_solute_types"]
    V = batch["V"]
    B, N, _ = x.shape

    print("\n--- Geometry characterization ---")
    x_wrapped = minimum_image(x, V)
    com, _ = masked_weighted_com(x_wrapped, masses, mask)
    r_true = (x_wrapped - com.unsqueeze(1)).norm(dim=-1)

    for b in range(B):
        m = mask[b]
        n_real = int(m.sum().item())
        r_b = r_true[b][m]
        print(f"  molecule {b}: n_real_atoms={n_real}  r_true(range from COM)=[{r_b.min().item():.4f}, {r_b.max().item():.4f}]"
              f"  atom_types_present={sorted(set(types_[b][m].tolist()))}")
        if n_real <= 2:
            print(f"    >> degenerate/near-degenerate molecule size (n_real_atoms={n_real})")

    x_i = x_wrapped.unsqueeze(2).expand(B, N, N, 3)
    x_j = x_wrapped.unsqueeze(1).expand(B, N, N, 3)
    dist = (x_j - x_i).norm(dim=-1)
    pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1) & ~torch.eye(N, dtype=torch.bool, device=x.device).unsqueeze(0)
    dist_masked = torch.where(pair_mask, dist, torch.full_like(dist, float("inf")))
    min_dist, min_idx = dist_masked.view(B, -1).min(dim=-1)
    for b in range(B):
        i, j = divmod(min_idx[b].item(), N)
        print(f"  molecule {b}: closest real-real contact = {min_dist[b].item():.4f} A between atoms ({i},{j})"
              f" [types {types_[b,i].item()},{types_[b,j].item()}]")


def report_full_gradient_breakdown(model: SPICEEnergyForceModel, batch: dict, config: dict):
    print("\n--- Full forward+backward replay with current hook coverage ---")
    model.train()
    mag_tracker = attach_hooks(model)
    grad_tracker = attach_grad_hooks(model)
    try:
        U, F = compute_energy_and_forces(
            model, batch["all_solute_x"], batch["all_solute_types"], batch["all_solute_masses"],
            batch["all_solute_mask"], batch["V"], batch["mol_props"], batch["all_solute_edge_feats"],
        )
        energy_mean = config["energy_mean"]
        energy_std = config["energy_std"]
        force_rms = config["force_rms"]
        U_n = (U - energy_mean) / energy_std
        U_target_n = (batch["energy"] - energy_mean) / energy_std
        F_n = F / force_rms
        F_target_n = batch["solute_forces"] / force_rms
        loss, energy_term, force_term = spice_loss(
            U_n, U_target_n, F_n, F_target_n, batch["all_solute_mask"],
            config["lambda1"], config["lambda2"],
        )
        loss.backward()
        print(f"  loss={loss.item():.6g}  energy_term={energy_term.item():.6g}  force_term={force_term.item():.6g}")
        print("\n  Forward-activation magnitudes (top 20):")
        print(mag_tracker.report(top_k=20))
        print("\n  Backward-activation-gradient norms (top 20 -- look here for the exploding module):")
        print(grad_tracker.report(top_k=20))
    finally:
        mag_tracker.remove()
        grad_tracker.remove()
        model.zero_grad(set_to_none=True)


def check_mol_props_raw(batch: dict, model: SPICEEnergyForceModel):
    print("\n--- Raw mol_props / mol_prop_emb weight check (before any model computation) ---")
    mol_props = batch["mol_props"]
    B = mol_props.shape[0]

    bad_mask = ~torch.isfinite(mol_props)
    n_bad = bad_mask.sum().item()
    if n_bad == 0:
        print(f"  mol_props: all finite. shape={tuple(mol_props.shape)}, "
              f"range=[{mol_props.min().item():.6g}, {mol_props.max().item():.6g}]")
    else:
        print(f"  mol_props: {n_bad} NON-FINITE entries out of {mol_props.numel()}!")
        bad_b, bad_f = bad_mask.nonzero(as_tuple=True)
        seen = set()
        for b, f in zip(bad_b.tolist(), bad_f.tolist()):
            if len(seen) >= 30:
                print(f"    ... ({n_bad - len(seen)} more, truncated)")
                break
            key = (b, f)
            if key in seen:
                continue
            seen.add(key)
            fname = _MOL_FEAT_NAMES[f] if f < len(_MOL_FEAT_NAMES) else f"<idx {f}>"
            print(f"    molecule {b}, feature[{f}]={fname!r}: value={mol_props[b, f].item()}")

    weight_reports = []
    for name, p in model.energy_model.mol_prop_emb.named_parameters():
        finite = torch.isfinite(p).all().item()
        weight_reports.append((name, finite, p.detach().abs().max().item() if finite else float("nan")))
    any_bad_weight = any(not finite for _, finite, _ in weight_reports)
    if any_bad_weight:
        print("  mol_prop_emb weights: NON-FINITE weights found:")
        for name, finite, val in weight_reports:
            if not finite:
                print(f"    {name}: NON-FINITE")
    else:
        worst = max(weight_reports, key=lambda t: t[2])
        print(f"  mol_prop_emb weights: all finite. largest |weight| = {worst[2]:.6g} ({worst[0]})")


def instrument_seed_pathway(model: SPICEEnergyForceModel, batch: dict):
    print("\n--- seed_equivariant_features pathway, step by step ---")
    em = model.energy_model
    x = batch["all_solute_x"]
    node_types = batch["all_solute_types"]
    node_mask = batch["all_solute_mask"].bool()
    atomic_masses = batch["all_solute_masses"] * node_mask.to(batch["all_solute_masses"].dtype)
    V = batch["V"]
    mol_props = batch["mol_props"]
    B, N = node_types.shape

    def report(name, t):
        finite = torch.isfinite(t).all().item()
        flag = "" if finite else "  <<< FIRST NON-FINITE TENSOR" if not getattr(report, "_found", False) else "  (non-finite)"
        if not finite and not getattr(report, "_found", False):
            report._found = True
        print(f"  {name:20s} finite={finite!s:5}  max|.|={t.abs().max().item() if finite else float('nan'):.6g}  shape={tuple(t.shape)}{flag}")
        return finite

    with torch.no_grad():
        f0_raw = em.input_embedding(node_types)
        report("f0_raw", f0_raw)
        f0_ffn_out = em.f0_ffn(em.ln1(f0_raw))
        report("f0_ffn_out", f0_ffn_out)

        mol_ctx = em.mol_prop_emb(mol_props).unsqueeze(1).expand(-1, N, -1)
        report("mol_ctx", mol_ctx)
        mol_ffn_out = em.mol_ffn(em.ln2(mol_ctx))
        report("mol_ffn_out", mol_ffn_out)

        combined_scalar = em.seed_proj(torch.cat([f0_ffn_out, mol_ffn_out], dim=-1))
        report("combined_scalar", combined_scalar)

        x_wrapped = minimum_image(x, V)
        report("x_wrapped", x_wrapped)
        center, _ = masked_weighted_com(x_wrapped, atomic_masses, node_mask)
        report("center (COM)", center)

        x_rel = x_wrapped - center.unsqueeze(1)
        report("x_rel", x_rel)

        r_true = x_rel.norm(dim=-1)
        report("r_true", r_true)
        r = r_true.clamp(min=1e-8)
        has_dir = (r_true > 1e-7).unsqueeze(-1)

        Y1 = get_spherical_harmonics(1, x_rel) * has_dir
        report("Y1", Y1)
        Y2 = get_spherical_harmonics(2, x_rel) * has_dir
        report("Y2", Y2)

        radial = em.seed_radial_net(r.reshape(-1, 1)).view(B, N, 1)
        report("seed radial_net(r)", radial)

        scalar = combined_scalar.unsqueeze(-1)
        f1 = scalar * (radial.unsqueeze(-1) * Y1.unsqueeze(2))
        report("f1", f1)
        f2 = scalar * (radial.unsqueeze(-1) * Y2.unsqueeze(2))
        report("f2", f2)

        cutoff = em.seed_radial_net.cutoff_radius
        r_masked = r_true[node_mask]
        n_beyond_cutoff = (r_masked > cutoff).sum().item()
        print(f"\n  seed_radial_net.cutoff_radius={cutoff}, r_true max in batch={r_masked.max().item():.4f}, "
              f"{n_beyond_cutoff}/{r_masked.numel()} real atoms have r_true beyond the cutoff "
              f"(these get an exactly-zero radial envelope by design -- not itself a NaN, but a real "
              f"expressivity gap for outer atoms of large molecules, worth knowing about regardless).")


def run_forward_with_f0_stats(model: SPICEEnergyForceModel, batch: dict, label: str):
    print(f"\n--- Forward pass: {label} ---")
    em = model.energy_model

    captured = {}
    orig_seed_fn = seed_equivariant_features

    def instrumented_seed(*args, **kwargs):
        out = orig_seed_fn(*args, **kwargs)
        captured["f0[1]_max"] = out[1].detach().abs().max().item()
        captured["f0[2]_max"] = out[2].detach().abs().max().item()
        return out

    import stage_1.model as model_module
    model_module.seed_equivariant_features = instrumented_seed
    try:
        U, F = compute_energy_and_forces(
            model, batch["all_solute_x"], batch["all_solute_types"], batch["all_solute_masses"],
            batch["all_solute_mask"], batch["V"], batch["mol_props"], batch["all_solute_edge_feats"],
        )
    finally:
        model_module.seed_equivariant_features = orig_seed_fn

    print(f"  f0[1] max|.|={captured.get('f0[1]_max'):.6g}   f0[2] max|.|={captured.get('f0[2]_max'):.6g}")
    print(f"  U finite={torch.isfinite(U).all().item()}  U range=[{U.min().item():.4g}, {U.max().item():.4g}]")
    print(f"  F finite={torch.isfinite(F).all().item()}  F max|.|={F.abs().max().item():.6g}")
    return U, F


def apply_f0_normalization_patch(model: SPICEEnergyForceModel):
    for layer in model.energy_model.layers:
        norms = torch.nn.ModuleDict({
            str(l): EquivariantRMSNorm(layer.feature_dim, degree=l).to(
                next(layer.parameters()).device, next(layer.parameters()).dtype)
            for l in range(layer.max_degree + 1)
        })
        layer.add_module("_diag_f0_norm", norms)

        orig_forward = layer.forward.__func__

        def patched_forward(self, f_in, f0, g_a_in, x, x_com, edge_feats, node_mask, V, _orig=orig_forward):
            f0_normed = {l: self._diag_f0_norm[str(l)](f0[l]) for l in f0}
            return _orig(self, f_in, f0_normed, g_a_in, x, x_com, edge_feats, node_mask, V)

        layer.forward = patched_forward.__get__(layer, EnergyLayer)


def report_master_attn_fix(model: SPICEEnergyForceModel, batch: dict):
    print("\n--- master_attn magnitude with the Ewald-damping fix (now live in attention.py) ---")
    tracker = attach_hooks(model)
    tracker.reset()
    try:
        with torch.no_grad():
            model(
                batch["all_solute_x"], batch["all_solute_types"], batch["all_solute_masses"],
                batch["all_solute_mask"], batch["V"], batch["mol_props"], batch["all_solute_edge_feats"],
            )
        print(format_hook_report(tracker.stats, top_k=10))
        alphas = {
            f"layer{i}": torch.nn.functional.softplus(layer.master_attn.alpha).item()
            for i, layer in enumerate(model.energy_model.layers)
        }
        print(f"  learned alpha (post-softplus) per layer: {alphas}")
    finally:
        tracker.remove()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--path", type=str, required=True)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    device = torch.device(args.device) if args.device else (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
    saved = torch.load(args.path, map_location=device, weights_only=False)

    step = saved.get("global_step", saved.get("step"))
    model_args = saved.get("config", saved.get("args"))
    if step is None or model_args is None:
        raise KeyError(
            f"Unrecognized dump format at {args.path}: expected 'step'/'args' "
            f"(live_hook_training.py) or 'global_step'/'config' (train_spice.py --diagnostics). "
            f"Found keys: {sorted(saved.keys())}"
        )

    batch = saved["batch"]
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.to(device)
    dtype = batch["all_solute_x"].dtype
    print(f"Loaded {args.path} (saved at step {step}, device={device}, dtype={dtype})")

    if saved.get("reasons"):
        print(f"\n--- Drift/explosion trigger reasons ---")
        for bucket, reason in saved["reasons"].items():
            print(f"  {bucket}: {reason}")

    describe_geometry(batch)

    print("\n--- Forward-activation hook stats at moment of explosion ---")
    print(format_hook_report(saved["hook_stats"], top_k=20))

    if saved.get("grad_stats"):
        print("\n--- Backward-gradient hook stats at moment of explosion (per-module grad norm) ---")
        print(format_hook_report(saved["grad_stats"], top_k=20))

    model = rebuild_model(model_args, device, dtype)
    load_state_dict_allow_new_params(model, saved["model_state_dict"])

    report_full_gradient_breakdown(model, batch, model_args)

    model.eval()

    report_master_attn_fix(model, batch)

    check_mol_props_raw(batch, model)

    instrument_seed_pathway(model, batch)

    run_forward_with_f0_stats(model, batch, label="BASELINE (unpatched)")

    print("\n" + "=" * 90)
    print("Applying f0-normalization patch (LayerNorm f0 before SRAttention reads it)...")
    print("=" * 90)
    apply_f0_normalization_patch(model)
    run_forward_with_f0_stats(model, batch, label="PATCHED (f0 LayerNorm'd before SR)")

    print("\nThe f0 A/B test above is kept for completeness -- it already showed no effect "
          "on this batch (f0 was never the problem; see the master_attn report above for "
          "the change that actually matters).")


if __name__ == "__main__":
    sys.exit(main())
