from __future__ import annotations

_BASE_CLI_KEYS = [
    "hdf5_paths", "subsets", "batch_size", "lr", "warmup_epochs", "min_lr", "weight_decay",
    "lambda1", "lambda2", "feature_dim", "hidden_dim", "max_degree", "num_layers", "edge_cutoff",
    "box_padding", "grad_clip_norm", "n_max", "pw_e_cutoff", "num_atom_types", "n_master_nodes",
    "num_fusion_heads", "val_fraction", "test_fraction", "num_workers", "prefetch_factor",
    "log_interval", "save_every", "val_every",
    "device", "diagnostics", "diag_skip_hooks", "diag_skip_buckets", "diag_explode_threshold",
    "diag_ema_decay", "diag_drift_k", "diag_ring_k", "diag_ckpt_every", "diag_ckpt_keep",
]

_FULL_EPOCH_NOMINAL_CEILING = 10_000_000


def base_overrides(base: dict) -> dict:
    return {k: base[k] for k in _BASE_CLI_KEYS if k in base}


def build_2_1(base: dict) -> list[dict]:
    common = base_overrides(base)
    variants = []
    for vid, diag, skip_hooks, skip_buckets in [
        ("diag_off", False, False, False),
        ("diag_full", True, False, False),
        ("diag_skip_hooks", True, True, False),
        ("diag_skip_buckets", True, False, True),
    ]:
        ov = dict(common)
        ov.update(diagnostics=diag, diag_skip_hooks=skip_hooks, diag_skip_buckets=skip_buckets)
        variants.append({"id": vid, "cli_overrides": ov, "max_steps": 1000, "epoch_cap": False})
    return variants


def build_2_3(base: dict) -> list[dict]:
    common = base_overrides(base)
    base_bs = common.get("batch_size", 32)
    base_lr = common.get("lr", 1e-3)
    variants = [{"id": f"bs{base_bs}_control", "cli_overrides": dict(common), "max_steps": 5000, "epoch_cap": False}]
    for bs in (16, 64, 128):
        if bs == base_bs:
            continue
        for lr_label, lr in (("lr_unscaled", base_lr), ("lr_scaled", base_lr * bs / base_bs)):
            ov = dict(common)
            ov.update(batch_size=bs, lr=lr)
            variants.append({"id": f"bs{bs}_{lr_label}", "cli_overrides": ov, "max_steps": 5000, "epoch_cap": False})
    return variants


def build_2_5(base: dict) -> list[dict]:
    common = base_overrides(base)
    base_l1 = common.get("lambda1", 1.0)
    base_l2 = common.get("lambda2", 1.0)
    grid = [
        ("baseline", base_l1, base_l2),
        ("lambda2_half", base_l1, base_l2 * 0.5),
        ("lambda2_quarter", base_l1, base_l2 * 0.25),
        ("lambda1_double", base_l1 * 2.0, base_l2),
    ]
    variants = []
    for vid, l1, l2 in grid:
        ov = dict(common)
        ov.update(lambda1=l1, lambda2=l2)
        variants.append({"id": vid, "cli_overrides": ov, "max_steps": _FULL_EPOCH_NOMINAL_CEILING, "epoch_cap": True})
    return variants


def build_2_6(base: dict) -> list[dict]:
    common = base_overrides(base)
    base_warmup = common.get("warmup_epochs", 1)
    base_min_lr = common.get("min_lr", 1e-6)
    base_wd = common.get("weight_decay", 1e-4)
    grid = [
        ("baseline", {}),
        ("warmup_plus1", {"warmup_epochs": base_warmup + 1}),
        ("warmup_plus2", {"warmup_epochs": base_warmup + 2}),
        ("weight_decay_5x", {"weight_decay": base_wd * 5}),
        ("min_lr_10x", {"min_lr": base_min_lr * 10}),
    ]
    variants = []
    for vid, overrides in grid:
        ov = dict(common)
        ov.update(overrides)
        variants.append({"id": vid, "cli_overrides": ov, "max_steps": 5000, "epoch_cap": True})
    return variants


def build_2_7(base: dict) -> list[dict]:
    common = base_overrides(base)
    ov = dict(common)
    ov["diagnostics"] = True
    return [{
        "id": "baseline_5epoch_watch", "cli_overrides": ov,
        "max_steps": _FULL_EPOCH_NOMINAL_CEILING, "epoch_cap": False, "epoch_multiplier": 5,
    }]


PHASE2_ITEMS = [
    ("2.1_diagnostics_overhead", build_2_1, "diagnostics-off vs. on (+2 partial configs) at realistic scale"),
    ("2.3_batch_size_lr", build_2_3, "batch-size sweep paired with an LR-scaling check"),
    ("2.5_lambda_rebalance", build_2_5, "lambda1/lambda2 rebalancing, up to a full epoch each"),
    ("2.6_warmup_schedule", build_2_6, "warmup/min_lr/weight_decay tuning for the epoch-0 spike"),
    ("2.7_progressive_instability_watch", build_2_7, "watch Z_min/g_a_norm_max/weight-norm buckets develop a trend across several real epochs"),
]