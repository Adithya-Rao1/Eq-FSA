from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from stage_1.experiments.neurreps.oversquashing.synthetic_dataset import (
    GeometricGraphTransferDataset, graph_transfer_collate_fn,
)
from stage_1.experiments.neurreps.oversquashing.train_graph_transfer import (
    build_model_for_variant, build_optimizer, call_model, detect_device, _avg_num_neighbors, _to_device,
)
from stage_1.experiments.neurreps.oversquashing.metrics import EmbeddingCapture, jacobian_sensitivity, mi_lower_bound

SEED = 23757
TOPOLOGY, R, VARIANT = "chain", 2, "eqfsa_full"
EPOCHS = 200

config = {
    "task": "geometric", "classes": 5, "samples_per_config": 300, "cutoff": 5.0,
    "geometric_class_spacing": None, "geometric_seed_retries": 20,
    "val_fraction": 0.2, "batch_size": 32, "lr": 1e-3, "qk_gamma_wd": 0.0,
    "epochs": EPOCHS, "eqfsa_num_layers": 4, "feature_dim": 24, "hidden_dim": 32,
    "eqfsa_normalize_z_by_n_real": True, "torch_compile": False,
    "_device": detect_device(None),
}
print(f"Using device: {config['_device']}")

seed = SEED
dataset = None
for attempt in range(config["geometric_seed_retries"]):
    trial_seed = seed + attempt * 1000003
    try:
        dataset = GeometricGraphTransferDataset(
            TOPOLOGY, R, config["classes"], config["samples_per_config"], seed=trial_seed,
            cutoff=config["cutoff"], class_spacing=config["geometric_class_spacing"],
        )
        seed = trial_seed
        break
    except RuntimeError:
        pass
assert dataset is not None, "no safe geometry found"

label_idx = dataset.label_idx
readout_idx = dataset.readout_idx
num_atom_types = dataset.num_atom_types
print(f"dataset: n_atoms={dataset.n_atoms} realized_r={dataset.realized_r} label_idx={label_idx} readout_idx={readout_idx}")

n_val = max(1, int(round(len(dataset) * config["val_fraction"])))
n_train = len(dataset) - n_val
train_ds, val_ds = random_split(
    dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed + 100000),
)
train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, collate_fn=graph_transfer_collate_fn)
val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, collate_fn=graph_transfer_collate_fn)

device = config["_device"]
avg_num_neighbors = _avg_num_neighbors(dataset.positions, config["cutoff"])
torch.manual_seed(seed)
model = build_model_for_variant(
    VARIANT, dataset.realized_r, config["classes"], num_atom_types, config, avg_num_neighbors=avg_num_neighbors,
).to(device)
embedding_module = model.input_embedding_module
optimizer = build_optimizer(model, config["lr"], config["qk_gamma_wd"])

# ---- training loop, with per-epoch train-set accuracy tracking added ----
train_losses, train_accs = [], []
grad_norm_min, grad_norm_max = float("inf"), 0.0

for epoch in range(EPOCHS):
    model.train()
    epoch_loss_sum, epoch_grad_norm_sum, epoch_batches = 0.0, 0.0, 0
    epoch_grad_norm_max = 0.0
    for batch in train_loader:
        batch = _to_device(batch, device)
        optimizer.zero_grad()
        logits = call_model(model, VARIANT, batch)
        loss = F.cross_entropy(logits, batch["labels"])
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss_sum += loss.item()
        epoch_grad_norm_sum += float(grad_norm)
        epoch_grad_norm_max = max(epoch_grad_norm_max, float(grad_norm))
        epoch_batches += 1
    avg_loss = epoch_loss_sum / max(1, epoch_batches)
    avg_grad_norm = epoch_grad_norm_sum / max(1, epoch_batches)
    train_losses.append(avg_loss)
    grad_norm_min = min(grad_norm_min, avg_grad_norm)
    grad_norm_max = max(grad_norm_max, avg_grad_norm)

    # train-set accuracy in eval mode (answers "does it even fit the 240 training examples")
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in train_loader:
            batch = _to_device(batch, device)
            logits = call_model(model, VARIANT, batch)
            correct += (logits.argmax(dim=-1) == batch["labels"]).sum().item()
            total += logits.shape[0]
    train_acc = correct / max(1, total)
    train_accs.append(train_acc)

    if epoch % 25 == 0 or epoch == EPOCHS - 1:
        print(
            f"  epoch={epoch} loss={avg_loss:.4f} train_acc={train_acc:.4f} "
            f"avg_grad_norm={avg_grad_norm:.4f} max_grad_norm_this_epoch={epoch_grad_norm_max:.4f}"
        )

last10_mean = sum(train_losses[-10:]) / len(train_losses[-10:])
print(f"\nGrad norm trajectory: min={grad_norm_min:.4f} max={grad_norm_max:.4f} last10_loss_mean={last10_mean:.4f}")
best_train_acc = max(train_accs)
min_train_loss = min(train_losses)
print(f"Min train loss over {EPOCHS} epochs: {min_train_loss:.4f}  (ln(classes)={torch.log(torch.tensor(float(config['classes']))).item():.4f})")
print(f"Best train accuracy over {EPOCHS} epochs: {best_train_acc:.4f}  (chance={1/config['classes']:.4f})")
print(f"Train acc trajectory (every 25 epochs): {[f'{a:.3f}' for a in train_accs[::25]]}")

# ---- final validation metrics (mirrors train_one_realization) ----
model.eval()
total_ce, total_correct, total_n = 0.0, 0, 0
sens_values = []
all_logits = []
for batch in val_loader:
    batch = _to_device(batch, device)
    B = batch["labels"].shape[0]
    label_idx_t = torch.full((B,), label_idx, dtype=torch.long, device=device)
    with EmbeddingCapture(embedding_module) as cap:
        logits = call_model(model, VARIANT, batch)
        sens_values.append(jacobian_sensitivity(logits, cap.captured, label_idx_t))
    logits_d = logits.detach()
    all_logits.append(logits_d)
    total_ce += F.cross_entropy(logits_d, batch["labels"], reduction="sum").item()
    total_correct += (logits_d.argmax(dim=-1) == batch["labels"]).sum().item()
    total_n += B

val_ce = total_ce / max(1, total_n)
val_error = 1.0 - total_correct / max(1, total_n)
val_sensitivity = sum(sens_values) / max(1, len(sens_values))
val_mi = mi_lower_bound(config["classes"], val_ce)

print(f"\nFinal: val_error={val_error:.4f} jacobian={val_sensitivity:.4f} mi={val_mi:.4f}")

all_logits = torch.cat(all_logits, dim=0)
probs = torch.softmax(all_logits, dim=-1)
max_probs = probs.max(dim=-1).values
print(f"logit magnitude: mean_abs={all_logits.abs().mean().item():.4f} max_abs={all_logits.abs().max().item():.4f}")
print(
    f"softmax confidence: mean_max_prob={max_probs.mean().item():.4f} min_max_prob={max_probs.min().item():.4f} "
    f"(1/classes={1/config['classes']:.4f} = totally unconfident, 1.0 = totally confident)"
)
print(f"fraction of val examples with max_prob > 0.999: {(max_probs > 0.999).float().mean().item():.4f}")

# ---- ContentWeightedCOM attention weights, one example per class ----
print("\n=== ContentWeightedCOM attention weights (one example per class) ===")
com_module = model.energy_model.com_weighting
samples_per_class = max(1, config["samples_per_config"] // config["classes"])
if com_module is not None:
    captured_score = {}

    def _score_hook(module, inputs, output):
        captured_score["raw"] = output

    handle = com_module.score.register_forward_hook(_score_hook)
    for c in range(config["classes"]):
        item = dataset[c * samples_per_class]
        assert item["label"] == c
        batch = graph_transfer_collate_fn([item])
        batch = _to_device(batch, device)
        with torch.no_grad():
            call_model(model, VARIANT, batch)
        node_mask = batch["mask"]
        logits_raw = captured_score["raw"].squeeze(-1)
        logits_raw = logits_raw.masked_fill(~node_mask, float("-inf"))
        logits_raw = logits_raw.scatter(1, batch["target_idx"].view(-1, 1), float("-inf"))
        weights = torch.softmax(logits_raw, dim=-1)[0]
        top = torch.topk(weights, k=min(3, weights.shape[0]))
        top_str = [f"atom{i.item()}({w.item():.3f})" for w, i in zip(top.values, top.indices)]
        w_label = weights[label_idx].item()
        print(
            f"  class {c}: top attention -> {top_str}   "
            f"(label_idx={label_idx}, readout_idx={readout_idx}, weight_on_label_atom={w_label:.4f})"
        )
    handle.remove()
else:
    print("  content_weighted_com is disabled on this model instance.")