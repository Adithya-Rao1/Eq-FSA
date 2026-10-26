from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from stage_1.train_spice import SPICEEnergyForceModel

class MagnitudeTracker:
    def __init__(self):
        self.stats: dict[str, float] = {}
        self.handles = []

    def _hook(self, name):
        def fn(module, inputs, output):
            if isinstance(output, dict):
                vals = [v for v in output.values() if torch.is_tensor(v)]
            elif isinstance(output, (tuple, list)):
                vals = [v for v in output if torch.is_tensor(v)]
            elif torch.is_tensor(output):
                vals = [output]
            else:
                return
            for v in vals:
                if v.numel() == 0:
                    continue
                is_finite = torch.isfinite(v).all().item()
                self.stats[name] = float("nan") if not is_finite else v.detach().abs().max().item()
        return fn

    def attach(self, module: torch.nn.Module, name: str):
        self.handles.append(module.register_forward_hook(self._hook(name)))

    def reset(self):
        self.stats.clear()

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def report(self, top_k: int = 30) -> str:
        return format_hook_report(self.stats, top_k=top_k)


class GradientTracker:
    def __init__(self):
        self.stats: dict[str, float] = {}
        self.handles = []

    def _make_grad_hook(self, name):
        def grad_hook(grad):
            if grad is None or grad.numel() == 0:
                return
            is_finite = torch.isfinite(grad).all().item()
            self.stats[name] = float("nan") if not is_finite else grad.detach().norm().item()
        return grad_hook

    def _forward_hook(self, name):
        def fn(module, inputs, output):
            if isinstance(output, dict):
                vals = [v for v in output.values() if torch.is_tensor(v)]
            elif isinstance(output, (tuple, list)):
                vals = [v for v in output if torch.is_tensor(v)]
            elif torch.is_tensor(output):
                vals = [output]
            else:
                return
            for v in vals:
                if v.requires_grad:
                    v.register_hook(self._make_grad_hook(name))
        return fn

    def attach(self, module: torch.nn.Module, name: str):
        self.handles.append(module.register_forward_hook(self._forward_hook(name)))

    def reset(self):
        self.stats.clear()

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def report(self, top_k: int = 30) -> str:
        return format_hook_report(self.stats, top_k=top_k)


def format_hook_report(stats: dict, top_k: int = 30) -> str:
    non_finite = sorted(n for n, v in stats.items() if v != v)
    finite = sorted(((n, v) for n, v in stats.items() if v == v), key=lambda kv: -abs(kv[1]))
    lines = [f"  {name:60s} max|.|=nan  [NON-FINITE]" for name in non_finite]
    lines += [f"  {name:60s} max|.|={val:.6g}" for name, val in finite[:top_k]]
    return "\n".join(lines)


def _instrumented_submodules_production(model: SPICEEnergyForceModel):
    em = model.energy_model
    yield em.seed_radial_net, "seed_radial_net"
    yield em.master_pool_agg, "master_pool_agg"
    yield em.master_pool_ffn, "master_pool_ffn"
    yield em.master_expand, "master_expand"
    yield em.master_expand_norm, "master_expand_norm"
    yield em.master_fusion, "master_fusion"
    yield em.mol_prop_emb, "mol_prop_emb"
    yield em.f0_ffn, "f0_ffn"
    yield em.mol_ffn, "mol_ffn"
    yield em.seed_proj, "seed_proj"
    yield em.seed_proj_out, "seed_proj_out"
    yield em.seed_proj_norm, "seed_proj_norm"
    yield em.input_embedding, "input_embedding"

    yield em.readout.ffn0, "energy_model.readout.ffn0"
    yield em.readout.ffn1, "energy_model.readout.ffn1"
    yield em.readout.ffn2, "energy_model.readout.ffn2"

    for i, layer in enumerate(em.layers):
        yield layer.sr_attn, f"layer{i}.sr_attn"
        yield layer.sr_attn.psi_e, f"layer{i}.sr_attn.psi_e"
        yield layer.sr_attn.pair_readout, f"layer{i}.sr_attn.pair_readout"
        for lk_key, rnet in layer.sr_attn.radial_V_sr.items():
            yield rnet, f"layer{i}.sr_attn.radial_V_sr[{lk_key}]"
        yield layer.master_attn, f"layer{i}.master_attn"
        yield layer.master_ln, f"layer{i}.master_ln"
        yield layer.W_V_self_linear["0"], f"layer{i}.W_V_self_linear[0]"
        yield layer.W_V_self_norm, f"layer{i}.W_V_self_norm"
        yield layer.sigma_gate, f"layer{i}.sigma_gate"
        yield layer.layer_norm_in, f"layer{i}.layer_norm_in"
        for l, ffn in layer.readout.ffn.items():
            yield ffn, f"layer{i}.readout.ffn{l}"

    yield model.final_out, "final_out"


def _instrumented_submodules_lr_restructure(model: SPICEEnergyForceModel):
    em = model.energy_model
    yield em.seed_radial_net, "seed_radial_net"
    yield em.mol_prop_emb, "mol_prop_emb"
    yield em.f0_ffn, "f0_ffn"
    yield em.mol_ffn, "mol_ffn"
    yield em.seed_proj, "seed_proj"
    yield em.seed_proj_out, "seed_proj_out"
    yield em.seed_proj_norm, "seed_proj_norm"
    yield em.input_embedding, "input_embedding"

    yield em.readout.ffn0, "energy_model.readout.ffn0"
    yield em.readout.ffn1, "energy_model.readout.ffn1"
    yield em.readout.ffn2, "energy_model.readout.ffn2"

    for i, layer in enumerate(em.layers):
        yield layer.sr_attn, f"layer{i}.sr_attn"
        yield layer.sr_attn.psi_e, f"layer{i}.sr_attn.psi_e"
        yield layer.sr_attn.pair_readout, f"layer{i}.sr_attn.pair_readout"
        yield layer.sr_attn.direct_sum_w, f"layer{i}.sr_attn.direct_sum_w"
        if getattr(layer, "use_long_range", True):
            yield layer.aggregator, f"layer{i}.aggregator"
            yield layer.update, f"layer{i}.update"
            yield layer.broadcast, f"layer{i}.broadcast"
            yield layer.W_master_self, f"layer{i}.W_master_self"
            yield layer.master_ln, f"layer{i}.master_ln"
        yield layer.W_V_self_linear["0"], f"layer{i}.W_V_self_linear[0]"
        yield layer.W_V_self_norm, f"layer{i}.W_V_self_norm"
        yield layer.sigma_gate, f"layer{i}.sigma_gate"
        yield layer.layer_norm_in, f"layer{i}.layer_norm_in"
        for l, ffn in layer.readout.ffn.items():
            yield ffn, f"layer{i}.readout.ffn{l}"

    yield model.final_out, "final_out"


def _instrumented_submodules(model: SPICEEnergyForceModel):
    backbone = getattr(model, "backbone", "production")
    if backbone == "lr_restructure":
        yield from _instrumented_submodules_lr_restructure(model)
    else:
        yield from _instrumented_submodules_production(model)


def attach_hooks(model: SPICEEnergyForceModel) -> MagnitudeTracker:
    tracker = MagnitudeTracker()
    for module, name in _instrumented_submodules(model):
        tracker.attach(module, name)
    return tracker


def attach_grad_hooks(model: SPICEEnergyForceModel) -> GradientTracker:
    tracker = GradientTracker()
    for module, name in _instrumented_submodules(model):
        tracker.attach(module, name)
    return tracker
