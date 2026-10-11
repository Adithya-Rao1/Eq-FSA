from __future__ import annotations

import torch
import torch.nn as nn

from stage_1.experiments.neurreps.lr_sr.rsa_attention import RSAAttention
from stage_1.lr_restructure.aggregation import MasterNodeAggregator
from stage_1.lr_restructure.broadcast import MasterBroadcastAttention
from stage_1.lr_restructure.update import MasterNodeUpdate
from stage_1.model import masked_mean, masked_weighted_com, seed_equivariant_features
from utils.pw_bases import make_miller_indices, make_wavevectors, minimum_image
from utils.se3_utils import (
    AggregateFeatures, EquivariantRMSNorm, IrrepLinear, RadialNetwork, init_sequential_, scalar_ffn,
)

MACE_OFF_MODEL_SIZE = "small"
MACE_OFF_DTYPE = "float32"
CHARGE_TO_ELEMENT = {1.0: 1, -1.0: 8}  # +1e -> H (Z=1), -1e -> O (Z=8); frozen MACE-OFF needs real elements
CHARGE_TO_ELEMENT_SCRATCH = {1.0: 1, -1.0: 2}


def load_frozen_mace_off(model_size: str = MACE_OFF_MODEL_SIZE, device: str = "cpu", default_dtype: str = MACE_OFF_DTYPE):
    from mace.calculators.foundations_models import mace_off

    mace_model = mace_off(model=model_size, device=device, default_dtype=default_dtype, return_raw_model=True)
    mace_model.to(device=device, dtype=getattr(torch, default_dtype))
    for p in mace_model.parameters():
        p.requires_grad_(False)
    return mace_model


def unfreeze_readouts(mace_model) -> None:
    for p in mace_model.readouts.parameters():
        p.requires_grad_(True)


def build_z_table(mace_model):
    from mace.tools.utils import AtomicNumberTable

    return AtomicNumberTable(mace_model.atomic_numbers.tolist())


def build_scratch_z_table():
    from mace.tools.utils import AtomicNumberTable

    return AtomicNumberTable(sorted(set(CHARGE_TO_ELEMENT_SCRATCH.values())))


def configuration_to_atomic_data(
    positions: torch.Tensor, types: torch.Tensor, box: float, z_table, cutoff: float,
    charge_to_element: "dict | None" = None,
):
    import numpy as np
    from mace.data import AtomicData, Configuration

    from stage_1.experiments.neurreps.lr_sr.dataset import POSITIVE_TYPE_IDX

    charge_to_element = charge_to_element if charge_to_element is not None else CHARGE_TO_ELEMENT
    atomic_numbers = np.array(
        [charge_to_element[1.0] if int(t.item()) == POSITIVE_TYPE_IDX else charge_to_element[-1.0] for t in types],
        dtype=int,
    )
    config = Configuration(
        atomic_numbers=atomic_numbers,
        positions=positions.detach().cpu().numpy().astype(np.float64),
        properties={},
        property_weights={},
        cell=np.eye(3) * box,
        pbc=(True, True, True),
    )
    return AtomicData.from_config(config, z_table=z_table, cutoff=cutoff)


def batch_configurations(
    positions_batch: torch.Tensor, types_batch: torch.Tensor, boxes, z_table, cutoff: float, device,
    idx_batch: "torch.Tensor | None" = None, graph_cache: "dict | None" = None,
    charge_to_element: "dict | None" = None,
):
    from mace.tools.torch_geometric.batch import Batch

    data_list = []
    for b in range(positions_batch.shape[0]):
        cache_key = int(idx_batch[b].item()) if (idx_batch is not None and graph_cache is not None) else None
        if cache_key is not None and cache_key in graph_cache:
            data_list.append(graph_cache[cache_key])
        else:
            data = configuration_to_atomic_data(
                positions_batch[b], types_batch[b], float(boxes[b]), z_table, cutoff,
                charge_to_element=charge_to_element,
            )
            if cache_key is not None:
                graph_cache[cache_key] = data
            data_list.append(data)
    batch = Batch.from_data_list(data_list)
    return batch.to(device)


def compute_avg_num_neighbors(
    dataset, z_table, cutoff: float, charge_to_element: dict, num_samples: int = 50,
) -> float:
    import numpy as np

    n = min(num_samples, len(dataset))
    counts = []
    for i in range(n):
        cfg = dataset.configs[i]
        data = configuration_to_atomic_data(
            cfg["x"], cfg["types"], float(cfg["box"]), z_table, cutoff, charge_to_element=charge_to_element,
        )
        counts.append(data.edge_index.shape[1] / data.positions.shape[0])
    return float(np.mean(counts))


def build_scratch_mace(z_table, config: dict, device):
    import numpy as np
    from e3nn import o3
    from mace import modules

    num_elements = len(z_table)
    feature_dim = config["feature_dim"]
    hidden_irreps = o3.Irreps(f"{feature_dim}x0e + {feature_dim}x1o + {feature_dim}x2e")
    model = modules.ScaleShiftMACE(
        r_max=config.get("mace_r_max", 5.0),
        num_bessel=8,
        num_polynomial_cutoff=5,
        max_ell=2,
        interaction_cls=modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        interaction_cls_first=modules.interaction_classes["RealAgnosticInteractionBlock"],
        num_interactions=config.get("mace_num_interactions", 2),
        num_elements=num_elements,
        hidden_irreps=hidden_irreps,
        edge_irreps=None,
        MLP_irreps=o3.Irreps("16x0e"),
        atomic_energies=np.zeros(num_elements),
        apply_cutoff=True,
        avg_num_neighbors=config["avg_num_neighbors"],
        atomic_numbers=z_table.zs,
        use_reduced_cg=False,
        use_so3=False,
        use_edge_irreps_first=False,
        cueq_config=None,
        correlation=config.get("mace_correlation", 3),
        gate=modules.gate_dict["silu"],
        pair_repulsion=False,
        distance_transform="None",
        radial_MLP=[config["hidden_dim"]] * 3,
        radial_type="bessel",
        atomic_inter_scale=1.0,
        atomic_inter_shift=0.0,
        heads=None,
        embedding_specs=None,
        use_embedding_readout=False,
        use_last_readout_only=False,
        use_agnostic_product=False,
    )
    return model.to(device=device, dtype=torch.get_default_dtype())


class ChargeReadoutEmbedding(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, feature_dim))

    def forward(self, node_energy: torch.Tensor, charges: torch.Tensor) -> torch.Tensor:
        x = torch.stack([node_energy, charges], dim=-1) 
        return self.net(x).unsqueeze(-1)  


class MaceLRModel(nn.Module):
    def __init__(
        self, mace_model, variant: str, feature_dim: int = 32, hidden_dim: int = 64,
        n_max: int = 2, num_master_nodes: int = 4, rsa_sigma: float = 5.0,
        max_degree: int = 2, num_broadcast_heads: int = 4, num_lr_layers: int = 1,
        rbf_num_basis: int = 12, rbf_min: float = 0.3, rbf_cutoff: float = 25.0,
        lr_sigma: "float | None" = None, alpha_floor: float = 1e-4,
        from_scratch: bool = False, charge_to_element: "dict | None" = None,
    ):
        super().__init__()
        assert variant in ("none", "sr_only", "eqfsa_lr", "rsa")
        self.mace_model = mace_model
        self.variant = variant
        self.n_max = n_max
        self.num_lr_layers = num_lr_layers
        self.from_scratch = from_scratch
        self.charge_to_element = charge_to_element if charge_to_element is not None else CHARGE_TO_ELEMENT
        self._graph_cache: dict = {}
        if from_scratch:
            for p in self.mace_model.parameters():
                p.requires_grad_(True)
        else:
            for p in self.mace_model.parameters():
                p.requires_grad_(False)
        if variant == "none":
            return
        if variant == "sr_only":
            if not from_scratch:
                unfreeze_readouts(self.mace_model)
            return

        if not from_scratch:
            unfreeze_readouts(self.mace_model)
        self.embed = ChargeReadoutEmbedding(feature_dim, hidden_dim)
        if variant == "eqfsa_lr":
            self.num_master_nodes = num_master_nodes
            self.max_degree = max_degree
            self.seed_radial_net = RadialNetwork(num_basis=1, hidden_dim=hidden_dim)
            self.g_a_init = nn.Parameter(torch.randn(num_master_nodes, feature_dim) * 0.1)
            self.lr_aggregator = MasterNodeAggregator(
                feature_dim, num_master_nodes, hidden_dim, rbf_num_basis, rbf_min, rbf_cutoff,
            )
            self.lr_update = MasterNodeUpdate(sigma=lr_sigma, alpha_floor=alpha_floor)
            self.W_master_self = IrrepLinear(feature_dim)
            self.master_ln = EquivariantRMSNorm(feature_dim, degree=0)
            self.lr_broadcast = MasterBroadcastAttention(
                RadialNetwork, max_degree, feature_dim, hidden_dim, num_heads=num_broadcast_heads,
            )
            self.register_buffer(
                "lr_miller_indices", make_miller_indices(n_max, dtype=torch.get_default_dtype()),
            )
            self.fusion_pool = AggregateFeatures(
                out_feat_dim=feature_dim, inp_feat_dim=feature_dim, hidden_dim=hidden_dim,
            )
            self.decoder = nn.Sequential(
                nn.Linear(num_master_nodes * feature_dim + feature_dim, hidden_dim),
                nn.SiLU(), nn.Linear(hidden_dim, 1),
            )
        else:
            self.lr_attn = RSAAttention(feature_dim, n_max, sigma=rsa_sigma, hidden_dim=hidden_dim)
            self.decoder = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))

    def _lr_correction(self, node_energy, charges, positions, mask, V):
        f_in0 = self.embed(node_energy, charges)
        if self.variant == "eqfsa_lr":
            B = charges.shape[0]
            masses = mask.float()
            x_com, _ = masked_weighted_com(positions, masses, mask)
            x_wrapped = minimum_image(positions, V)

            seeded = seed_equivariant_features(
                f_in0.squeeze(-1), self.seed_radial_net, positions, masses, mask, V,
            )
            f_atoms = {0: f_in0, 1: seeded[1], 2: seeded[2]}

            g_a = self.g_a_init.unsqueeze(0).expand(B, -1, -1).unsqueeze(-1).clone()
            X_a = x_com.unsqueeze(1).expand(B, self.num_master_nodes, -1).clone()
            kvecs, _, mode_mask = make_wavevectors(V, self.n_max, None, indices=self.lr_miller_indices)

            fusion_accum = {l: torch.zeros_like(f_atoms[l]) for l in f_atoms}
            for _ in range(self.num_lr_layers):
                pooled, X_a_new = self.lr_aggregator(g_a.squeeze(-1), f_atoms[0].squeeze(-1), x_wrapped, X_a, mask, V)
                g_a_new = self.lr_update(pooled, X_a_new, V, kvecs, mode_mask)
                g_a_out = self.master_ln(self.W_master_self(g_a) + g_a_new.unsqueeze(-1))
                layer_fusion = self.lr_broadcast(f_atoms, g_a_out, X_a_new, x_wrapped, mask, V)
                fusion_accum = {l: fusion_accum[l] + layer_fusion[l] for l in fusion_accum}
                g_a, X_a = g_a_out, X_a_new
            fusion_scalar = self.fusion_pool(fusion_accum[0], fusion_accum[1], fusion_accum[2]).squeeze(-1)
            fusion_pooled = masked_mean(fusion_scalar, mask)

            decoder_in = torch.cat([g_a.squeeze(-1).reshape(B, -1), fusion_pooled], dim=-1)
            return self.decoder(decoder_in).squeeze(-1)
        per_atom = self.lr_attn(f_in0, positions, mask, V).squeeze(-1)
        pooled = (per_atom * mask.unsqueeze(-1)).sum(dim=1)
        return self.decoder(pooled).squeeze(-1)

    def forward(self, positions_batch: torch.Tensor, types_batch, charges_batch, boxes, z_table, cutoff: float,
                device, idx=None):
        mace_batch = batch_configurations(
            positions_batch.detach(), types_batch, boxes, z_table, cutoff, device,
            idx_batch=idx, graph_cache=self._graph_cache, charge_to_element=self.charge_to_element,
        )
        N = positions_batch.shape[1]
        mace_batch.positions = positions_batch.reshape(-1, 3)

        data_dict = mace_batch.to_dict()  
        out = self.mace_model(data_dict, training=self.training, compute_force=False)
        node_energy = out["node_energy"].reshape(positions_batch.shape[0], N)  
        node_energy = node_energy.to(positions_batch.dtype)

        node_e0 = self.mace_model.atomic_energies_fn(data_dict["node_attrs"]).squeeze(-1)
        node_e0 = node_e0.reshape(positions_batch.shape[0], N).to(positions_batch.dtype)
        node_energy = node_energy - node_e0

        total_sr_energy = node_energy.sum(dim=1)

        if self.variant in ("none", "sr_only"):
            return total_sr_energy

        mask = torch.ones(positions_batch.shape[0], N, dtype=torch.bool, device=device)
        V = torch.stack([torch.eye(3, device=device, dtype=positions_batch.dtype) * float(b) for b in boxes], dim=0)
        e_lr = self._lr_correction(node_energy, charges_batch, positions_batch, mask, V)
        return total_sr_energy + e_lr


def compute_energy_and_forces(model: MaceLRModel, positions_batch, types_batch, charges_batch, boxes, z_table, cutoff, device, idx=None):
    x = positions_batch.clone().requires_grad_(True)
    U = model(x, types_batch, charges_batch, boxes, z_table, cutoff, device, idx=idx)
    (grad_x,) = torch.autograd.grad(U.sum(), x, create_graph=model.training)
    return U, -grad_x
