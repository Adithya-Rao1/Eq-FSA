from utils.se3_utils import RadialNetwork
from stage_1.spice_dataset import _EDGE_FEAT_DIM
from stage_1.lr_restructure.model import LREnergyModel

CONFIG = dict(
    radial_net=RadialNetwork,
    num_atom_types=23,
    max_degree=2,
    num_layers=4,
    feature_dim=64,
    hidden_dim=64,
    edge_cutoff=5.0,
    n_max=1,
    n_master_nodes=8,
    num_broadcast_heads=4,
    edge_feat_dim=_EDGE_FEAT_DIM,
)


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def per_layer_counts(model) -> list:
    return [count_params(layer) for layer in model.layers]


def main():
    m_off = LREnergyModel(**CONFIG, lazy_degree_gating=False)
    m_on = LREnergyModel(**CONFIG, lazy_degree_gating=True)

    n_off, n_on = count_params(m_off), count_params(m_on)
    print(f"lazy_degree_gating=False: {n_off:,} params")
    print(f"lazy_degree_gating=True:  {n_on:,} params")
    print(f"reduction: {n_off - n_on:,} params ({100 * (n_off - n_on) / n_off:.1f}%)")
    print()
    print("per-layer breakdown (k_max schedule = [0, 1, 2, 1] under lazy gating):")
    for i, (c_off, c_on) in enumerate(zip(per_layer_counts(m_off), per_layer_counts(m_on))):
        print(f"  layer {i}: off={c_off:,}  on={c_on:,}  reduction={100*(c_off-c_on)/c_off:.1f}%")


if __name__ == "__main__":
    main()
