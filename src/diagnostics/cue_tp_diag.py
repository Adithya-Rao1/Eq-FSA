import torch

torch.manual_seed(0)
DTYPE = torch.double
device = torch.device("cuda")

from base_tests import random_rotation_matrix, to_cartesian_vector, to_cartesian_tensor
from utils.se3_utils import RadialNetwork, BatchedRadialNetwork
from utils.spherical_harm import get_spherical_harmonics
from utils.cue_backend import e3nn_to_cue_basis, cuequivariance_available
from stage_1.lr_restructure.sr_attention import LRDirectSumW
from stage_1.lr_restructure.broadcast import MasterBroadcastAttention


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def report(name, diff, scale):
    print(f"  {name}: max|diff| = {diff:.3e}   (reference scale ~ {scale:.3e})")


def main():
    assert cuequivariance_available(), "cuequivariance not available on this device -- aborting"
    R = random_rotation_matrix(dtype=DTYPE).to(device)

    section("A. e3nn<->cuEquivariance basis-change matrices for l=0,1,2")
    for l in range(3):
        A = e3nn_to_cue_basis(l, device, DTYPE)
        is_identity = torch.allclose(A, torch.eye(2 * l + 1, dtype=DTYPE, device=device), atol=1e-4)
        print(f"l={l}: is_identity={is_identity}")
        print(A)

    section("B. LRDirectSumW isolated equivariance check (cue path)")
    max_degree, k_max, feature_dim = 2, 2, 4

    def build_radial_nets():
        torch.manual_seed(1)
        radial_nets = torch.nn.ModuleDict()
        for l in range(max_degree + 1):
            n_copies = sum((l + k + 1) - abs(l - k) for k in range(k_max + 1))
            radial_nets[str(l)] = BatchedRadialNetwork(
                n_copies=n_copies, num_basis=(2 * l + 1), hidden_dim=16, radial_net_cls=RadialNetwork,
            )
        return radial_nets.to(device=device, dtype=DTYPE)

    def build_direct_sum_w(use_cue):
        torch.manual_seed(2)
        mod = LRDirectSumW(
            build_radial_nets(), max_degree, k_max, feature_dim=feature_dim,
            radial_net_cls=RadialNetwork, cue_hidden_dim=16, use_cuequivariance=use_cue,
        )
        return mod.to(device=device, dtype=DTYPE)

    N = 9
    base_vecs = torch.randn(N, 3, dtype=DTYPE, device=device)
    x = torch.randn(N, 3, dtype=DTYPE, device=device)
    f0 = torch.randn(N, feature_dim, 1, dtype=DTYPE, device=device)
    f1 = get_spherical_harmonics(1, base_vecs).unsqueeze(1).expand(N, feature_dim, 3).contiguous()
    f2 = get_spherical_harmonics(2, base_vecs).unsqueeze(1).expand(N, feature_dim, 5).contiguous()
    f = {0: f0, 1: f1, 2: f2}

    base_vecs_rot = base_vecs @ R.T
    x_rot = x @ R.T
    f_rot = {
        0: f0.clone(),
        1: get_spherical_harmonics(1, base_vecs_rot).unsqueeze(1).expand(N, feature_dim, 3).contiguous(),
        2: get_spherical_harmonics(2, base_vecs_rot).unsqueeze(1).expand(N, feature_dim, 5).contiguous(),
    }

    for label, use_cue in [("reference", False), ("cue", True)]:
        mod = build_direct_sum_w(use_cue)
        if use_cue and not mod._cue_ready:
            print(f"  [{label}] cue path not ready, skipping")
            continue
        out = mod(f, x)
        out_rot = mod(f_rot, x_rot)

        print(f" -- {label} --")
        report("out[0] invariance", (out[0] - out_rot[0]).abs().max().item(), out[0].abs().max().item())

        cart1 = to_cartesian_vector(out[1].reshape(-1, 3))
        cart1_rot = to_cartesian_vector(out_rot[1].reshape(-1, 3))
        expected1 = torch.einsum("ij,...j->...i", R, cart1)
        report("out[1] equivariance", (cart1_rot - expected1).abs().max().item(), cart1.abs().max().item())

        cart2 = to_cartesian_tensor(out[2].reshape(-1, 5))
        cart2_rot = to_cartesian_tensor(out_rot[2].reshape(-1, 5))
        expected2 = torch.einsum("ij,...jk,lk->...il", R, cart2, R)
        report("out[2] equivariance", (cart2_rot - expected2).abs().max().item(), cart2.abs().max().item())

    section("C. MasterBroadcastAttention._lift_master_content_cue isolated equivariance check")
    max_degree2, feature_dim2, num_heads2, hidden_dim2 = 2, 8, 2, 16

    def build_broadcast(use_cue):
        torch.manual_seed(3)
        mod = MasterBroadcastAttention(
            RadialNetwork, max_degree2, feature_dim2, hidden_dim2,
            num_heads=num_heads2, use_cuequivariance=use_cue,
        )
        return mod.to(device=device, dtype=DTYPE)

    B, Nn, N_a = 1, 4, 2
    content_all = torch.randn(2, B, N_a, max_degree2 + 1, feature_dim2, dtype=DTYPE, device=device)
    x_rel_flat = torch.randn(B * Nn * N_a, 3, dtype=DTYPE, device=device)
    has_dir = torch.ones(B, Nn, N_a, 1, 1, dtype=DTYPE, device=device)
    x_rel_flat_rot = (x_rel_flat.view(B, Nn, N_a, 3) @ R.T).reshape(B * Nn * N_a, 3)

    for label, use_cue in [("reference", False), ("cue", True)]:
        mod = build_broadcast(use_cue)
        if use_cue and not mod._cue_ready:
            print(f"  [{label}] cue path not ready, skipping")
            continue
        H2, Ch2 = mod.num_heads, mod.head_dim
        fn = mod._lift_master_content_cue if use_cue else mod._lift_master_content_reference
        if use_cue:
            K, V = fn(content_all, x_rel_flat, has_dir, B, Nn, N_a, H2, Ch2)
            K_rot, V_rot = fn(content_all, x_rel_flat_rot, has_dir, B, Nn, N_a, H2, Ch2)
        else:
            r = x_rel_flat.norm(dim=-1, keepdim=True)
            r_rot = x_rel_flat_rot.norm(dim=-1, keepdim=True)
            L, W = max_degree2 + 1, 2 * max_degree2 + 1
            K, V = fn(content_all, x_rel_flat, has_dir, r, B, Nn, N_a, L, W, H2, Ch2, device, DTYPE)
            K_rot, V_rot = fn(content_all, x_rel_flat_rot, has_dir, r_rot, B, Nn, N_a, L, W, H2, Ch2, device, DTYPE)

        print(f" -- {label} --")
        report("K_lifted[0] invariance", (K[0] - K_rot[0]).abs().max().item(), K[0].abs().max().item())

        cartK1 = to_cartesian_vector(K[1].reshape(-1, 3))
        cartK1_rot = to_cartesian_vector(K_rot[1].reshape(-1, 3))
        expectedK1 = torch.einsum("ij,...j->...i", R, cartK1)
        report("K_lifted[1] equivariance", (cartK1_rot - expectedK1).abs().max().item(), cartK1.abs().max().item())

        cartK2 = to_cartesian_tensor(K[2].reshape(-1, 5))
        cartK2_rot = to_cartesian_tensor(K_rot[2].reshape(-1, 5))
        expectedK2 = torch.einsum("ij,...jk,lk->...il", R, cartK2, R)
        report("K_lifted[2] equivariance", (cartK2_rot - expectedK2).abs().max().item(), cartK2.abs().max().item())


if __name__ == "__main__":
    main()
