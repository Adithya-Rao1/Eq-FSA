from __future__ import annotations

# energy_phys = e_n * rmsd_E + mean_E (+ per-element references, constant in pos)      
# forces_n = forces_phys / rmsd_F  (mean 0)
# normalised force prediction is  -(rmsd_E / rmsd_F) * d e_n / d pos.

import importlib.metadata

import torch
from fairchem.core.common.registry import registry
from fairchem.core.trainers.ocp_trainer import OCPTrainer

TRAINER_NAME = "forces_conservative"
PATCHED_FAIRCHEM_VERSION = "1.2.0"

def _init_edge_rot_mat_with_grad(edge_distance_vec):
    import logging

    edge_vec_0 = edge_distance_vec
    edge_vec_0_distance = torch.sqrt(torch.sum(edge_vec_0**2, dim=1))
    if torch.min(edge_vec_0_distance) < 0.0001:
        logging.error(f"Error edge_vec_0_distance: {torch.min(edge_vec_0_distance)}")
    norm_x = edge_vec_0 / (edge_vec_0_distance.view(-1, 1))

    edge_vec_2 = torch.rand_like(edge_vec_0) - 0.5
    edge_vec_2 = edge_vec_2 / (torch.sqrt(torch.sum(edge_vec_2**2, dim=1)).view(-1, 1))
    edge_vec_2b = edge_vec_2.clone()
    edge_vec_2b[:, 0] = -edge_vec_2[:, 1]
    edge_vec_2b[:, 1] = edge_vec_2[:, 0]
    edge_vec_2c = edge_vec_2.clone()
    edge_vec_2c[:, 1] = -edge_vec_2[:, 2]
    edge_vec_2c[:, 2] = edge_vec_2[:, 1]
    vec_dot_b = torch.abs(torch.sum(edge_vec_2b * norm_x, dim=1)).view(-1, 1)
    vec_dot_c = torch.abs(torch.sum(edge_vec_2c * norm_x, dim=1)).view(-1, 1)

    vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)
    edge_vec_2 = torch.where(torch.gt(vec_dot, vec_dot_b), edge_vec_2b, edge_vec_2)
    vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)
    edge_vec_2 = torch.where(torch.gt(vec_dot, vec_dot_c), edge_vec_2c, edge_vec_2)

    vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1))
    assert torch.max(vec_dot) < 0.99

    norm_z = torch.cross(norm_x, edge_vec_2, dim=1)
    norm_z = norm_z / (torch.sqrt(torch.sum(norm_z**2, dim=1, keepdim=True)))
    norm_z = norm_z / (torch.sqrt(torch.sum(norm_z**2, dim=1)).view(-1, 1))
    norm_y = torch.cross(norm_x, norm_z, dim=1)
    norm_y = norm_y / (torch.sqrt(torch.sum(norm_y**2, dim=1, keepdim=True)))

    norm_x = norm_x.view(-1, 3, 1)
    norm_y = -norm_y.view(-1, 3, 1)
    norm_z = norm_z.view(-1, 3, 1)
    edge_rot_mat_inv = torch.cat([norm_z, norm_x, norm_y], dim=2)
    return torch.transpose(edge_rot_mat_inv, 1, 2)  # 1.2.0 returns .detach() here


def enable_frame_gradients() -> None:
    version = importlib.metadata.version("fairchem-core")
    if version != PATCHED_FAIRCHEM_VERSION:
        raise RuntimeError(f"fairchem_conservative copies fairchem-core {PATCHED_FAIRCHEM_VERSION} internals; found {version}, re-check the copies")
    from fairchem.core.models.equiformer_v2 import equiformer_v2 as eqv2_module
    from fairchem.core.models.equiformer_v2 import so3 as so3_module

    def rotation_to_wigner_d_matrix(self, edge_rot_mat, start_lmax, end_lmax):
        o3, wigner_D = so3_module.o3, so3_module.wigner_D
        x = edge_rot_mat @ edge_rot_mat.new_tensor([0.0, 1.0, 0.0])
        alpha, beta = o3.xyz_to_angles(x)
        R = o3.angles_to_matrix(alpha, beta, torch.zeros_like(alpha)).transpose(-1, -2) @ edge_rot_mat
        gamma = torch.atan2(R[..., 0, 2], R[..., 0, 0])
        size = (end_lmax + 1) ** 2 - (start_lmax) ** 2
        wigner = torch.zeros(len(alpha), size, size, device=self.device)
        start = 0
        for lmax in range(start_lmax, end_lmax + 1):
            block = wigner_D(lmax, alpha, beta, gamma)
            end = start + block.size()[1]
            wigner[:, start:end, start:end] = block
            start = end
        return wigner  # 1.2.0 returns .detach() here

    def set_wigner(self, rot_mat3x3):
        self.device, self.dtype = rot_mat3x3.device, rot_mat3x3.dtype
        self.wigner = self.RotationToWignerDMatrix(rot_mat3x3, 0, self.lmax)
        self.wigner_inv = torch.transpose(self.wigner, 1, 2).contiguous()  # 1.2.0 detaches both here

    eqv2_module.init_edge_rot_mat = _init_edge_rot_mat_with_grad
    so3_module.SO3_Rotation.RotationToWignerDMatrix = rotation_to_wigner_d_matrix
    so3_module.SO3_Rotation.set_wigner = set_wigner


enable_frame_gradients()


def conservative_force_scale(normalizers) -> torch.Tensor:
    # rmsd_E / rmsd_F, the factor that turns d(normalised energy)/d pos into normalised forces
    if "energy" not in normalizers or "forces" not in normalizers:
        raise RuntimeError("forces_conservative needs fitted normalizers for both 'energy' and 'forces'")
    return normalizers["energy"].rmsd / normalizers["forces"].rmsd


@registry.register_trainer(TRAINER_NAME)
class ConservativeForcesTrainer(OCPTrainer):
    def _forward(self, batch):
        batch = batch.to(self.device)
        with torch.enable_grad():  # the validation / predict loops run under no_grad
            batch.pos.requires_grad_(True)
            out = self.model(batch)
            energy = out["energy"]
            if isinstance(energy, dict):
                energy = energy["energy"]
            energy = energy.reshape(-1)
            # create_graph only while training --> the force loss must backpropagate through this gradient
            (grad,) = torch.autograd.grad(energy.sum(), batch.pos, create_graph=self.model.training)
        forces = -grad * conservative_force_scale(self.normalizers).to(grad.dtype)
        if not self.model.training:
            energy, forces = energy.detach(), forces.detach()
        return {"energy": energy.view(int(batch.natoms.numel()), -1), "forces": forces.view(int(batch.natoms.sum()), -1)}
