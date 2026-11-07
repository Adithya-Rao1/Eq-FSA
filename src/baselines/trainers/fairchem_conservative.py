from __future__ import annotations

# energy_phys = e_n * rmsd_E + mean_E (+ per-element references, constant in pos)      
# forces_n = forces_phys / rmsd_F  (mean 0)
# normalised force prediction is  -(rmsd_E / rmsd_F) * d e_n / d pos.

import torch
from fairchem.core.common.registry import registry
from fairchem.core.trainers.ocp_trainer import OCPTrainer

TRAINER_NAME = "forces_conservative"


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
