from __future__ import annotations

import abc
from typing import List, Optional

import torch
import torch.nn as nn
from rdkit import Chem


def _install_numpy_legacy_aliases():
    import numpy as np
    for name, replacement in [("bool", bool), ("int", int), ("float", float), ("object", object), ("str", str)]:
        if not hasattr(np, name):
            setattr(np, name, replacement)


def _smiles_and_positions_to_ase_atoms(smiles: str, positions: torch.Tensor):
    from ase import Atoms
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    if len(atomic_numbers) != positions.shape[0]:
        raise ValueError(
            f"atom count mismatch: Chem.AddHs(MolFromSmiles({smiles!r})) has {len(atomic_numbers)} "
            f"atoms but positions has {positions.shape[0]} -- conformer and SMILES must come from "
            f"the same Chem.AddHs(...) atom ordering (see load_combisolv.py's _embed_molecule_conformer "
            f"and _atom_types_from_smiles, which both call Chem.AddHs on the same parsed mol)"
        )
    return Atoms(numbers=atomic_numbers, positions=positions.detach().cpu().numpy())


class ReadoutHead(nn.Module):
    def __init__(self, solute_descriptor_dim: int, solvent_descriptor_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(solute_descriptor_dim + solvent_descriptor_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, solute_desc: torch.Tensor, solvent_desc: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([solute_desc, solvent_desc], dim=-1)).squeeze(-1)


class SolvationBaseline(abc.ABC):
    solute_descriptor_dim: int
    supports_backbone_finetune: bool = False

    def __init__(self, solvent_descriptor_dim: int, device: str = "cpu"):
        self.device = device
        self.readout = ReadoutHead(self.solute_descriptor_dim, solvent_descriptor_dim).to(device)
        self._fine_tuned = False

    @abc.abstractmethod
    def _solute_descriptor(self, solute_smiles: str, solute_x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def predict(self, solute_smiles: str, solute_x: torch.Tensor, solvent_descriptor: torch.Tensor) -> float:
        if not self._fine_tuned:
            raise RuntimeError(
                f"{type(self).__name__}.predict() called before fine_tune(): the pretrained backbone "
                f"has no native solvation-free-energy output, and the readout head is randomly "
                f"initialized -- its output right now would be meaningless, not a zero-shot estimate. "
                f"Call fine_tune(dataset) first."
            )
    
        solute_desc = self._solute_descriptor(solute_smiles, solute_x).unsqueeze(0)
        solvent_desc = solvent_descriptor.to(self.device).unsqueeze(0)
        with torch.no_grad():
            return float(self.readout(solute_desc, solvent_desc).item())

    def fine_tune(
        self,
        dataset, 
        n_epochs: int = 20,
        lr: float = 1e-3,
        batch_size: int = 16,
        freeze_backbone: bool = True,
        verbose: bool = True,
    ) -> List[float]:
        if not freeze_backbone and not self.supports_backbone_finetune:
            raise NotImplementedError(
                f"{type(self).__name__} does not support freeze_backbone=False (see module "
                f"docstring) -- only readout-head fine-tuning is available for this backbone."
            )

        params = list(self.readout.parameters())
        if not freeze_backbone:
            params += list(self._backbone_parameters())
        optimizer = torch.optim.Adam(params, lr=lr)

        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda b: b)

        history: List[float] = []
        for epoch in range(n_epochs):
            epoch_losses = []
            for batch in loader:
                optimizer.zero_grad()

                solute_descs = torch.stack([
                    self._solute_descriptor(item["solute_smiles"], item["solute_x"]) for item in batch
                ])
                solvent_descs = torch.stack(
                    [item["solvent_descriptor"] for item in batch]
                ).to(self.device)
                targets = torch.tensor(
                    [item["target_dG"] for item in batch], dtype=torch.float32, device=self.device,
                )

                preds = self.readout(solute_descs, solvent_descs)
                loss = nn.functional.mse_loss(preds, targets)
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.item()))

            mean_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
            history.append(mean_loss)
            if verbose:
                print(f"[{type(self).__name__}.fine_tune] epoch {epoch}: mse={mean_loss:.4f}")

        self._fine_tuned = True
        return history

    def _backbone_parameters(self):
        return []


class MACEBaseline(SolvationBaseline):
    supports_backbone_finetune = False

    def __init__(
        self,
        solvent_descriptor_dim: int,
        model: str = "small",
        device: str = "cpu",
        default_dtype: str = "float32",
    ):
        from mace.calculators import mace_off
        from ase import Atoms

        self.calc = mace_off(model=model, device=device, default_dtype=default_dtype, return_raw_model=False)

        probe = Atoms(numbers=[6], positions=[[0.0, 0.0, 0.0]])
        probe_desc = self.calc.get_descriptors(probe, invariants_only=True)
        self.solute_descriptor_dim = int(probe_desc.shape[-1])

        super().__init__(solvent_descriptor_dim=solvent_descriptor_dim, device=device)

    def _solute_descriptor(self, solute_smiles: str, solute_x: torch.Tensor) -> torch.Tensor:
        atoms = _smiles_and_positions_to_ase_atoms(solute_smiles, solute_x)
        desc = self.calc.get_descriptors(atoms, invariants_only=True)  
        pooled = desc.mean(axis=0)
        return torch.tensor(pooled, dtype=torch.float32, device=self.device)


class FieldSchNetBaseline(SolvationBaseline):
    supports_backbone_finetune = True

    def __init__(
        self,
        solvent_descriptor_dim: int,
        model_path: str,
        device: str = "cpu",
    ):
        _install_numpy_legacy_aliases()
        from schnetpack.data import AtomsConverter
        from schnetpack import Properties

        self.device = device
        self.model = torch.load(model_path, map_location=device, weights_only=False).to(device)
        self.model.eval()
        self.atoms_converter = AtomsConverter(device=device)
        self.Properties = Properties

        self.solute_descriptor_dim = self.model.field_representation.embedding.embedding_dim

        super().__init__(solvent_descriptor_dim=solvent_descriptor_dim, device=device)

    def _representation_inputs(self, atoms):
        from ase import units

        model_inputs = self.atoms_converter(atoms)
        model_inputs[self.Properties.R] = model_inputs[self.Properties.R] / units.Bohr
        for field in self.model.field_representation.fields:
            model_inputs[field] = torch.zeros(1, 3, device=self.device)
        return model_inputs

    def _solute_descriptor(self, solute_smiles: str, solute_x: torch.Tensor) -> torch.Tensor:
        atoms = _smiles_and_positions_to_ase_atoms(solute_smiles, solute_x)
        model_inputs = self._representation_inputs(atoms)
        x, _ = self.model.field_representation(model_inputs)  
        mask = model_inputs[self.Properties.atom_mask]
        pooled = (x * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        return pooled.squeeze(0)

    def _backbone_parameters(self):
        return list(self.model.field_representation.parameters())


class DeePMDBaseline(SolvationBaseline):
    supports_backbone_finetune = False

    def __init__(
        self,
        solvent_descriptor_dim: int,
        model_name: str = "DPA-3.1-3M",
        head: Optional[str] = "SPICE2",
        device: str = "cpu",
    ):
        from deepmd.infer import DeepPot
        from deepmd.pretrained.download import download_model

        model_path = download_model(model_name)
        self.dp = DeepPot(str(model_path), head=head) if head is not None else DeepPot(str(model_path))
        self.type_map = self.dp.get_type_map()

        probe_coords = torch.zeros(1, 1, 3).numpy()
        probe_types = torch.zeros(1, dtype=torch.long).numpy()
        probe_desc = self.dp.eval_descriptor(probe_coords, cells=None, atom_types=probe_types)
        self.solute_descriptor_dim = int(probe_desc.shape[-1])

        super().__init__(solvent_descriptor_dim=solvent_descriptor_dim, device=device)

    def _atom_types_for_smiles(self, solute_smiles: str):
        import numpy as np
        mol = Chem.AddHs(Chem.MolFromSmiles(solute_smiles))
        atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        symbols = [Chem.GetPeriodicTable().GetElementSymbol(z) for z in atomic_numbers]
        try:
            return np.array([self.type_map.index(s) for s in symbols], dtype=np.int64)
        except ValueError as e:
            raise ValueError(
                f"an element in {solute_smiles!r} ({set(symbols) - set(self.type_map)}) is not in "
                f"this DeePMD checkpoint's type_map ({self.type_map}) -- this checkpoint/head can't "
                f"be used for this molecule"
            ) from e

    def _solute_descriptor(self, solute_smiles: str, solute_x: torch.Tensor) -> torch.Tensor:
        coords = solute_x.detach().cpu().numpy().reshape(1, -1, 3)
        atom_types = self._atom_types_for_smiles(solute_smiles)
        desc = self.dp.eval_descriptor(coords, cells=None, atom_types=atom_types)
        pooled = desc[0].mean(axis=0)
        return torch.tensor(pooled, dtype=torch.float32, device=self.device)


if __name__ == "__main__":
    import argparse

    from stage_1.experiments.load_combisolv import (
        Stage2SolvationDataset, load_combined_pairs, SOLVENT_DESCRIPTOR_DIM,
    )

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit", type=int, default=8, help="Number of (solute, solvent) pairs for this smoke test")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--skip-fieldschnet", action="store_true",
                    help="Skip FieldSchNetBaseline (needs schnetpack==0.3 + the field_schnet fork installed)")
    p.add_argument("--skip-deepmd", action="store_true",
                    help="Skip DeePMDBaseline (needs deepmd-kit's compiled op library)")
    args = p.parse_args()

    pairs = load_combined_pairs(verbose=True)[: args.limit]
    dataset = Stage2SolvationDataset(pairs=pairs, verbose=True)
    if len(dataset) < 2:
        raise SystemExit("Need at least 2 usable pairs for a meaningful smoke test; increase --limit.")

    print("\n=== MACEBaseline fine-tune smoke test ===")
    mace_baseline = MACEBaseline(solvent_descriptor_dim=SOLVENT_DESCRIPTOR_DIM, model="small", device="cpu")
    history = mace_baseline.fine_tune(dataset, n_epochs=args.epochs, batch_size=min(4, len(dataset)))
    assert len(history) == args.epochs
    sample = dataset[0]
    pred = mace_baseline.predict(sample["solute_smiles"], sample["solute_x"], sample["solvent_descriptor"])
    print(f"MACE post-fine-tune prediction for {sample['solute_smiles']}/{sample['solvent_smiles']}: "
          f"{pred:.4f} kcal/mol (target: {sample['target_dG']:.4f})")

    if not args.skip_fieldschnet:
        print("\n=== FieldSchNetBaseline fine-tune smoke test ===")
        import os
        model_path = os.path.normpath(os.path.join(
            os.path.dirname(__file__), "..", "field_schnet", "example", "ethanol_vacuum.model",
        ))
        fs_baseline = FieldSchNetBaseline(
            solvent_descriptor_dim=SOLVENT_DESCRIPTOR_DIM, model_path=model_path, device="cpu",
        )
        history = fs_baseline.fine_tune(dataset, n_epochs=args.epochs, batch_size=min(4, len(dataset)))
        assert len(history) == args.epochs
        pred = fs_baseline.predict(sample["solute_smiles"], sample["solute_x"], sample["solvent_descriptor"])
        print(f"FieldSchNet post-fine-tune prediction for {sample['solute_smiles']}/{sample['solvent_smiles']}: "
              f"{pred:.4f} kcal/mol (target: {sample['target_dG']:.4f})")

    if not args.skip_deepmd:
        print("\n=== DeePMDBaseline fine-tune smoke test ===")
        dp_baseline = DeePMDBaseline(solvent_descriptor_dim=SOLVENT_DESCRIPTOR_DIM, device="cpu")
        history = dp_baseline.fine_tune(dataset, n_epochs=args.epochs, batch_size=min(4, len(dataset)))
        assert len(history) == args.epochs
        pred = dp_baseline.predict(sample["solute_smiles"], sample["solute_x"], sample["solvent_descriptor"])
        print(f"DeePMD post-fine-tune prediction for {sample['solute_smiles']}/{sample['solvent_smiles']}: "
              f"{pred:.4f} kcal/mol (target: {sample['target_dG']:.4f})")

    print("\nSMOKE TEST PASSED")