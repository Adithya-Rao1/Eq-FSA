import math
import sys

import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.QED import qed

_MOL_FEAT_NAMES: list[str] = [
    "LogP", "MolWt", "QED", "HBA", "HBD",
    # Structural / topological
    "BalabanJ", "BertzCT",
    "Chi0", "Chi1",
    "Chi0n", "Chi1n", "Chi2n", "Chi3n", "Chi4n",
    "Chi0v", "Chi1v", "Chi2v", "Chi3v", "Chi4v",
    "HallKierAlpha",
    "HeavyAtomCount", "HeavyAtomMolWt",
    "Kappa1", "Kappa2", "Kappa3",
    "MolMR",
    "NHOHCount", "NOCount",
    "NumAliphaticCarbocycles", "NumAliphaticHeterocycles", "NumAliphaticRings",
    "NumAromaticCarbocycles", "NumAromaticHeterocycles", "NumAromaticRings",
    "NumHeteroatoms", "NumRadicalElectrons", "NumRotatableBonds",
    "NumSaturatedRings", "NumValenceElectrons",
    "TPSA",
    # Electronic
    "MaxAbsEStateIndex", "MinAbsEStateIndex",
    "MaxEStateIndex", "MinEStateIndex",
    "MaxPartialCharge", "MinPartialCharge",
    # Surface-area (VSA) families
    *[f"VSA_EState{i}" for i in range(1, 11)],
    *[f"PEOE_VSA{i}" for i in range(1, 15)],
    *[f"SMR_VSA{i}" for i in range(1, 11)],
    *[f"SlogP_VSA{i}" for i in range(1, 13)],
]

_MOL_FEAT_DIM: int = len(_MOL_FEAT_NAMES)
_ADMET_FEAT_DIM: int = _MOL_FEAT_DIM

_DUMMY_SMILES: str = "O"
_MASK_VALUE: float = 0.0

def _build_feature_fns():
    fns = [
        Descriptors.MolLogP,
        Descriptors.MolWt,
        qed,
        lambda m: float(rdMolDescriptors.CalcNumHBA(m)),
        lambda m: float(rdMolDescriptors.CalcNumHBD(m)),
        Descriptors.BalabanJ,
        Descriptors.BertzCT,
        Descriptors.Chi0,
        Descriptors.Chi1,
        Descriptors.Chi0n,
        Descriptors.Chi1n,
        Descriptors.Chi2n,
        Descriptors.Chi3n,
        Descriptors.Chi4n,
        Descriptors.Chi0v,
        Descriptors.Chi1v,
        Descriptors.Chi2v,
        Descriptors.Chi3v,
        Descriptors.Chi4v,
        Descriptors.HallKierAlpha,
        lambda m: float(Descriptors.HeavyAtomCount(m)),
        Descriptors.HeavyAtomMolWt,
        Descriptors.Kappa1,
        Descriptors.Kappa2,
        Descriptors.Kappa3,
        Descriptors.MolMR,
        lambda m: float(Descriptors.NHOHCount(m)),
        lambda m: float(Descriptors.NOCount(m)),
        lambda m: float(rdMolDescriptors.CalcNumAliphaticCarbocycles(m)),
        lambda m: float(rdMolDescriptors.CalcNumAliphaticHeterocycles(m)),
        lambda m: float(rdMolDescriptors.CalcNumAliphaticRings(m)),
        lambda m: float(rdMolDescriptors.CalcNumAromaticCarbocycles(m)),
        lambda m: float(rdMolDescriptors.CalcNumAromaticHeterocycles(m)),
        lambda m: float(rdMolDescriptors.CalcNumAromaticRings(m)),
        lambda m: float(rdMolDescriptors.CalcNumHeteroatoms(m)),
        lambda m: float(Descriptors.NumRadicalElectrons(m)),
        lambda m: float(rdMolDescriptors.CalcNumRotatableBonds(m)),
        lambda m: float(rdMolDescriptors.CalcNumSaturatedRings(m)),
        lambda m: float(Descriptors.NumValenceElectrons(m)),
        Descriptors.TPSA,
        Descriptors.MaxAbsEStateIndex,
        Descriptors.MinAbsEStateIndex,
        Descriptors.MaxEStateIndex,
        Descriptors.MinEStateIndex,
        Descriptors.MaxPartialCharge,
        Descriptors.MinPartialCharge,
    ]
    fns += [(lambda m, i=i: getattr(Descriptors, f"VSA_EState{i}")(m)) for i in range(1, 11)]
    fns += [(lambda m, i=i: getattr(Descriptors, f"PEOE_VSA{i}")(m)) for i in range(1, 15)]
    fns += [(lambda m, i=i: getattr(Descriptors, f"SMR_VSA{i}")(m)) for i in range(1, 11)]
    fns += [(lambda m, i=i: getattr(Descriptors, f"SlogP_VSA{i}")(m)) for i in range(1, 13)]
    return fns

_MOL_FEAT_FNS = _build_feature_fns()
assert len(_MOL_FEAT_FNS) == _MOL_FEAT_DIM, (
    f"Feature fn/name length mismatch: {len(_MOL_FEAT_FNS)} vs {_MOL_FEAT_DIM}"
)

def calc_rdkit_mol_props(mol) -> list[float]:
    zero_vec = [_MASK_VALUE] * _MOL_FEAT_DIM
    if mol is None:
        return zero_vec

    out = [0.0] * _MOL_FEAT_DIM
    for i, (name, fn) in enumerate(zip(_MOL_FEAT_NAMES, _MOL_FEAT_FNS)):
        try:
            val = fn(mol)
            if not math.isfinite(val) or math.isnan(val):
                val = _MASK_VALUE
        except Exception as e:
            print(f"  [WARN] {name} failed: {e} — masking with {_MASK_VALUE}", file=sys.stderr)
            val = _MASK_VALUE
        out[i] = val

    return out

_H_MASS = 1.008
_O_MASS = 15.999
_H_VALENCE_ELECTRONS = 1
_O_VALENCE_ELECTRONS = 6

def mol_props_from_atom_counts(n_h: torch.Tensor, n_o: torch.Tensor) -> torch.Tensor:
    n_h = n_h.float() if torch.is_tensor(n_h) else torch.tensor(float(n_h))
    n_o = n_o.float() if torch.is_tensor(n_o) else torch.tensor(float(n_o))
    n_h, n_o = torch.broadcast_tensors(n_h, n_o)

    vec = torch.zeros(*n_h.shape, _MOL_FEAT_DIM, dtype=n_h.dtype, device=n_h.device)
    vec[..., _MOL_FEAT_NAMES.index("MolWt")] = n_h * _H_MASS + n_o * _O_MASS
    vec[..., _MOL_FEAT_NAMES.index("HeavyAtomCount")] = n_o
    vec[..., _MOL_FEAT_NAMES.index("HeavyAtomMolWt")] = n_o * _O_MASS
    vec[..., _MOL_FEAT_NAMES.index("NumHeteroatoms")] = n_o
    vec[..., _MOL_FEAT_NAMES.index("NumValenceElectrons")] = (
        n_h * _H_VALENCE_ELECTRONS + n_o * _O_VALENCE_ELECTRONS
    )
    return vec


def calc_rdkit_mol_props_smiles(smiles: str | None) -> list[float]:
    zero_vec = [0.0] * _MOL_FEAT_DIM
    if smiles is None:
        return zero_vec
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return zero_vec
        return calc_rdkit_mol_props(mol)
    except Exception as e:
        print(f"  [WARN] calc_rdkit_mol_props_smiles failed: {e} — using zero-vector", file=sys.stderr)
        return zero_vec

def _pad_dim1(t, n_max):
    if t.shape[1] >= n_max:
        return t[:, :n_max]

    pad = torch.zeros(t.shape[0], n_max - t.shape[1], *t.shape[2:], dtype=t.dtype, device=t.device,)
    return torch.cat([t, pad], dim=1)