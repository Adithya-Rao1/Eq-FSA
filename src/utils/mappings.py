import torch

ATOM_TYPE_MAP = {
    "H": 1,
    "C": 2,
    "N": 3,
    "O": 4,
    "S": 5,
    "P": 6,
    "F": 7,
    "Cl": 8,
    "Br": 9,
    "I": 10,
    "B": 11,
    "Si": 12,
    "Se": 13,
    "Na": 14,
    "K": 15,
    "Mg": 16,
    "Ca": 17,
    "Zn": 18,
    "Fe": 19,
    "Cu": 20,
    "Mn": 21,
    "Li": 22,
}

_MASS_LOOKUP = torch.tensor(
    [
        0.0,        # 0 = padding
        1.008,      # 1 = H
        12.011,     # 2 = C
        14.007,     # 3 = N
        15.999,     # 4 = O
        32.06,      # 5 = S
        30.974,     # 6 = P
        18.998,     # 7 = F
        35.45,      # 8 = Cl
        79.904,     # 9 = Br
        126.904,    # 10 = I
        10.81,      # 11 = B
        28.085,     # 12 = Si
        78.971,     # 13 = Se
        22.990,     # 14 = Na
        39.098,     # 15 = K
        24.305,     # 16 = Mg
        40.078,     # 17 = Ca
        65.38,      # 18 = Zn
        55.845,     # 19 = Fe
        63.546,     # 20 = Cu
        54.938,     # 21 = Mn
        6.94,       # 22 = Li
    ],
    dtype=torch.float32,
)

def atom_types_to_masses(atom_types: torch.Tensor) -> torch.Tensor:
    return _MASS_LOOKUP.to(atom_types.device)[atom_types.long()]
