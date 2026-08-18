from typing import Dict

import torch

torch.serialization.add_safe_globals([slice])
from e3nn import o3

_SH_MODULE_CACHE: Dict[int, torch.nn.Module] = {}


def get_spherical_harmonics(l: int, x: torch.Tensor) -> torch.Tensor:
    if l not in _SH_MODULE_CACHE:
        _SH_MODULE_CACHE[l] = o3.SphericalHarmonics(l, normalize=True, normalization="component")
    return _SH_MODULE_CACHE[l](x)