from typing import Dict, Tuple

import torch

torch.serialization.add_safe_globals([slice])
from e3nn import o3

_SH_MODULE_CACHE: Dict[Tuple[int, torch.dtype], torch.nn.Module] = {}


def get_spherical_harmonics(l: int, x: torch.Tensor) -> torch.Tensor:
    # Cache key MUST include dtype -- o3.SphericalHarmonics has no explicit dtype argument, so
    # its internal constants are fixed at whatever torch.get_default_dtype() is at construction
    # time; a dtype-less cache key means whichever precision touches a given `l` FIRST in the
    # process permanently seeds it there for every later caller, same class of bug as
    # clebsch_gordan_matrix's cache (see stage_1/CLAUDE.md's 2026-08-21 correction).
    key = (l, x.dtype)
    if key not in _SH_MODULE_CACHE:
        prev_default = torch.get_default_dtype()
        torch.set_default_dtype(x.dtype)
        try:
            _SH_MODULE_CACHE[key] = o3.SphericalHarmonics(l, normalize=True, normalization="component")
        finally:
            torch.set_default_dtype(prev_default)
    return _SH_MODULE_CACHE[key](x)