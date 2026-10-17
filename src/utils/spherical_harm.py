from typing import Dict, Iterable, Tuple

import torch

torch.serialization.add_safe_globals([slice])
from e3nn import o3

_SH_MODULE_CACHE: Dict[Tuple[int, torch.dtype], torch.nn.Module] = {}


def _build_sh_module(l: int, dtype: torch.dtype) -> torch.nn.Module:
    prev_default = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        return o3.SphericalHarmonics(l, normalize=True, normalization="component")
    finally:
        torch.set_default_dtype(prev_default)


def warm_spherical_harmonics_cache(
    degrees: Iterable[int], dtype: torch.dtype = None,
) -> None:
    dtype = torch.get_default_dtype() if dtype is None else dtype
    for l in degrees:
        key = (l, dtype)
        if key not in _SH_MODULE_CACHE:
            _SH_MODULE_CACHE[key] = _build_sh_module(l, dtype)


def get_spherical_harmonics(l: int, x: torch.Tensor) -> torch.Tensor:
    key = (l, x.dtype)
    if key not in _SH_MODULE_CACHE:
        _SH_MODULE_CACHE[key] = _build_sh_module(l, x.dtype)
    return _SH_MODULE_CACHE[key](x)