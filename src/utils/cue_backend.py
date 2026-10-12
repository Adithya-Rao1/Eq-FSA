from functools import lru_cache
from typing import Dict, Sequence, Tuple

import torch

_CUE_SH_MODULE_CACHE: Dict[Tuple[Tuple[int, ...], str], "torch.nn.Module"] = {}


@lru_cache(maxsize=1)
def cuequivariance_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import cuequivariance  # noqa: F401
        import cuequivariance_torch  # noqa: F401
    except ImportError:
        return False
    return True


def _cue():
    import cuequivariance as cue
    return cue


def _cuet():
    import cuequivariance_torch as cuet
    return cuet


def so3_irreps(mults: Dict[int, int]) -> str:
    return "+".join(f"{mults[l]}x{l}" for l in sorted(mults))


def spherical_harmonics_many(ls: Sequence[int], x: torch.Tensor) -> Dict[int, torch.Tensor]:
    cuet = _cuet()
    ls_sorted = sorted(set(ls))
    key = (tuple(ls_sorted), str(x.device))
    if key not in _CUE_SH_MODULE_CACHE:
        _CUE_SH_MODULE_CACHE[key] = cuet.SphericalHarmonics(ls_sorted, device=x.device)
    out = _CUE_SH_MODULE_CACHE[key](x)
    result: Dict[int, torch.Tensor] = {}
    start = 0
    for l in ls_sorted:
        width = 2 * l + 1
        result[l] = out[..., start:start + width]
        start += width
    return result


def build_channelwise_tp(
    irreps_in1: str,
    irreps_in2: str,
    irreps_out: str,
    device: torch.device,
    dtype: torch.dtype,
):
    cue = _cue()
    cuet = _cuet()
    i1 = cue.Irreps("SO3", irreps_in1)
    i2 = cue.Irreps("SO3", irreps_in2)
    out = cue.Irreps("SO3", irreps_out)
    return cuet.ChannelWiseTensorProduct(
        i1, i2, filter_irreps_out=out, layout=cue.mul_ir,
        shared_weights=False, internal_weights=False,
        device=device, dtype=dtype,
    )
