from functools import lru_cache
from typing import Dict, Sequence, Tuple

import torch

_CUE_SH_MODULE_CACHE: Dict[Tuple[Tuple[int, ...], str], "torch.nn.Module"] = {}
_E3NN_TO_CUE_BASIS_CACHE: Dict[Tuple[int, str], torch.Tensor] = {}


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


@torch._dynamo.disable()
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


@torch._dynamo.disable()
def e3nn_to_cue_basis(l: int, device: torch.device, dtype: torch.dtype, n_probe: int = 4000) -> torch.Tensor:
    key = (l, str(device))
    if key in _E3NN_TO_CUE_BASIS_CACHE:
        return _E3NN_TO_CUE_BASIS_CACHE[key].to(dtype=dtype)

    from utils.spherical_harm import get_spherical_harmonics

    probe_dtype = torch.double
    x = torch.randn(n_probe, 3, device=device, dtype=probe_dtype)
    y_e3nn = get_spherical_harmonics(l, x)
    y_cue = spherical_harmonics_many([l], x)[l].to(probe_dtype)
    A, *_ = torch.linalg.lstsq(y_e3nn, y_cue)

    width = 2 * l + 1
    resid = (y_e3nn @ A - y_cue).abs().max().item()
    orth_err = (A.T @ A - torch.eye(width, dtype=A.dtype, device=A.device)).abs().max().item()
    if resid > 1e-3 or orth_err > 1e-3:
        raise RuntimeError(
            f"No clean e3nn<->cuEquivariance change of basis for degree {l}"
            f"(fit residual={resid:.2e}, orthogonality error={orth_err:.2e})"
        )

    _E3NN_TO_CUE_BASIS_CACHE[key] = A.detach()
    return A.to(dtype=dtype)


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
