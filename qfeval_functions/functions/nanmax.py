import math
import typing

import torch


class NanmaxResult(typing.NamedTuple):
    values: torch.Tensor
    indices: torch.Tensor


def nanmax(x: torch.Tensor, dim: int, keepdim: bool = False) -> NanmaxResult:
    r"""Returns a namedtuple `(values, indices)` where values is the maximum
    value of each row of the `input` tensor in the given dimension `dim`,
    ignoring Not a Numbers (NaNs).
    """

    # 1. Replace NaN -> -inf and name it `a`.
    a = x.nan_to_num(-math.inf, math.inf, -math.inf)

    # 2. Apply `a` to torch.max.
    a_v, a_idx = a.max(dim=dim, keepdim=keepdim)
    # If x has no negative inf values, no conflicts of -inf should happen and
    # the result can be computed by restoring NaN from -inf.
    if not x.isneginf().any():
        return NanmaxResult(a_v.nan_to_num(0, math.inf, math.nan), a_idx)

    # 3. Create b representing -1 == NaN, 0 == -inf, 1 == others.  This enables
    # `(b * -inf)` to have NaN and -inf like `a` has and inf for the others.
    b = (x.detach() * math.inf).nan_to_num(0, 2, 1)
    b_v, b_idx = b.max(dim=dim, keepdim=keepdim)

    # 4. Build the final result.
    use_b = a_v.isneginf()
    return NanmaxResult(
        torch.where(use_b, b_v * -math.inf, a_v),
        torch.where(use_b, b_idx, a_idx),
    )
