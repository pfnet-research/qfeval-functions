import math

import torch

from .apply_for_axis import apply_for_axis
from .rcumsum import rcumsum


def _msum(x: torch.Tensor, span: int) -> torch.Tensor:
    """Returns the moving sum of the given tensor `x`, whose shape is
    `(B, N)`, along the 2nd dimension."""

    # 1. Reshape the target dimension into `(*, span)` with prepending NaNs.
    pad_len = span * 2 - x.shape[1] % span
    x = torch.nn.functional.pad(x, (pad_len, 0), value=math.nan)
    x = x.reshape((x.shape[0], x.shape[1] // span, span))

    # 2. Calculate `sum(x[:, i:i+span], dim=1)` by splitting it into
    # `sum(x[:, i:s], dim=1)+sum(x[:, s:i+span], dim=1)` where `s` is a
    # multiple of `span`.  They can be calculated by cumsum and rcumsum.
    a, b = x.cumsum(dim=2), rcumsum(x, dim=2)
    x = torch.cat((a[:, 1:, :-1] + b[:, :-1, 1:], a[:, 1:, -1:]), dim=2)
    return x.flatten(start_dim=1)[:, pad_len - span :]


def msum(x: torch.Tensor, span: int, dim: int = -1) -> torch.Tensor:
    r"""Returns the moving sum of the given tensor."""
    return apply_for_axis(lambda x: _msum(x, span), x, dim)
