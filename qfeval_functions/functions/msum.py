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
    r"""Compute the moving (sliding window) sum of a tensor.

    This function calculates the sum of elements within a sliding window of
    size :attr:`span` along the specified dimension. The output tensor has the
    same shape as the input tensor. For positions where the sliding window
    cannot fully cover preceding elements (i.e., the first `span - 1` elements
    along the selected dimension), the result is `nan`.

    Args:
        x (Tensor): The input tensor.
        span (int): The size of the sliding window.
        dim (int, optional): The dimension along which to compute the moving sum.
            Default is -1 (the last dimension).

    Returns:
        Tensor: A tensor of the same shape as the input, containing the moving sums.

    Example:
        >>> import torch
        >>> import qfeval_functions.functions as QF
        >>> x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> QF.msum(x, span=3)
        tensor([nan, nan,  6.,  9., 12.])

        >>> x = torch.tensor([[1.0, 2.0, 3.0, 4.0],
        ...                   [5.0, 6.0, 7.0, 8.0]])
        >>> QF.msum(x, span=2, dim=1)
        tensor([[nan,  3.,  5.,  7.],
                [nan, 11., 13., 15.]])
    """
    return apply_for_axis(lambda x: _msum(x, span), x, dim)
