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
    r"""Returns the moving sum of the given tensor.

    Computes the sum of elements within a sliding window of size :attr:`span` along
    the specified dimension. For each position in the output tensor, the value is the
    sum of :attr:`span` consecutive elements ending at that position.

    For a 1-D tensor with ``dim=0``, the output is computed as::

        output[i] = sum(x[i-span+1:i+1])  # for i >= span-1
        output[i] = NaN                    # for i < span-1

    For a 2-D tensor, the operation is applied along the specified dimension::

        output[i][j] = sum(x[i-span+1:i+1][j])  # if dim == 0, i >= span-1
        output[i][j] = sum(x[i][j-span+1:j+1])  # if dim == 1, j >= span-1

    The first ``span-1`` elements along the specified dimension are set to NaN
    because there are insufficient elements to compute the sum.

    Note:
        This function handles NaN values by propagating them. If any value within
        the window is NaN, the output for that window will be NaN.

    Args:
        x (Tensor): the input tensor
        span (int): the size of the moving window. Must be positive.
        dim (int): the dimension along which to compute the moving sum.
            Default: -1 (last dimension)

    Returns:
        Tensor: A tensor of the same shape as :attr:`x` containing the moving sums.

    Example::

        >>> import torch
        >>> import qfeval_functions.functions as QF
        >>> x = torch.tensor([1., 2., 3., 4., 5.])
        >>> QF.msum(x, span=3, dim=0)
        tensor([nan, nan,  6.,  9., 12.])

        >>> # 2D example
        >>> x = torch.tensor([[1., 2., 3., 4.],
        ...                   [5., 6., 7., 8.]])
        >>> QF.msum(x, span=2, dim=1)
        tensor([[nan,  3.,  5.,  7.],
                [nan, 11., 13., 15.]])

        >>> # With NaN values
        >>> x = torch.tensor([1., 2., float('nan'), 4., 5.])
        >>> QF.msum(x, span=2, dim=0)
        tensor([nan, 3., nan, nan, 9.])
    """
    return apply_for_axis(lambda x: _msum(x, span), x, dim)
