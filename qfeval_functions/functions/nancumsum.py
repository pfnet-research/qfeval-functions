import math

import torch


def nancumsum(x: torch.Tensor, dim: int) -> torch.Tensor:
    r"""Compute the cumulative sum along a dimension, treating NaN as 0.

    This function calculates the cumulative sum of elements along the
    specified dimension, where NaN values are treated as 0 (additive
    identity) for the purpose of computing the cumulative sum. However,
    positions with NaN values in the original tensor remain NaN in the output.

    Args:
        x (Tensor):
            The input tensor containing values.
        dim (int):
            The dimension along which to compute the cumulative sum.

    Returns:
        Tensor:
            A tensor of the same shape as the input, containing the
            cumulative sum values. Original NaN positions remain NaN.

    Example:

        >>> # Simple cumulative sum with NaN
        >>> x = torch.tensor([1.0, nan, 3.0, 4.0])
        >>> QF.nancumsum(x, dim=0)
        tensor([1., nan, 4., 8.])

        >>> # 2D tensor along columns
        >>> x = torch.tensor([[1.0, 2.0, nan, 4.0],
        ...                   [2.0, nan, 3.0, 5.0]])
        >>> QF.nancumsum(x, dim=1)
        tensor([[ 1.,  3., nan,  7.],
                [ 2., nan,  5., 10.]])

        >>> # Cumulative sum along rows
        >>> x = torch.tensor([[1.0, 2.0],
        ...                   [nan, 3.0],
        ...                   [4.0, 5.0]])
        >>> QF.nancumsum(x, dim=0)
        tensor([[ 1.,  2.],
                [nan,  5.],
                [ 5., 10.]])

        >>> # Multiple NaNs
        >>> x = torch.tensor([1.0, nan, nan, 2.0, 3.0])
        >>> QF.nancumsum(x, dim=0)
        tensor([1., nan, nan, 3., 6.])

        >>> # All NaNs
        >>> x = torch.tensor([nan, nan, nan])
        >>> QF.nancumsum(x, dim=0)
        tensor([nan, nan, nan])

    .. note::
        This function treats NaN values as the additive identity (0)
        for computation purposes, allowing the cumulative sum to continue
        past NaN values. However, the original NaN positions are preserved
        in the output, maintaining data integrity while enabling meaningful
        cumulative operations on partially missing data.

    .. seealso::
        - :func:`nancumprod`: NaN-aware cumulative product.
        - :func:`torch.cumsum`: Standard cumulative sum (NaN propagates).
        - :func:`torch.nan_to_num`: Convert NaN values to specified numbers.
    """
    return torch.where(
        x.isnan(),
        torch.as_tensor(math.nan, dtype=x.dtype, device=x.device),
        x.nan_to_num().cumsum(dim=dim),
    )
