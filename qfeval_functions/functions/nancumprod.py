import math

import torch


def nancumprod(x: torch.Tensor, dim: int) -> torch.Tensor:
    r"""Compute the cumulative product along a dimension, treating NaN as 1.

    This function calculates the cumulative product of elements along the
    specified dimension, where NaN values are treated as 1 (multiplicative
    identity) for the purpose of computing the cumulative product. However,
    positions with NaN values in the original tensor remain NaN in the output.

    Args:
        x (Tensor):
            The input tensor containing values.
        dim (int):
            The dimension along which to compute the cumulative product.

    Returns:
        Tensor:
            A tensor of the same shape as the input, containing the
            cumulative product values. Original NaN positions remain NaN.

    Example:

        >>> # Simple cumulative product with NaN
        >>> x = torch.tensor([2.0, nan, 3.0, 4.0])
        >>> QF.nancumprod(x, dim=0)
        tensor([ 2., nan,  6., 24.])

        >>> # 2D tensor along columns
        >>> x = torch.tensor([[1.0, 2.0, nan, 4.0],
        ...                   [2.0, nan, 3.0, 5.0]])
        >>> QF.nancumprod(x, dim=1)
        tensor([[ 1.,  2., nan,  8.],
                [ 2., nan,  6., 30.]])

        >>> # Cumulative product along rows
        >>> x = torch.tensor([[2.0, 3.0],
        ...                   [nan, 4.0],
        ...                   [5.0, 2.0]])
        >>> QF.nancumprod(x, dim=0)
        tensor([[ 2.,  3.],
                [nan, 12.],
                [10., 24.]])

        >>> # Multiple NaNs
        >>> x = torch.tensor([1.0, nan, nan, 2.0, 3.0])
        >>> QF.nancumprod(x, dim=0)
        tensor([1., nan, nan, 2., 6.])

    .. seealso::
        - :func:`nancumsum`: NaN-aware cumulative sum.
        - :func:`torch.cumprod`: Standard cumulative product (NaN propagates).
        - :func:`torch.nan_to_num`: Convert NaN values to specified numbers.
    """
    return torch.where(
        x.isnan(),
        torch.as_tensor(math.nan, dtype=x.dtype, device=x.device),
        x.nan_to_num(1.0).cumprod(dim=dim),
    )
