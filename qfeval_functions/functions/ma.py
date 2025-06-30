import torch

from .msum import msum


def ma(x: torch.Tensor, span: int, dim: int = -1) -> torch.Tensor:
    r"""Compute the moving (sliding window) average of a tensor.

    This function calculates the average of elements within a sliding window of
    size :attr:`span` along the specified dimension. The output tensor has the
    same shape as the input tensor. For positions where the sliding window
    cannot fully cover preceding elements (i.e., the first ``span - 1``
    elements along the selected dimension), the result is ``nan``.

    The moving average is computed as:

    .. math::
        \text{MA}[i] = \frac{1}{\text{span}} \sum_{j=i-\text{span}+1}^{i} x[j]

    This is a simple moving average (SMA) where all values in the window have
    equal weight.

    Args:
        x (Tensor):
            The input tensor containing values to be averaged.
        span (int):
            The size of the sliding window. Must be positive.
        dim (int, optional):
            The dimension along which to compute the moving average.
            Default is -1 (the last dimension).

    Returns:
        Tensor:
            A tensor of the same shape as the input, containing the moving
            averages. The first ``span - 1`` elements along the specified
            dimension are ``nan``.

    Example:

        >>> # Simple moving average with window size 3
        >>> x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> QF.ma(x, span=3)
        tensor([nan, nan, 2., 3., 4.])

        >>> # 2D tensor with moving average along columns
        >>> x = torch.tensor([[1.0, 2.0, 3.0, 4.0],
        ...                   [5.0, 6.0, 7.0, 8.0],
        ...                   [9.0, 10.0, 11.0, 12.0]])
        >>> QF.ma(x, span=2, dim=1)
        tensor([[    nan,  1.5000,  2.5000,  3.5000],
                [    nan,  5.5000,  6.5000,  7.5000],
                [    nan,  9.5000, 10.5000, 11.5000]])

        >>> # Moving average along the first dimension
        >>> x = torch.tensor([[1.0, 2.0],
        ...                   [3.0, 4.0],
        ...                   [5.0, 6.0],
        ...                   [7.0, 8.0]])
        >>> QF.ma(x, span=3, dim=0)
        tensor([[nan, nan],
                [nan, nan],
                [3., 4.],
                [5., 6.]])

    .. seealso::
        :func:`msum`: The underlying moving sum function.
        :func:`ema`: Exponential moving average for weighted averaging.
    """
    return msum(x, span, dim) / span
