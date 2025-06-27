import torch

from .msum import msum


def mvar(
    x: torch.Tensor, span: int, dim: int = -1, ddof: int = 1
) -> torch.Tensor:
    r"""Compute the moving (sliding window) variance of a tensor.

    This function calculates the variance of elements within a sliding window of
    size :attr:`span` along the specified dimension. The output tensor has the
    same shape as the input tensor. For positions where the sliding window
    cannot fully cover preceding elements (i.e., the first ``span - 1`` elements
    along the selected dimension), the result is ``nan``.

    The moving variance is computed using the formula:

    .. math::
        \text{MVAR}[i] = \frac{1}{\text{span} - \text{ddof}} \left(
        \sum_{j=i-\text{span}+1}^{i} x[j]^2 -
        \frac{(\sum_{j=i-\text{span}+1}^{i} x[j])^2}{\text{span}} \right)

    This uses the computational formula for variance that is numerically stable
    and efficient for sliding window calculations.

    Args:
        x (Tensor):
            The input tensor containing values.
        span (int):
            The size of the sliding window. Must be positive.
        dim (int, optional):
            The dimension along which to compute the moving variance.
            Default is -1 (the last dimension).
        ddof (int, optional):
            Delta degrees of freedom. The divisor used in the calculation is
            ``span - ddof``. Use 0 for population variance.
            Default is 1 (sample variance).

    Returns:
        Tensor:
            A tensor of the same shape as the input, containing the moving
            variance values. The first ``span - 1`` elements along the specified
            dimension are ``nan``.

    Example:

        >>> # Simple moving variance with window size 3
        >>> x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> QF.mvar(x, span=3)
        tensor([nan, nan, 1., 1., 1.])

        >>> # 2D tensor with moving variance along columns
        >>> x = torch.tensor([[1.0, 2.0, 1.0, 3.0],
        ...                   [4.0, 5.0, 4.0, 6.0],
        ...                   [2.0, 3.0, 2.0, 4.0]])
        >>> QF.mvar(x, span=2, dim=1)
        tensor([[   nan, 0.5000, 0.5000, 2.0000],
                [   nan, 0.5000, 0.5000, 2.0000],
                [   nan, 0.5000, 0.5000, 2.0000]])

        >>> # Population variance (ddof=0)
        >>> x = torch.tensor([1.0, 3.0, 5.0, 7.0])
        >>> QF.mvar(x, span=2, ddof=0)
        tensor([nan, 1., 1., 1.])

        >>> # Sample variance (ddof=1, default)
        >>> QF.mvar(x, span=2, ddof=1)
        tensor([nan, 2., 2., 2.])

        >>> # Moving variance along rows
        >>> x = torch.tensor([[1.0, 2.0],
        ...                   [3.0, 4.0],
        ...                   [5.0, 6.0]])
        >>> QF.mvar(x, span=2, dim=0)
        tensor([[nan, nan],
                [2., 2.],
                [2., 2.]])

    .. seealso::
        :func:`mstd`: Moving standard deviation function (square root of this).
        :func:`msum`: Moving sum function used in the implementation.
        :func:`ma`: Moving average function.
    """
    numerator = msum(x**2, span, dim) - msum(x, span, dim) ** 2 / span
    result: torch.Tensor = numerator / (span - ddof)
    return result
