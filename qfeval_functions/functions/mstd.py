import torch

from .mvar import mvar


def mstd(
    x: torch.Tensor, span: int, dim: int = -1, ddof: int = 1
) -> torch.Tensor:
    r"""Compute the moving (sliding window) standard deviation of a tensor.

    This function calculates the standard deviation of elements within a sliding
    window of size :attr:`span` along the specified dimension. The output tensor
    has the same shape as the input tensor. For positions where the sliding window
    cannot fully cover preceding elements (i.e., the first ``span - 1`` elements
    along the selected dimension), the result is ``nan``.

    The moving standard deviation is computed as:

    .. math::
        \text{MSTD}[i] = \sqrt{\frac{1}{N - \text{ddof}}
                         \sum_{j=i-\text{span}+1}^{i} (x[j] - \mu[i])^2}

    where :math:`\mu[i]` is the moving average at position :math:`i` and
    :math:`N` is the number of elements in the window.

    Args:
        x (Tensor):
            The input tensor containing values.
        span (int):
            The size of the sliding window. Must be positive.
        dim (int, optional):
            The dimension along which to compute the moving
            standard deviation. Default is -1 (the last dimension).
        ddof (int, optional):
            Delta degrees of freedom. The divisor used in
            the calculation is ``N - ddof``, where ``N`` represents the number
            of elements in the window. Default is 1 (sample standard deviation).

    Returns:
        Tensor:
            A tensor of the same shape as the input, containing the moving
            standard deviation values. The first ``span - 1`` elements along
            the specified dimension are ``nan``.

    Example:

        >>> # Simple moving standard deviation with window size 3
        >>> x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> QF.mstd(x, span=3)
        tensor([nan, nan, 1., 1., 1.])

        >>> # 2D tensor with moving standard deviation along columns
        >>> x = torch.tensor([[1.0, 2.0, 1.0, 3.0],
        ...                   [4.0, 5.0, 4.0, 6.0],
        ...                   [2.0, 3.0, 2.0, 4.0]])
        >>> QF.mstd(x, span=2, dim=1)
        tensor([[   nan, 0.7071, 0.7071, 1.4142],
                [   nan, 0.7071, 0.7071, 1.4142],
                [   nan, 0.7071, 0.7071, 1.4142]])

        >>> # Population standard deviation (ddof=0)
        >>> x = torch.tensor([1.0, 3.0, 5.0, 7.0])
        >>> QF.mstd(x, span=2, ddof=0)
        tensor([nan, 1., 1., 1.])

        >>> # Sample standard deviation (ddof=1, default)
        >>> QF.mstd(x, span=2, ddof=1)
        tensor([   nan, 1.4142, 1.4142, 1.4142])

    .. seealso::
        - :func:`mvar`: Moving variance function.
        - :func:`ma`: Moving average function.
        - :func:`msum`: Moving sum function.
    """
    result: torch.Tensor = mvar(x, span=span, dim=dim, ddof=ddof) ** 0.5
    return result
