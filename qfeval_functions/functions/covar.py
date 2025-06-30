import torch

from .mulmean import mulmean


def covar(
    x: torch.Tensor,
    y: torch.Tensor,
    dim: int = -1,
    keepdim: bool = False,
    ddof: int = 1,
) -> torch.Tensor:
    r"""Compute covariance between two tensors along a specified dimension.

    This function calculates the covariance between tensors :attr:`x` and
    :attr:`y` along the specified dimension. Covariance measures how much
    two variables change together. Unlike ``numpy.cov``, this function
    computes element-wise covariance for each batch index rather than
    producing a covariance matrix.

    The covariance is computed as:

    .. math::
        \text{Cov}(X, Y) = \frac{1}{N - \text{ddof}}
        \sum_{i=1}^{N} (X_i - \bar{X})(Y_i - \bar{Y})

    where :math:`\bar{X}` and :math:`\bar{Y}` are the means of :attr:`x` and
    :attr:`y` respectively along the specified dimension, and :math:`N` is the
    number of elements along that dimension.

    The function is memory-efficient when broadcasting tensors. For example,
    when operating on tensors with shapes ``(N, 1, D)`` and ``(1, M, D)``,
    the space complexity remains ``O(ND + MD)`` instead of ``O(NMD)``.

    Args:
        x (Tensor):
            The first input tensor.
        y (Tensor):
            The second input tensor. Must be broadcastable with :attr:`x`.
        dim (int, optional):
            The dimension along which to compute the covariance.
            Default is -1 (the last dimension).
        keepdim (bool, optional):
            Whether the output tensor has :attr:`dim` retained or not.
            Default is False.
        ddof (int, optional):
            Delta degrees of freedom. The divisor used in the calculation is
            ``N - ddof``, where ``N`` represents the number of elements.
            Default is 1.

    Returns:
        Tensor:
            The covariance values. The shape depends on the input dimensions
            and the :attr:`keepdim` parameter.

    Example:
        >>> x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> y = torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0])
        >>> QF.covar(x, y, dim=0)
        tensor(5.)

        >>> x = torch.tensor([[1.0, 2.0, 3.0],
        ...                   [4.0, 5.0, 6.0]])
        >>> y = torch.tensor([[2.0, 4.0, 5.0],
        ...                   [8.0, 10.0, 12.0]])
        >>> QF.covar(x, y, dim=1)
        tensor([1.5000, 2.0000])
        >>> QF.covar(x, y, dim=1, keepdim=True, ddof=0)
        tensor([[1.0000],
                [1.3333]])
    """
    x = x - x.mean(dim=dim, keepdim=True)
    y = y - y.mean(dim=dim, keepdim=True)
    return mulmean(x, y, dim=dim, keepdim=keepdim, _ddof=ddof)
