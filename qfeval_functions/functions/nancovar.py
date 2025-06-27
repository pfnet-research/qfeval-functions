import torch

from .nanmean import nanmean
from .nanmulmean import nanmulmean
from .nanones import nanones


def nancovar(
    x: torch.Tensor,
    y: torch.Tensor,
    dim: int = -1,
    keepdim: bool = False,
    ddof: int = 1,
) -> torch.Tensor:
    r"""Compute covariance between two tensors, ignoring NaN values.

    This function calculates the covariance between tensors :attr:`x` and
    :attr:`y` along the specified dimension, excluding any pairs where either
    value is NaN. Covariance measures how much two variables change together.
    Unlike ``numpy.cov``, this function computes element-wise covariance for
    each batch index rather than producing a covariance matrix.

    The function is memory-efficient when broadcasting tensors but may have
    reduced precision (approximately half-precision) when dealing with many
    NaN values due to the prioritization of memory efficiency over numerical
    precision.

    The NaN-aware covariance is computed as:

    .. math::
        \text{Cov}(X, Y) = \text{E}[(X - \mu_X)(Y - \mu_Y)]

    where the expectation is computed only over valid (non-NaN) pairs, and
    :math:`\mu_X`, :math:`\mu_Y` are the means computed over valid values.

    Args:
        x (Tensor):
            The first input tensor.
        y (Tensor):
            The second input tensor. Must be broadcastable with :attr:`x`.
        dim (int, optional):
            The dimension along which to compute the covariance.
            Default is -1 (the last dimension).
        keepdim (bool, optional):
            Whether the output tensor has :attr:`dim`
            retained or not. Default is False.
        ddof (int, optional):
            Delta degrees of freedom. The divisor used in
            the calculation is ``N - ddof``, where ``N`` represents the number
            of valid (non-NaN) pairs. Default is 1.

    Returns:
        Tensor:
            The covariance values computed only over valid (non-NaN) pairs.
            The shape depends on the input dimensions and the :attr:`keepdim`
            parameter.

    Example:

        >>> # Simple covariance with NaN values
        >>> x = torch.tensor([1.0, 2.0, nan, 4.0, 5.0])
        >>> y = torch.tensor([2.0, 4.0, 6.0, nan, 10.0])
        >>> QF.nancovar(x, y, dim=0)
        tensor(8.6250)

        >>> # 2D tensors with NaN values
        >>> x = torch.tensor([[1.0, nan, 3.0, 4.0],
        ...                   [5.0, 6.0, nan, 8.0]])
        >>> y = torch.tensor([[2.0, 4.0, 6.0, nan],
        ...                   [10.0, nan, 14.0, 16.0]])
        >>> QF.nancovar(x, y, dim=1)
        tensor([4.0000, 9.1111])

        >>> # Population covariance (ddof=0)
        >>> x = torch.tensor([1.0, nan, 3.0, 4.0])
        >>> y = torch.tensor([2.0, 4.0, nan, 8.0])
        >>> QF.nancovar(x, y, dim=0, ddof=0)
        tensor(4.5000)

        >>> # With keepdim
        >>> x = torch.tensor([[1.0, 2.0, nan],
        ...                   [4.0, nan, 6.0]])
        >>> y = torch.tensor([[2.0, nan, 6.0],
        ...                   [8.0, 10.0, nan]])
        >>> QF.nancovar(x, y, dim=1, keepdim=True)
        tensor([[nan],
                [nan]])

    .. warning::
        The calculation may have reduced precision (approximately half-precision)
        when dealing with many NaN values due to memory efficiency optimizations.
        For higher precision with many NaNs, consider using CUDA kernels via
        PyTorch JIT compilation.

    .. seealso::
        - :func:`covar`: Covariance without NaN handling.
        - :func:`nancorrel`: NaN-aware correlation function.
        - :func:`nanmean`: NaN-aware mean function.
    """
    # Improve the precision by subtracting their averages first.
    x = x - nanmean(x, dim=dim, keepdim=True)
    y = y - nanmean(y, dim=dim, keepdim=True)
    mx = nanmulmean(x, nanones(y), dim=dim, keepdim=keepdim, _ddof=ddof)
    my = nanmulmean(nanones(x), y, dim=dim, keepdim=keepdim, _ddof=ddof)
    mxy = nanmulmean(x, y, dim=dim, keepdim=keepdim, _ddof=ddof)
    # NOTE: E((X - E[X])(Y - E[Y])) = E(XY) - E(X)E(Y)
    return (mxy - mx * my).to(x)
