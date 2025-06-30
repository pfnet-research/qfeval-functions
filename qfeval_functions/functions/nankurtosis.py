import math
import typing

import torch


def nankurtosis(
    x: torch.Tensor,
    dim: typing.Union[int, typing.Tuple[int, ...]] = (),
    unbiased: bool = True,
    *,
    fisher: bool = True,
    keepdim: bool = False,
) -> torch.Tensor:
    r"""Compute the kurtosis along specified dimensions, ignoring NaN values.

    This function calculates the kurtosis (fourth standardized moment) of a
    tensor along the specified dimension(s), excluding NaN values from the
    computation. Kurtosis measures the "tailedness" of a probability distribution,
    indicating how much of the data is concentrated in the tails versus the center.

    The kurtosis is computed as:

    .. math::
        \kappa = \frac{\text{E}[(X - \mu)^4]}{\sigma^4}

    where :math:`\mu` is the mean and :math:`\sigma` is the standard deviation,
    computed only over valid (non-NaN) values. When :attr:`fisher` is True
    (default), Fisher's definition is used where normal distribution has
    kurtosis 0. When False, Pearson's definition is used where normal
    distribution has kurtosis 3.

    Args:
        x (Tensor):
            The input tensor containing values.
        dim (int or tuple of ints, optional):
            The dimension(s) along which to compute the kurtosis. If not
            specified (default is empty tuple), computes over all dimensions.
        unbiased (bool, optional):
            If True (default), uses unbiased estimation
            with bias correction. If False, uses biased estimation.
        fisher (bool, optional):
            If True (default), uses Fisher's definition (normal distribution
            has kurtosis 0). If False, uses Pearson's definition (normal
            distribution has kurtosis 3).
        keepdim (bool, optional):
            Whether the output tensor has :attr:`dim` retained or not.
            Default is False.

    Returns:
        Tensor:
            The kurtosis values computed only over valid (non-NaN) values.
            The shape depends on the input dimensions and the :attr:`keepdim`
            parameter.

    Example:

        >>> # Simple kurtosis with some NaN values
        >>> x = torch.tensor([1.0, 2.0, nan, 4.0, 5.0, 6.0])
        >>> QF.nankurtosis(x, dim=0)
        tensor(-1.9632)

        >>> # 2D tensor with kurtosis along columns
        >>> x = torch.tensor([[1.0, nan, 3.0, 4.0],
        ...                   [2.0, 5.0, nan, 6.0],
        ...                   [3.0, 7.0, 8.0, nan]])
        >>> QF.nankurtosis(x, dim=1)
        tensor([-inf, inf, nan])

        >>> # Pearson's definition (fisher=False)
        >>> x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> QF.nankurtosis(x, dim=0, fisher=False)
        tensor(6.8000)

        >>> # With keepdim
        >>> x = torch.tensor([[1.0, 2.0, nan],
        ...                   [4.0, nan, 6.0]])
        >>> QF.nankurtosis(x, dim=1, keepdim=True)
        tensor([[nan],
                [nan]])

        >>> # Biased estimation
        >>> x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        >>> QF.nankurtosis(x, dim=0, unbiased=False)
        tensor(-1.2686)

    .. seealso::
        :func:`nanskew`: NaN-aware skewness function.
        :func:`nanvar`: NaN-aware variance function.
        :func:`nanstd`: NaN-aware standard deviation function.
    """
    n = (~x.isnan()).to(x).sum(dim=dim, keepdim=True)
    ddof = 1 if unbiased else 0
    x = torch.where(n <= ddof * 2, torch.as_tensor(math.nan).to(x), x)
    m1 = x.nansum(dim=dim, keepdim=True) / n
    m2 = ((x - m1) ** 2).nansum(dim=dim, keepdim=True)
    m4 = ((x - m1) ** 4).nansum(dim=dim, keepdim=True)
    r = (m4 / m2**2) * (n + ddof) * n - (3 if fisher else 0) * (n - ddof)
    r = r * (n - ddof) / (n - ddof * 2) / (n - ddof * 3)
    r = r.sum(dim=dim, keepdim=keepdim)
    return typing.cast(torch.Tensor, r)
