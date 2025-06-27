import math
import typing

import torch


def nanskew(
    x: torch.Tensor,
    dim: typing.Union[int, typing.Tuple[int, ...]] = (),
    unbiased: bool = True,
    *,
    keepdim: bool = False,
) -> torch.Tensor:
    r"""Compute the skewness along specified dimensions, ignoring NaN values.

    This function calculates the skewness (third standardized moment) of a
    tensor along the specified dimension(s), excluding NaN values from the
    computation. Skewness measures the asymmetry of a probability distribution
    around its mean, indicating whether the data is concentrated more on one
    side of the distribution.

    The skewness is computed as:

    .. math::
        \text{skew} = \frac{\text{E}[(X - \mu)^3]}{\sigma^3}

    where :math:`\mu` is the mean and :math:`\sigma` is the standard deviation,
    computed only over valid (non-NaN) values. Positive skewness indicates a
    longer tail on the right side of the distribution, while negative skewness
    indicates a longer tail on the left side.

    Args:
        x (Tensor):
            The input tensor containing values.
        dim (int or tuple of ints, optional):
            The dimension(s) along which to
            compute the skewness. If not specified (default is empty tuple),
            computes over all dimensions.
        unbiased (bool, optional):
            If True (default), uses unbiased estimation
            with bias correction. If False, uses biased estimation.
        keepdim (bool, optional):
            Whether the output tensor has :attr:`dim`
            retained or not. Default is False.

    Returns:
        Tensor:
            The skewness values computed only over valid (non-NaN) values.
            The shape depends on the input dimensions and the :attr:`keepdim`
            parameter.

    Example:

        >>> # Simple skewness with some NaN values
        >>> x = torch.tensor([1.0, 2.0, nan, 4.0, 5.0, 10.0])
        >>> QF.nanskew(x, dim=0)
        tensor(1.1846)

        >>> # 2D tensor with skewness along columns
        >>> x = torch.tensor([[1.0, nan, 3.0, 4.0],
        ...                   [2.0, 5.0, nan, 6.0],
        ...                   [3.0, 7.0, 8.0, nan]])
        >>> QF.nanskew(x, dim=1)
        tensor([-0.9352, -1.2933, -1.4579])

        >>> # Skewed distribution
        >>> x = torch.tensor([1.0, 1.0, 1.0, nan, 2.0, 10.0])
        >>> QF.nanskew(x, dim=0)
        tensor(2.1713)

        >>> # With keepdim
        >>> x = torch.tensor([[1.0, 2.0, nan],
        ...                   [4.0, nan, 6.0]])
        >>> QF.nanskew(x, dim=1, keepdim=True)
        tensor([[nan],
                [nan]])

        >>> # Biased estimation
        >>> x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        >>> QF.nanskew(x, dim=0, unbiased=False)
        tensor(0.)

        >>> # Negative skewness (left tail)
        >>> x = torch.tensor([1.0, 8.0, 9.0, nan, 9.0, 10.0])
        >>> QF.nanskew(x, dim=0)
        tensor(-2.0287)

    .. warning::
        Skewness calculations can be sensitive to outliers. A single extreme
        value can significantly affect the skewness measure, especially with
        small sample sizes.

    .. seealso::
        :func:`nankurtosis`: NaN-aware kurtosis function.
        :func:`nanvar`: NaN-aware variance function.
        :func:`nanstd`: NaN-aware standard deviation function.
        :func:`nanmean`: NaN-aware mean function.
    """
    n = (~x.isnan()).to(x).sum(dim=dim, keepdim=True)
    ddof = 1 if unbiased else 0
    x = torch.where(n <= ddof * 2, torch.as_tensor(math.nan).to(x), x)
    m1 = x.nansum(dim=dim, keepdim=True) / n
    m2 = ((x - m1) ** 2).nansum(dim=dim, keepdim=True)
    m3 = ((x - m1) ** 3).nansum(dim=dim, keepdim=True)
    r = (m3 / m2**1.5) * n * (n - ddof).sqrt() / (n - ddof * 2)
    r = r.sum(dim=dim, keepdim=keepdim)
    return typing.cast(torch.Tensor, r)
