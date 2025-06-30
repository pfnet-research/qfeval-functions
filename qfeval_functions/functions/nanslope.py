import math
import typing

import torch

from .nanmean import nanmean
from .nanmulmean import nanmulmean
from .nansum import nansum


def nanslope(
    x: torch.Tensor,
    y: torch.Tensor,
    dim: typing.Union[int, typing.Tuple[int, ...]] = (),
    keepdim: bool = False,
) -> torch.Tensor:
    r"""Compute the slope of simple linear regression between two tensors,
    ignoring NaN values.

    This function calculates the slope (beta coefficient) of the linear regression
    line that best fits the relationship between :attr:`x` and :attr:`y`, excluding
    NaN values from the computation. The slope represents the rate of change in
    :attr:`y` per unit change in :attr:`x`.

    The function implements the standard ordinary least squares (OLS) formula for
    the slope coefficient in simple linear regression:

    .. math::
        \beta = \frac{\text{Cov}(X, Y)}{\text{Var}(X)} =
        \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}

    where :math:`\bar{x}` and :math:`\bar{y}` are the means of :attr:`x` and
    :attr:`y` respectively, computed only over valid (non-NaN) pairs.

    If either :attr:`x` or :attr:`y` has a NaN at a given position, that pair is
    excluded from all calculations including mean computation and slope estimation.

    Args:
        x (Tensor):
            The independent variable tensor (predictor).
        y (Tensor):
            The dependent variable tensor (response). Must be broadcastable
            with :attr:`x`.
        dim (int or tuple of ints, optional):
            The dimension(s) along which to
            compute the slope. If not specified (default is empty tuple), the
            slope is computed over all dimensions.
        keepdim (bool, optional):
            Whether the output tensor has :attr:`dim`
            retained or not. Default is False.

    Returns:
        Tensor:
            The slope coefficients computed only over valid (non-NaN) pairs.
            If there are insufficient valid pairs or if the variance of :attr:`x`
            is zero, the result may contain NaN values. The shape depends on the
            input dimensions, :attr:`dim`, and :attr:`keepdim` parameters.

    Example:

        >>> # Simple linear relationship with some NaN values
        >>> x = torch.tensor([1.0, 2.0, nan, 4.0, 5.0])
        >>> y = torch.tensor([2.0, 4.0, 6.0, nan, 10.0])
        >>> QF.nanslope(x, y)
        tensor(2.)

        >>> # 2D tensors with slopes along rows
        >>> x = torch.tensor([[1.0, 2.0, 3.0],
        ...                   [1.0, nan, 3.0]])
        >>> y = torch.tensor([[2.0, 4.0, 6.0],
        ...                   [3.0, 5.0, nan]])
        >>> QF.nanslope(x, y, dim=1)
        tensor([2., nan])

        >>> # Perfect linear relationship
        >>> x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        >>> y = torch.tensor([3.0, 5.0, 7.0, 9.0])  # y = 2x + 1
        >>> QF.nanslope(x, y)
        tensor(2.)

        >>> # With keepdim
        >>> x = torch.tensor([[1.0, nan, 3.0],
        ...                   [2.0, 4.0, 6.0]])
        >>> y = torch.tensor([[2.0, 4.0, nan],
        ...                   [1.0, 2.0, 3.0]])
        >>> QF.nanslope(x, y, dim=1, keepdim=True)
        tensor([[   nan],
                [0.5000]])

        >>> # Broadcasting example
        >>> x = torch.tensor([[1.0], [2.0], [3.0]])
        >>> y = torch.tensor([2.0, 4.0, 6.0])
        >>> QF.nanslope(x, y)
        tensor(0.)

        >>> # Zero variance in x (undefined slope)
        >>> x = torch.tensor([2.0, 2.0, 2.0])
        >>> y = torch.tensor([1.0, 3.0, 5.0])
        >>> QF.nanslope(x, y)
        tensor(nan)

    .. warning::
        The slope calculation requires at least 2 valid (non-NaN) pairs of
        observations. With fewer valid pairs, the result will be NaN. Additionally,
        if all valid :attr:`x` values are identical (zero variance), the slope
        is mathematically undefined.

    .. seealso::
        :func:`nancorrel`: NaN-aware correlation coefficient computation.
        :func:`nancovar`: NaN-aware covariance computation.
        :func:`nanmean`: NaN-aware mean computation.
        :func:`nanmulmean`: NaN-aware element-wise product mean.
    """
    isnan = x.isnan() | y.isnan()
    x = torch.where(isnan, torch.as_tensor(math.nan).to(x), x)
    y = torch.where(isnan, torch.as_tensor(math.nan).to(y), y)
    ax = x - nanmean(x, dim=dim, keepdim=True)
    ay = y - nanmean(y, dim=dim, keepdim=True)
    axy = nanmulmean(ax, ay, dim=dim, keepdim=True)
    ax2 = nanmean(ax**2, dim=dim, keepdim=True)
    return nansum(axy / ax2, dim=dim, keepdim=keepdim)
