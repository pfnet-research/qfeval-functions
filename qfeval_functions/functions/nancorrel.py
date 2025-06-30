import math
import typing

import torch

from .nanmean import nanmean
from .nanmulmean import nanmulmean
from .nansum import nansum


def nancorrel(
    x: torch.Tensor,
    y: torch.Tensor,
    dim: typing.Union[int, typing.Tuple[int, ...]] = (),
    keepdim: bool = False,
) -> torch.Tensor:
    r"""Compute Pearson correlation coefficient between two tensors, ignoring
    NaN values.

    This function calculates the Pearson correlation coefficient between
    tensors :attr:`x` and :attr:`y` along the specified dimension(s), excluding
    any pairs where either value is NaN. The correlation coefficient measures
    the linear relationship between two variables and ranges from -1 (perfect
    negative correlation) to 1 (perfect positive correlation), with 0 indicating
    no linear correlation.

    Unlike :func:`correl`, this function handles missing data by ignoring
    NaN values in the computation. If either :attr:`x` or :attr:`y` has a NaN
    at a given position, that pair is excluded from the correlation calculation.

    The NaN-aware Pearson correlation is computed as:

    .. math::
        r = \frac{\text{E}[(X - \mu_X)(Y - \mu_Y)]}{
                \sqrt{\text{E}[(X - \mu_X)^2]}\sqrt{\text{E}[(Y - \mu_Y)^2]}}

    where the expectations are computed only over valid (non-NaN) pairs.

    Args:
        x (Tensor):
            The first input tensor.
        y (Tensor):
            The second input tensor. Must be the same shape as :attr:`x`.
        dim (int or tuple of ints, optional):
            The dimension(s) along which to compute the correlation. If not
            specified (default is empty tuple), computes element-wise
            correlation and sums the result.
        keepdim (bool, optional):
            Whether the output tensor has :attr:`dim` retained or not.
            Default is False.

    Returns:
        Tensor:
            The Pearson correlation coefficient(s), computed only over
            valid (non-NaN) pairs. The shape depends on the input dimensions
            and the :attr:`keepdim` parameter.

    Example:

        >>> # Perfect positive correlation with some NaNs
        >>> x = torch.tensor([1.0, 2.0, nan, 4.0, 5.0])
        >>> y = torch.tensor([2.0, 4.0, 6.0, 8.0, nan])
        >>> QF.nancorrel(x, y, dim=0)
        tensor(1.)

        >>> # 2D tensors with NaN values
        >>> x = torch.tensor([[1.0, nan, 3.0],
        ...                   [4.0, 5.0, nan]])
        >>> y = torch.tensor([[2.0, 4.0, 5.0],
        ...                   [nan, 10.0, 12.0]])
        >>> QF.nancorrel(x, y, dim=1)
        tensor([1., nan])

        >>> # Comparison with regular correl (which would give NaN)
        >>> x_with_nan = torch.tensor([1.0, 2.0, nan, 4.0])
        >>> y_with_nan = torch.tensor([2.0, 4.0, 6.0, 8.0])
        >>> QF.nancorrel(x_with_nan, y_with_nan, dim=0)  # Ignores NaN
        tensor(1.)

        >>> # With keepdim
        >>> x = torch.tensor([[1.0, nan, 3.0],
        ...                   [4.0, 5.0, 6.0]])
        >>> y = torch.tensor([[2.0, 4.0, 6.0],
        ...                   [8.0, 10.0, 12.0]])
        >>> QF.nancorrel(x, y, dim=1, keepdim=True)
        tensor([[1.],
                [1.]])

    .. seealso::
        - :func:`correl`: Pearson correlation without NaN handling.
        - :func:`nancovar`: NaN-aware covariance function.
        - :func:`nanmean`: NaN-aware mean function.
    """
    isnan = x.isnan() | y.isnan()
    x = torch.where(isnan, torch.as_tensor(math.nan).to(x), x)
    y = torch.where(isnan, torch.as_tensor(math.nan).to(y), y)
    ax = x - nanmean(x, dim=dim, keepdim=True)
    ay = y - nanmean(y, dim=dim, keepdim=True)
    axy = nanmulmean(ax, ay, dim=dim, keepdim=True)
    ax2 = nanmean(ax**2, dim=dim, keepdim=True)
    ay2 = nanmean(ay**2, dim=dim, keepdim=True)
    result = axy / ax2.sqrt() / ay2.sqrt()
    return nansum(result, dim=dim, keepdim=keepdim)
