import typing

import torch

from .mulmean import mulmean


def slope(
    x: torch.Tensor,
    y: torch.Tensor,
    dim: typing.Union[int, typing.Tuple[int, ...]] = (),
    keepdim: bool = False,
) -> torch.Tensor:
    r"""Returns the slope of simple linear regression of ``y`` on ``x`` over the
    given dimension ``dim``.

    This function computes the ordinary least squares slope coefficient
    :math:`\beta` of the regression :math:`y = \alpha + \beta x`:

    .. math::
        \beta = \frac{\sum_i (x_i - \bar{x})(y_i - \bar{y})}
        {\sum_i (x_i - \bar{x})^2}

    In quantitative finance, this is used, for example, to estimate the
    beta of an asset's returns against market returns.

    NOTE: This is based on calculation of slope beta at:
    https://en.wikipedia.org/wiki/Simple_linear_regression

    Args:
        x (Tensor):
            The input tensor of explanatory (independent) values.
        y (Tensor):
            The input tensor of response (dependent) values.  Its shape
            must be broadcastable to the shape of ``x``.
        dim (int or tuple of ints, optional):
            The dimension or dimensions to reduce.
            Default is ``()`` (reduce over all dimensions).
        keepdim (bool, optional):
            Whether the output tensor has ``dim`` retained or not.
            Default is ``False``.

    Returns:
        Tensor:
            The slope coefficient of the linear regression over the
            specified dimensions.

    Example:

        >>> x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        >>> y = torch.tensor([2.0, 4.0, 6.0, 8.0])
        >>> QF.slope(x, y)
        tensor(2.)

        >>> y = torch.tensor([1.0, 3.0, 2.0, 5.0])
        >>> QF.slope(x, y)
        tensor(1.1000)

    .. note::
        If all :attr:`x` values are identical (zero variance), the slope is
        mathematically undefined and the result is NaN.

    .. seealso::
        - :func:`nanslope`: NaN-aware slope function.
        - :func:`correl`: Pearson correlation function.
        - :func:`covar`: Covariance function.
    """
    ax = x - x.mean(dim=dim, keepdim=True)
    ay = y - y.mean(dim=dim, keepdim=True)
    axy = mulmean(ax, ay, dim=dim, keepdim=True)
    ax2 = (ax**2).mean(dim=dim, keepdim=True)
    result: torch.Tensor = (axy / ax2).sum(dim=dim, keepdim=keepdim)
    return result
