import typing

import torch

from .mulmean import mulmean


def slope(
    x: torch.Tensor,
    y: torch.Tensor,
    dim: typing.Union[int, typing.Tuple[int, ...]] = (),
    keepdim: bool = False,
) -> torch.Tensor:
    r"""Compute the slope of linear regression between two tensors.

    This function computes the slope coefficient (beta) of the linear regression
    line that best fits the relationship between tensors :attr:`x` and :attr:`y`.
    The slope represents the rate of change in :attr:`y` with respect to changes
    in :attr:`x`, providing insight into the linear relationship between variables.

    The slope is calculated using the least squares method:

    .. math::
        \beta = \frac{\text{Cov}(x, y)}{\text{Var}(x)} = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}

    where :math:`\bar{x}` and :math:`\bar{y}` are the means of :attr:`x` and :attr:`y` respectively.

    Args:
        x (Tensor):
            The independent variable tensor. Must have the same shape as :attr:`y`.
        y (Tensor):
            The dependent variable tensor. Must have the same shape as :attr:`x`.
        dim (int or tuple of ints, optional):
            The dimension or dimensions along which to compute the slope.
            If empty tuple (default), computes slope over all elements.
        keepdim (bool, optional):
            Whether the output tensor retains the specified dimensions.
            Default is False.

    Returns:
        Tensor:
            The slope coefficients. If :attr:`keepdim` is True, the output tensor
            has the same number of dimensions as the input with the reduced
            dimensions having size 1.

    Example:
        >>> # Simple linear relationship
        >>> x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> y = torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0])  # y = 2x
        >>> slope(x, y)
        tensor(2.0000)

        >>> # Negative slope
        >>> x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        >>> y = torch.tensor([10.0, 8.0, 6.0, 4.0])  # y = -2x + 12
        >>> slope(x, y)
        tensor(-2.0000)

        >>> # 2D example with different dimensions
        >>> x = torch.tensor([[1.0, 2.0, 3.0],
        ...                   [1.0, 3.0, 5.0]])
        >>> y = torch.tensor([[2.0, 4.0, 6.0],
        ...                   [3.0, 7.0, 11.0]])
        >>> slope(x, y, dim=1)  # Slope along rows
        tensor([2.0000, 2.0000])

        >>> # Multi-dimensional reduction
        >>> x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]],
        ...                   [[2.0, 3.0], [4.0, 5.0]]])
        >>> y = torch.tensor([[[3.0, 6.0], [9.0, 12.0]],
        ...                   [[4.0, 6.0], [8.0, 10.0]]])
        >>> slope(x, y, dim=(1, 2))  # Slope over last two dimensions
        tensor([3.0000, 2.0000])

        >>> # Keep dimensions
        >>> x = torch.tensor([[1.0, 2.0, 3.0]])
        >>> y = torch.tensor([[5.0, 7.0, 9.0]])
        >>> slope(x, y, dim=1, keepdim=True)
        tensor([[2.0000]])

        >>> # Financial example: beta calculation
        >>> market_returns = torch.tensor([0.02, -0.01, 0.03, 0.01, -0.02])
        >>> stock_returns = torch.tensor([0.03, -0.015, 0.045, 0.015, -0.025])
        >>> beta = slope(market_returns, stock_returns)
        >>> beta  # Stock beta relative to market
        tensor(1.4244)

        >>> # Portfolio analysis
        >>> factors = torch.tensor([[0.01, 0.02, -0.01],
        ...                         [0.02, -0.01, 0.03]])
        >>> returns = torch.tensor([[0.015, 0.025, -0.005],
        ...                         [0.030, -0.015, 0.045]])
        >>> factor_loadings = slope(factors, returns, dim=1)
        >>> factor_loadings
        tensor([1.0000, 1.5000])

    See Also:
        :func:`correl`: Pearson correlation coefficient.
        :func:`covar`: Covariance between tensors.
        :func:`mulmean`: Multiplicative mean used in slope calculation.

    .. note::
        The slope function is fundamental in quantitative finance for:
        
        - Computing beta coefficients in factor models (CAPM, Fama-French)
        - Measuring sensitivity of asset returns to market movements
        - Risk attribution and factor exposure analysis
        - Hedge ratio calculation for pairs trading
        - Linear regression analysis in econometric models
        - Performance attribution analysis

    .. note::
        In financial applications, slope is commonly used for:
        
        - Beta calculation: sensitivity of stock returns to market returns
        - Currency hedging: exchange rate sensitivity analysis
        - Interest rate risk: duration and convexity calculations
        - Commodity exposure: price sensitivity to underlying factors
        - Credit risk modeling: default probability factor analysis
        - Volatility modeling: factor loadings in volatility surfaces

    .. note::
        The slope calculation assumes a linear relationship between variables.
        For non-linear relationships, consider using polynomial regression or
        other non-linear modeling techniques. The slope is sensitive to outliers
        and may benefit from robust regression methods in practice.

    .. warning::
        Division by zero occurs when :attr:`x` has zero variance (all values
        are identical). In such cases, the slope is undefined and will result
        in NaN values. Consider preprocessing the data to handle constant
        variables appropriately.

    References:
        - Simple linear regression: https://en.wikipedia.org/wiki/Simple_linear_regression
        - CAPM beta calculation: https://en.wikipedia.org/wiki/Capital_asset_pricing_model
        - Factor models in finance: https://en.wikipedia.org/wiki/Factor_model_(finance)
    """
    ax = x - x.mean(dim=dim, keepdim=True)
    ay = y - y.mean(dim=dim, keepdim=True)
    axy = mulmean(ax, ay, dim=dim, keepdim=True)
    ax2 = (ax**2).mean(dim=dim, keepdim=True)
    result: torch.Tensor = (axy / ax2).sum(dim=dim, keepdim=keepdim)
    return result
