import torch

from .apply_for_axis import apply_for_axis


def _rci(x: torch.Tensor, period: int) -> torch.Tensor:
    batch_index = torch.arange(x.shape[0], device=x.device)
    time_index = torch.arange(x.shape[1], device=x.device)
    period_index = torch.arange(period, device=x.device)
    prices = x[
        batch_index[:, None, None],
        (time_index[None, :, None] - period_index[None, None, :]).relu(),
    ]  # no meaning for the prices[:, span-1]
    price_rank = (-prices).argsort().argsort()
    d = (period_index[None, None] - price_rank).square().sum(dim=-1)
    denominator = period * (period**2 - 1)
    v: torch.Tensor = (1 - 6 * d / denominator) * 100
    v[:, : period - 1] = torch.nan  # fill first span-1 elements to nan
    return v


def rci(x: torch.Tensor, period: int = 9, dim: int = -1) -> torch.Tensor:
    r"""Compute Rank Correlation Index (RCI) technical indicator.

    The Rank Correlation Index is a momentum oscillator that measures the
    correlation between the rank of current prices and the rank of their
    positions in the time series. RCI values range from -100 to +100, where
    values near +100 indicate strong upward momentum and values near -100
    indicate strong downward momentum.

    The RCI is calculated using Spearman's rank correlation coefficient:

    .. math::
        \text{RCI} = \left(1 - \frac{6 \sum_{i=1}^{n} d_i^2}{n(n^2 - 1)}\right) \times 100

    where :math:`d_i` is the difference between the rank of the price at
    position :math:`i` and the position index :math:`i`, and :math:`n` is
    the period length.

    Args:
        x (Tensor):
            Input tensor containing price data. The RCI is computed along
            the specified dimension.
        period (int, optional):
            The number of periods to use in the RCI calculation. Must be
            greater than 1. Default is 9.
        dim (int, optional):
            The dimension along which to compute the RCI. Default is -1
            (the last dimension).

    Returns:
        Tensor:
            The RCI values with the same shape as the input. The first
            ``period - 1`` values along the specified dimension are set
            to NaN since they cannot be computed.

    Example:
        >>> # Simple price series with clear trend
        >>> from qfeval_functions.functions.rci import rci
        >>> prices = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.5]])
        >>> rci_values = rci(prices, period=5, dim=1)
        >>> rci_values.shape
        torch.Size([1, 10])
        >>> # First 4 values should be NaN
        >>> torch.isnan(rci_values[0, :4]).all()
        tensor(True)

        >>> # Multi-dimensional example
        >>> batch_prices = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0],
        ...                               [5.0, 4.0, 3.0, 2.0, 1.0]],
        ...                              [[2.0, 1.0, 3.0, 5.0, 4.0],
        ...                               [1.0, 3.0, 2.0, 4.0, 5.0]]])
        >>> rci_batch = rci(batch_prices, period=3, dim=-1)
        >>> rci_batch.shape
        torch.Size([2, 2, 5])

        >>> # RCI with different periods
        >>> data = torch.randn(1, 20)
        >>> rci_short = rci(data, period=5)
        >>> rci_long = rci(data, period=14)
        >>> rci_short.shape == rci_long.shape
        True

    See Also:
        :func:`torch.argsort`: Used internally for ranking prices.

    .. note::
        The RCI is commonly used in technical analysis for:

        - Identifying overbought/oversold conditions (RCI > 80 or RCI < -80)
        - Detecting momentum divergences between price and RCI
        - Generating buy/sell signals at extreme RCI levels
        - Confirming trend strength and potential reversals
        - Multi-timeframe analysis with different period settings

    .. note::
        In quantitative finance applications, RCI can be used for:

        - Momentum factor construction in factor models
        - Signal generation in systematic trading strategies
        - Risk management through momentum regime detection
        - Portfolio optimization based on momentum characteristics

    .. warning::
        RCI is a lagging indicator and may generate false signals in
        sideways or choppy markets. Consider combining with other
        technical indicators for better signal quality.

    References:
        - https://kabu.com/investment/guide/technical/14.html
    """
    return apply_for_axis(lambda x: _rci(x, period), x, dim)
