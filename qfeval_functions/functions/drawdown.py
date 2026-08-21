import torch


def drawdown(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    r"""Compute the drawdown series from the running peak along the
    specified dimension.

    Drawdown measures the relative loss from the highest value observed so
    far, which is widely used to assess the downside risk of a price or
    equity curve:

    .. math::
        \text{DD}[i] = \frac{x[i]}{\max_{j \le i} x[j]} - 1

    For a positive price series the result is non-positive: it is 0 at the
    first element and whenever a new running high is reached, and negative
    while the series is below its running peak.  This function assumes
    strictly positive prices; ratios are meaningless for non-positive
    values.

    Args:
        x (Tensor):
            The input tensor containing strictly positive prices.
        dim (int, optional):
            The dimension along which to compute the drawdown.
            Default is -1 (the last dimension).

    Returns:
        Tensor:
            A tensor of the same shape as the input, containing the
            drawdown values (non-positive for positive price series).

    Example:

        >>> x = torch.tensor([100.0, 120.0, 90.0, 130.0, 65.0])
        >>> QF.drawdown(x)
        tensor([ 0.0000,  0.0000, -0.2500,  0.0000, -0.5000])

        >>> # 2D example: each row is an independent series.
        >>> x = torch.tensor([[100.0, 120.0, 90.0, 130.0, 65.0],
        ...                   [10.0, 20.0, 30.0, 40.0, 50.0]])
        >>> QF.drawdown(x, dim=1)
        tensor([[ 0.0000,  0.0000, -0.2500,  0.0000, -0.5000],
                [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000]])

    .. note::
        NaN is sticky: ``torch.cummax`` propagates NaN, so every output at
        or after the first NaN along the dimension is NaN.  A ``+inf``
        price makes subsequent drawdowns relative to ``inf`` (``-1`` or
        NaN per IEEE 754).

    .. seealso::
        - :func:`max_drawdown`: The most negative drawdown of the series.
        - :func:`rcummax`: Reverse cumulative maximum (not used here; this
          function uses the forward cumulative maximum).
        - ``torch.cummax``: Standard (forward) cumulative maximum.
    """
    return x / torch.cummax(x, dim=dim).values - 1
