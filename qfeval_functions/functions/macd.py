import typing

import torch

from .ema import ema


def macd(
    x: torch.Tensor,
    fast_span: int = 12,
    slow_span: int = 26,
    signal_span: int = 9,
    dim: int = -1,
) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Compute the Moving Average Convergence Divergence (MACD) along the
    specified dimension.

    MACD is a trend-following momentum indicator built from the difference
    of a fast and a slow exponential moving average (EMA).  A signal line
    (an EMA of the MACD line) and a histogram (the difference between the
    two) are commonly used to detect trend changes and momentum shifts:

    .. math::
        \begin{aligned}
        \text{MACD}[i] &= \text{EMA}_{\text{fast}}(x)[i]
            - \text{EMA}_{\text{slow}}(x)[i] \\
        \text{Signal}[i] &= \text{EMA}_{\text{signal}}(\text{MACD})[i] \\
        \text{Histogram}[i] &= \text{MACD}[i] - \text{Signal}[i]
        \end{aligned}

    where :math:`\text{EMA}_n` denotes :func:`ema` with the standard
    span-to-alpha mapping :math:`\alpha_n = 2 / (n + 1)`.

    Args:
        x (Tensor):
            The input tensor containing prices.
        fast_span (int, optional):
            The span of the fast EMA.  Must be positive and less than
            ``slow_span``.  Default is 12.
        slow_span (int, optional):
            The span of the slow EMA.  Must be positive.
            Default is 26.
        signal_span (int, optional):
            The span of the EMA applied to the MACD line to obtain the
            signal line.  Must be positive.  Default is 9.
        dim (int, optional):
            The dimension along which to compute MACD.
            Default is -1 (the last dimension).

    Returns:
        Tuple[Tensor, Tensor, Tensor]:
            A tuple containing three tensors of the same shape as the input:

            - MACD line: fast EMA - slow EMA
            - Signal line: EMA of the MACD line
            - Histogram: MACD line - signal line

    Raises:
        ValueError: If a span is not positive, or if ``fast_span`` is not
            less than ``slow_span``.
        TypeError: If a span is not an integer (``bool`` is rejected).

    Example:

        >>> x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        >>> macd_line, signal_line, histogram = QF.macd(
        ...     x, fast_span=3, slow_span=6, signal_span=4)
        >>> macd_line
        tensor([0.0000, 0.0833, 0.2084, 0.3590, 0.5193, 0.6763])
        >>> signal_line
        tensor([0.0000, 0.0521, 0.1318, 0.2362, 0.3590, 0.4921])
        >>> histogram
        tensor([0.0000, 0.0312, 0.0766, 0.1228, 0.1603, 0.1842])

    .. note::
        The EMAs are computed by :func:`ema`, whose weighting is equivalent
        to ``pandas.DataFrame.ewm(span=n, adjust=True).mean()``.  This
        differs from TA-Lib, which uses a recursive EMA seeded with a
        simple moving average; the two conventions converge for long
        series.  Consequently, there is no warm-up NaN prefix: outputs are
        defined from the first element, whereas TA-Lib pads the lookback
        period with NaN.

    .. note::
        Like :func:`ema`, a NaN in the input contaminates all outputs at
        and after its position along the dimension.

    .. seealso::
        - :func:`ema`: Exponential moving average function used for all
          three lines.
        - :func:`rsi`: Relative Strength Index, another momentum indicator.
        - :func:`bollinger_band`: Another indicator returning multiple
          series.
    """
    # NOTE: bool is a subclass of int, so it must be rejected explicitly.
    for name, span in (
        ("fast_span", fast_span),
        ("slow_span", slow_span),
        ("signal_span", signal_span),
    ):
        if isinstance(span, bool) or not isinstance(span, int):
            raise TypeError(f"{name} must be an integer, but got {span!r}.")
        if span <= 0:
            raise ValueError(
                f"{name} must be a positive integer, but got {span}."
            )
    if fast_span >= slow_span:
        raise ValueError(
            f"fast_span must be less than slow_span, but got "
            f"fast_span={fast_span} and slow_span={slow_span}."
        )
    macd_line = ema(x, 2 / (fast_span + 1), dim=dim) - ema(
        x, 2 / (slow_span + 1), dim=dim
    )
    signal_line = ema(macd_line, 2 / (signal_span + 1), dim=dim)
    return macd_line, signal_line, macd_line - signal_line
