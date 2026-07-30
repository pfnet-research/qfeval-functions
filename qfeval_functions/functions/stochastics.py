import math
import typing

import torch

from .ma import ma
from .mmax import mmax
from .mmin import mmin


def stochastics(
    high: torch.Tensor,
    low: torch.Tensor,
    close: torch.Tensor,
    k_span: int = 14,
    d_span: int = 3,
    sd_span: int = 3,
    dim: int = -1,
) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Compute the Stochastic Oscillator (%K, %D, and Slow %D) along the
    specified dimension.

    The Stochastic Oscillator is a momentum indicator that locates the
    closing price within the trailing high-low range, ranging from 0 to
    100:

    .. math::
        \%K[i] = \frac{\text{close}[i] - \text{LL}[i]}
        {\text{HH}[i] - \text{LL}[i]} \times 100

    where :math:`\text{LL}[i]` and :math:`\text{HH}[i]` are the lowest low
    and the highest high over the trailing window of ``k_span`` elements.
    %D is the simple moving average of %K over ``d_span`` elements, and
    Slow %D is the simple moving average of %D over ``sd_span`` elements:

    .. math::
        \%D = \text{SMA}_{d\_span}(\%K), \quad
        \text{Slow \%D} = \text{SMA}_{sd\_span}(\%D)

    The returned tuple ``(k, d, slow_d)`` contains the fast stochastics
    pair (Fast %K, Fast %D) followed by one more smoothing stage; ``(d,
    slow_d)`` equals the commonly used slow stochastics pair (Slow %K,
    Slow %D).  The three input tensors are broadcast against each other.
    See https://www.investopedia.com/terms/s/stochasticoscillator.asp and
    https://info.monex.co.jp/technical-analysis/indicators/006.html

    Following the pandas rolling convention, positions whose trailing
    window does not fully cover preceding elements are NaN: the first
    ``k_span - 1`` elements of %K, the first ``k_span + d_span - 2``
    elements of %D, and the first ``k_span + d_span + sd_span - 3``
    elements of Slow %D along the dimension.

    If a trailing window contains no price variation at all (its highest
    high equals its lowest low), the raw formula is 0/0, and %K is defined
    to be the neutral value 50.  NOTE: This intentionally differs from
    TA-Lib, which returns 0 for such windows.  Windows whose price range
    is NaN (e.g., caused by NaN inputs or by equally infinite highs and
    lows) are propagated as NaN instead of being converted to a valid
    value.

    NaN values in the input propagate to the output: a NaN price makes the
    %K values whose windows contain it NaN, the NaN region grows through
    the moving averages of %D and Slow %D, and outputs recover once their
    windows no longer contain NaN values.

    Args:
        high (Tensor):
            The input tensor containing high prices.
        low (Tensor):
            The input tensor containing low prices.
        close (Tensor):
            The input tensor containing closing prices.
        k_span (int, optional):
            The window size of the highest high and the lowest low used
            for %K.  Default is 14.
        d_span (int, optional):
            The window size of the simple moving average used for %D.
            Default is 3.
        sd_span (int, optional):
            The window size of the simple moving average used for Slow %D.
            Default is 3.
        dim (int, optional):
            The dimension along which to compute the oscillator.
            Default is -1 (the last dimension).

    Returns:
        Tuple[Tensor, Tensor, Tensor]:
            A tuple ``(k, d, slow_d)`` of tensors of the broadcast input
            shape:

            - ``k``: Fast %K, the raw stochastic value in :math:`[0, 100]`
              (the first ``k_span - 1`` elements are NaN).
            - ``d``: Fast %D (= Slow %K), the moving average of ``k`` over
              ``d_span`` elements.
            - ``slow_d``: Slow %D, the moving average of ``d`` over
              ``sd_span`` elements.

    Raises:
        ValueError: If any of ``k_span``, ``d_span``, and ``sd_span`` is
            not positive.
        TypeError: If any of ``k_span``, ``d_span``, and ``sd_span`` is
            not an integer (``bool`` is rejected).
        TypeError: If any of ``high``, ``low``, and ``close`` is not a
            floating point tensor.

    Example:

        >>> high = torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0])
        >>> low = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
        >>> close = torch.tensor([1.0, 3.0, 4.0, 4.0, 5.0])
        >>> k, d, slow_d = QF.stochastics(
        ...     high, low, close, k_span=3, d_span=2, sd_span=2)
        >>> k
        tensor([ nan,  nan, 100.,  75.,  75.])
        >>> d
        tensor([    nan,     nan,     nan, 87.5000, 75.0000])
        >>> slow_d
        tensor([    nan,     nan,     nan,     nan, 81.2500])

        >>> # A window with no price variation yields the neutral value 50.
        >>> c = torch.full((4,), 5.0)
        >>> QF.stochastics(c, c, c, k_span=2, d_span=2, sd_span=2)[0]
        tensor([nan, 50., 50., 50.])

    .. note::
        The inputs are not validated against the OHLC invariants (such as
        ``high >= low`` and ``low <= close <= high``); each output is
        simply the result of the formulas above for the given inputs.  For
        inputs satisfying the invariants, all outputs lie in
        :math:`[0, 100]`.

    .. seealso::
        - :func:`rsi`: Relative Strength Index, another momentum
          oscillator ranging from 0 to 100.
        - :func:`mmax`: Moving maximum, used for the highest high.
        - :func:`mmin`: Moving minimum, used for the lowest low.
        - :func:`ma`: Simple moving average, used for %D and Slow %D.
    """
    for name, span in (
        ("k_span", k_span),
        ("d_span", d_span),
        ("sd_span", sd_span),
    ):
        # NOTE: bool is a subclass of int, so it must be rejected
        # explicitly.
        if isinstance(span, bool) or not isinstance(span, int):
            raise TypeError(f"{name} must be an integer, but got {span!r}.")
        if span <= 0:
            raise ValueError(
                f"{name} must be a positive integer, but got {span}."
            )
    for name, tensor in (("high", high), ("low", low), ("close", close)):
        if not tensor.is_floating_point():
            raise TypeError(
                f"stochastics only supports floating point tensors, but "
                f"{name} has dtype {tensor.dtype}."
            )
    high, low, close = torch.broadcast_tensors(high, low, close)
    lowest = mmin(low, k_span, dim)
    highest = mmax(high, k_span, dim)
    price_range = highest - lowest
    raw = (close - lowest) / price_range * 100
    # A window with no price variation at all (highest == lowest) yields
    # 0/0.  %K is defined to be the neutral value 50 there.  NOTE: This
    # intentionally differs from TA-Lib, which returns 0 for such windows.
    # A NaN price range (e.g., caused by NaN or infinite inputs) fails the
    # comparison below, so it is propagated as is instead of being
    # converted to a valid value.
    k = torch.where(price_range == 0, raw.new_tensor(50.0), raw)
    # `mmax`/`mmin` use partial windows at the start, but this indicator
    # follows the pandas rolling convention, so mask the first
    # `k_span - 1` positions along `dim` with NaN explicitly.
    index_shape = [1] * k.dim()
    index_shape[dim] = -1
    index = torch.arange(k.shape[dim], device=k.device).reshape(index_shape)
    k = torch.where(index < k_span - 1, k.new_tensor(math.nan), k)
    d = ma(k, d_span, dim)
    slow_d = ma(d, sd_span, dim)
    return k, d, slow_d
