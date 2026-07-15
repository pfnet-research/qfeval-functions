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
    r"""Compute the Rank Correlation Index (RCI) along the specified
    dimension.

    RCI is a technical indicator that measures the Spearman rank correlation
    between time order and price rank over a trailing window of ``period``
    elements, expressed as a percentage.  It ranges from :math:`-100` to
    :math:`100`: values close to :math:`100` indicate that prices have been
    consistently rising within the window, and values close to :math:`-100`
    indicate that prices have been consistently falling.

    The mathematical formulation is:

    .. math::
        \text{RCI} = \left(1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}\right)
        \times 100

    where :math:`n` is ``period`` and :math:`d_i` is the difference between
    the time rank and the price rank of the :math:`i`-th element in the
    window.

    The first ``period - 1`` elements along the dimension are filled with NaN
    because they do not have enough preceding elements.

    Reference:
        https://kabu.com/investment/guide/technical/14.html

    Args:
        x (Tensor):
            The input tensor containing prices.
        period (int, optional):
            The window size used to compute rank correlation.
            Default is 9.
        dim (int, optional):
            The dimension along which to compute RCI.
            Default is -1 (the last dimension).

    Returns:
        Tensor:
            A tensor of the same shape as the input, containing RCI values
            in :math:`[-100, 100]` (the first ``period - 1`` elements along
            the dimension are NaN).

    Example:

        >>> x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        >>> QF.rci(x, period=3)
        tensor([  nan,   nan,  100.,  100.,  100.,   50., -100., -100., -100.])

    .. warning::
        This function is not NaN-aware: NaN values participate in the
        internal ranking, so windows containing NaN produce unreliable RCI
        values rather than NaN.

    .. seealso::
        - :func:`rsi`: Relative Strength Index, another momentum indicator.
        - :func:`bollinger_band`: Bollinger Bands indicator.
    """
    return apply_for_axis(lambda x: _rci(x, period), x, dim)
