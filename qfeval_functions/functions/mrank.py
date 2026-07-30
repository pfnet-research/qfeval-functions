import math

import torch

from .apply_for_axis import apply_for_axis


def _mrank(x: torch.Tensor, span: int, pct: bool) -> torch.Tensor:
    """Returns the moving rank of the latest element of the given tensor
    ``x``, whose shape is ``(B, N)``, along the 2nd dimension."""

    # 1. A series shorter than the window has no valid output position.
    if x.shape[1] < span:
        return torch.full_like(x, math.nan)

    # 2. Materialize all windows as `(B, N - span + 1, span)` and compare
    # every window element with the window's latest element.
    w = x.unfold(1, span, 1)
    last = w[..., -1:]
    less = (w < last).sum(dim=-1)
    equal = (w == last).sum(dim=-1)

    # 3. The group tied with the latest element occupies the ranks
    # `less + 1, ..., less + equal`, so its average rank is
    # `less + (equal + 1) / 2`.
    r = less.to(x.dtype) + (equal.to(x.dtype) + 1) / 2
    if pct:
        r = r / span

    # 4. A window containing NaN has no well-defined rank.
    r = r.masked_fill(torch.isnan(w).any(dim=-1), math.nan)

    # 5. Prepend NaNs for the first `span - 1` positions.
    nans = x.new_full((x.shape[0], span - 1), math.nan)
    return torch.cat((nans, r), dim=1)


def mrank(
    x: torch.Tensor, span: int, dim: int = -1, pct: bool = True
) -> torch.Tensor:
    r"""Compute the moving (sliding window) rank of the latest element.

    This function calculates the rank of the newest element within its
    trailing window of size :attr:`span` along the specified dimension,
    resolving ties by the average rank of the tied group.  The result is
    compatible with
    ``pandas.DataFrame.rolling(span).rank(method="average", pct=pct)``.
    The output tensor has the same shape as the input tensor.  For
    positions where the sliding window cannot fully cover preceding
    elements (i.e., the first ``span - 1`` elements along the selected
    dimension), the result is ``nan``.

    The moving rank is computed as:

    .. math::
        \text{MRANK}[i] = \#\{j : x[j] < x[i]\}
        + \frac{\#\{j : x[j] = x[i]\} + 1}{2}

    where :math:`j` ranges over the trailing window
    :math:`i - \text{span} + 1 \le j \le i`.  If :attr:`pct` is true, the
    rank is further divided by :math:`\text{span}`.

    Args:
        x (Tensor):
            The input tensor containing values.  Must be a floating point
            tensor.
        span (int):
            The size of the sliding window. Must be positive.
        dim (int, optional):
            The dimension along which to compute the moving rank.
            Default is -1 (the last dimension).
        pct (bool, optional):
            If ``True``, return the percentile rank, i.e., the average
            rank divided by ``span``, which lies in :math:`(0, 1]`.
            If ``False``, return the 1-based average rank, which lies in
            :math:`[1, \text{span}]`.  Default is ``True``.

    Returns:
        Tensor:
            A tensor of the same shape as the input, containing the moving
            rank values.  The first ``span - 1`` elements along the
            specified dimension are ``nan``.

    Raises:
        ValueError: If ``span`` is not positive.
        TypeError: If ``span`` is not an integer (``bool`` is rejected).
        TypeError: If ``x`` is not a floating point tensor.

    Example:

        >>> # Percentile rank of the latest element (default)
        >>> x = torch.tensor([1.0, 2.0, 3.0, 2.0, 1.0])
        >>> QF.mrank(x, span=3)
        tensor([   nan,    nan, 1.0000, 0.5000, 0.3333])

        >>> # 1-based average rank
        >>> QF.mrank(x, span=3, pct=False)
        tensor([   nan,    nan, 3.0000, 1.5000, 1.0000])

        >>> # Ties are resolved by the average rank.
        >>> QF.mrank(torch.full((4,), 2.0), span=2)
        tensor([   nan, 0.7500, 0.7500, 0.7500])

        >>> # 2D tensor with moving rank along rows
        >>> x = torch.tensor([[1.0, 2.0, 3.0],
        ...                   [3.0, 2.0, 1.0]])
        >>> QF.mrank(x, span=2, dim=1)
        tensor([[   nan, 1.0000, 1.0000],
                [   nan, 0.5000, 0.5000]])

    .. note::
        If a window contains any NaN value, the moving rank for that
        window is NaN.  This matches ``pandas.DataFrame.rolling``, whose
        default ``min_periods`` equals the window size.

    .. note::
        The implementation compares every window element with the
        window's latest element, so it takes ``O(N * span)`` time and
        memory, unlike the ``O(N)`` moving aggregations such as
        :func:`msum`.

    .. seealso::
        - :func:`rank`: Cross-sectional rank along an entire dimension.
        - :func:`mmax`: Moving maximum function.
        - :func:`mmin`: Moving minimum function.
    """
    # NOTE: bool is a subclass of int, so it must be rejected explicitly.
    if isinstance(span, bool) or not isinstance(span, int):
        raise TypeError(f"span must be an integer, but got {span!r}.")
    if span <= 0:
        raise ValueError(f"span must be a positive integer, but got {span}.")
    if not x.is_floating_point():
        raise TypeError(
            f"mrank only supports floating point tensors, but got {x.dtype}."
        )
    return apply_for_axis(lambda x: _mrank(x, span, pct), x, dim)
