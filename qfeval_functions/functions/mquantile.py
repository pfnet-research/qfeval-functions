import math

import torch

from .apply_for_axis import apply_for_axis


def _mquantile(x: torch.Tensor, span: int, q: float) -> torch.Tensor:
    """Returns the moving quantile of the given tensor ``x``, whose shape is
    ``(B, N)``, along the 2nd dimension."""

    # If the data is shorter than the window, no window is complete.
    if x.shape[1] < span:
        return torch.full_like(x, math.nan)

    # 1. Materialize all windows as a `(B, N - span + 1, span)` view and
    # sort each window (the dominant cost; see the docstring note).
    w = x.unfold(1, span, 1)
    s = w.sort(dim=-1).values

    # 2. Linearly interpolate between the two order statistics adjacent to
    # the quantile position `q * (span - 1)`.
    pos = q * (span - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    frac = pos - lo
    if lo == hi:
        # NOTE: This must not add `s[..., hi] * 0.0`: `0.0 * inf` is NaN,
        # so that would corrupt windows containing infinite values.
        v = s[..., lo]
    else:
        v = s[..., lo] * (1 - frac) + s[..., hi] * frac

    # 3. Sorting moves NaNs to the end of each window, so a window
    # containing NaN would silently yield the quantile of its non-NaN
    # values; mask such windows explicitly instead.
    v = v.masked_fill(w.isnan().any(dim=-1), math.nan)

    # 4. The first `span - 1` positions have no complete window.
    nans = x.new_full((x.shape[0], span - 1), math.nan)
    return torch.cat((nans, v), dim=1)


def mquantile(
    x: torch.Tensor, span: int, q: float, dim: int = -1
) -> torch.Tensor:
    r"""Compute the moving (sliding window) quantile of a tensor.

    This function calculates the :attr:`q`-th quantile of elements within
    a sliding window of size :attr:`span` along the specified dimension.
    The output tensor has the same shape as the input tensor. For
    positions where the sliding window cannot fully cover preceding
    elements (i.e., the first ``span - 1`` elements along the selected
    dimension), the result is ``nan``.

    Letting :math:`s_0 \le s_1 \le \dots \le s_{\text{span}-1}` denote the
    sorted values of a window, the quantile is computed with linear
    interpolation between adjacent order statistics, compatibly with
    ``pandas.DataFrame.rolling(span).quantile(q, interpolation="linear")``:

    .. math::
        \text{MQUANTILE}[i] = (1 - \gamma) \, s_{\lfloor h \rfloor}
        + \gamma \, s_{\lceil h \rceil},
        \quad h = q \, (\text{span} - 1),
        \quad \gamma = h - \lfloor h \rfloor

    In particular, ``q=0.0`` yields the moving minimum, ``q=0.5`` the
    moving median, and ``q=1.0`` the moving maximum.

    Args:
        x (Tensor):
            The input tensor containing values. Must be a floating point
            tensor.
        span (int):
            The size of the sliding window. Must be positive.
        q (float):
            The quantile to compute. Must be in the range ``[0, 1]``.
        dim (int, optional):
            The dimension along which to compute the moving quantile.
            Default is -1 (the last dimension).

    Returns:
        Tensor:
            A tensor of the same shape as the input, containing the moving
            quantile values. The first ``span - 1`` elements along the
            specified dimension are ``nan``.

    Raises:
        TypeError: If ``span`` is not an integer (``bool`` is rejected).
        ValueError: If ``span`` is not positive.
        ValueError: If ``q`` is not in the range ``[0, 1]``.
        TypeError: If ``x`` is not a floating point tensor.

    Example:

        >>> # Moving median (q=0.5) with window size 3
        >>> x = torch.tensor([1.0, 3.0, 2.0, 5.0, 4.0])
        >>> QF.mquantile(x, span=3, q=0.5)
        tensor([nan, nan, 2., 3., 4.])

        >>> # q=0.0 and q=1.0 yield the moving minimum and maximum
        >>> QF.mquantile(x, span=3, q=0.0)
        tensor([nan, nan, 1., 2., 2.])
        >>> QF.mquantile(x, span=3, q=1.0)
        tensor([nan, nan, 3., 5., 5.])

        >>> # Linear interpolation between adjacent order statistics
        >>> QF.mquantile(x, span=2, q=0.25)
        tensor([   nan, 1.5000, 2.2500, 2.7500, 4.2500])

        >>> # 2D tensor with moving quantile along rows
        >>> x = torch.tensor([[1.0, 2.0, 4.0, 8.0],
        ...                   [8.0, 4.0, 2.0, 1.0]])
        >>> QF.mquantile(x, span=2, q=0.5, dim=1)
        tensor([[   nan, 1.5000, 3.0000, 6.0000],
                [   nan, 6.0000, 3.0000, 1.5000]])

        >>> # A window containing NaN yields NaN
        >>> x = torch.tensor([1.0, 2.0, nan, 4.0, 5.0, 6.0])
        >>> QF.mquantile(x, span=2, q=0.5)
        tensor([   nan, 1.5000,    nan,    nan, 4.5000, 5.5000])

    .. note::
        If a window contains any NaN value, the moving quantile for that
        window is NaN. Unlike ``pandas.DataFrame.rolling``, there is no
        ``min_periods``-style option to skip NaN values.

    .. note::
        Infinite values are ordered like ordinary numbers, so e.g.
        ``q=1.0`` returns ``inf`` for windows containing ``inf``, whereas
        pandas treats infinities as missing values in rolling
        aggregations. Interpolating between ``-inf`` and ``inf`` yields
        ``nan``.

    .. note::
        Each window is sorted independently, so this function takes
        ``O(N * span * log(span))`` time and ``O(N * span)`` transient
        memory, in contrast to the ``O(N)`` moving sums and averages
        (:func:`msum`, :func:`ma`); order statistics admit no similarly
        vectorizable ``O(N)`` algorithm.

    .. seealso::
        - :func:`mmedian`: Moving median function (this with ``q=0.5``).
        - :func:`nanquantile`: Quantile over a whole dimension, skipping
          NaNs.
        - :func:`mmax`: Moving maximum function.
        - :func:`mmin`: Moving minimum function.
    """
    # NOTE: bool is a subclass of int, so it must be rejected explicitly.
    if isinstance(span, bool) or not isinstance(span, int):
        raise TypeError(f"span must be an integer, but got {span!r}.")
    if span <= 0:
        raise ValueError(f"span must be a positive integer, but got {span}.")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in the range [0, 1], but got {q}.")
    if not x.is_floating_point():
        raise TypeError(
            "mquantile only supports floating point tensors, "
            f"but got {x.dtype}."
        )
    return apply_for_axis(lambda x: _mquantile(x, span, q), x, dim)
