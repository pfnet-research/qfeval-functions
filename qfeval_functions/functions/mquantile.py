import math
import typing

import torch

from .apply_for_axis import apply_for_axis

MQuantileAlgorithm = typing.Literal["auto", "sort", "wavelet"]
_MQUANTILE_ALGORITHMS = frozenset(("auto", "sort", "wavelet"))


def _quantile_position(span: int, q: float) -> tuple[int, int, float]:
    pos = q * (span - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return lo, hi, pos - lo


def _prepend_incomplete(
    x: torch.Tensor, completed: torch.Tensor, span: int
) -> torch.Tensor:
    nans = x.new_full((x.shape[0], span - 1), math.nan)
    return torch.cat((nans, completed), dim=1)


def _interpolate_order_statistics(
    lower: torch.Tensor,
    upper: typing.Optional[torch.Tensor],
    frac: float,
) -> torch.Tensor:
    if upper is None:
        # Do not add an unused endpoint multiplied by zero: `0 * inf` is
        # NaN and would corrupt an exact order statistic.
        return lower
    return lower * (1 - frac) + upper * frac


def _mquantile_sort(x: torch.Tensor, span: int, q: float) -> torch.Tensor:
    """Compute moving quantiles by sorting every materialized window."""
    if x.shape[1] < span:
        return torch.full_like(x, math.nan)
    if span == 1 or x.shape[0] == 0:
        return x.clone()

    windows = x.unfold(1, span, 1)
    sorted_windows = windows.sort(dim=-1).values
    lo, hi, frac = _quantile_position(span, q)
    lower = sorted_windows[..., lo]
    upper = None if lo == hi else sorted_windows[..., hi]
    completed = _interpolate_order_statistics(lower, upper, frac)
    completed = completed.masked_fill(windows.isnan().any(dim=-1), math.nan)
    return _prepend_incomplete(x, completed, span)


def _wavelet_order_indices(
    values: torch.Tensor,
    span: int,
    orders: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Answer all fixed-width range selections with a wavelet matrix.

    The wavelet matrix is built one bit level at a time.  Every window's
    left/right query bounds are advanced through that level immediately,
    allowing the prefix counts to be discarded before the next level.
    Consequently the workspace is linear rather than ``O(N log U)``,
    where ``U`` is the number of distinct values.
    """
    ordering_values = values.detach()
    is_nan = ordering_values.isnan()
    ordering_values = torch.where(
        is_nan, torch.zeros_like(ordering_values), ordering_values
    )
    unique_values, sequence = torch.unique(
        ordering_values, sorted=True, return_inverse=True
    )

    window_count = values.shape[0] - span + 1
    starts = torch.arange(window_count, dtype=torch.long, device=values.device)
    query_count = len(orders)
    left = starts.unsqueeze(0).expand(query_count, -1).clone()
    right = left + span
    kth = (
        torch.tensor(orders, dtype=torch.long, device=values.device)
        .unsqueeze(1)
        .expand(-1, window_count)
        .clone()
    )

    permutation = torch.arange(
        values.shape[0], dtype=torch.long, device=values.device
    )
    bit_count = max(1, (unique_values.numel() - 1).bit_length())
    for shift in range(bit_count - 1, -1, -1):
        is_zero = ((sequence >> shift) & 1) == 0
        zero_prefix = torch.cat(
            (
                torch.zeros(1, dtype=torch.long, device=values.device),
                is_zero.to(torch.long).cumsum(dim=0),
            )
        )
        zero_left = zero_prefix[left]
        zero_right = zero_prefix[right]
        zeros_in_range = zero_right - zero_left
        take_one = kth >= zeros_in_range
        zero_count = zero_prefix[-1]
        left = torch.where(take_one, zero_count + left - zero_left, zero_left)
        right = torch.where(
            take_one, zero_count + right - zero_right, zero_right
        )
        kth = torch.where(take_one, kth - zeros_in_range, kth)

        permutation = torch.cat((permutation[is_zero], permutation[~is_zero]))
        sequence = torch.cat((sequence[is_zero], sequence[~is_zero]))

    selected = permutation[left + kth]
    nan_prefix = torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=values.device),
            is_nan.to(torch.long).cumsum(dim=0),
        )
    )
    invalid = (nan_prefix[span:] - nan_prefix[:-span]) != 0
    return selected, invalid


def _mquantile_wavelet(x: torch.Tensor, span: int, q: float) -> torch.Tensor:
    """Compute moving quantiles with batched wavelet-matrix queries."""
    if x.shape[1] < span:
        return torch.full_like(x, math.nan)
    if span == 1 or x.shape[0] == 0:
        return x.clone()

    lo, hi, frac = _quantile_position(span, q)
    orders = (lo,) if lo == hi else (lo, hi)
    rows: list[torch.Tensor] = []
    for values in x:
        indices, invalid = _wavelet_order_indices(values, span, orders)
        selected = values[indices]
        lower = selected[0]
        upper = None if lo == hi else selected[1]
        completed = _interpolate_order_statistics(lower, upper, frac)
        rows.append(completed.masked_fill(invalid, math.nan))

    return _prepend_incomplete(x, torch.stack(rows), span)


def _choose_mquantile_algorithm(
    x: torch.Tensor, span: int
) -> MQuantileAlgorithm:
    """Choose an implementation without changing numerical semantics."""
    window_count = max(0, x.shape[1] - span + 1)
    # Sorting is highly vectorized and wins for narrow windows.  It also
    # avoids wavelet-matrix setup when only a handful of windows exist.
    # Benchmarks in `benchmarks/mquantile_results.md` show the crossover
    # near span=256 on CPU for both one and multiple slices.
    if span < 256 or window_count <= 8:
        return "sort"
    return "wavelet"


def _mquantile(
    x: torch.Tensor,
    span: int,
    q: float,
    algorithm: MQuantileAlgorithm,
) -> torch.Tensor:
    selected_algorithm = (
        _choose_mquantile_algorithm(x, span)
        if algorithm == "auto"
        else algorithm
    )
    if selected_algorithm == "sort":
        return _mquantile_sort(x, span, q)
    if selected_algorithm == "wavelet":
        return _mquantile_wavelet(x, span, q)
    raise AssertionError(f"Unhandled mquantile algorithm: {selected_algorithm}")


def mquantile(
    x: torch.Tensor,
    span: int,
    q: float,
    dim: int = -1,
    *,
    algorithm: MQuantileAlgorithm = "auto",
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
        algorithm ({"auto", "sort", "wavelet"}, optional):
            The implementation to use. ``"sort"`` is the original
            all-window sort, while ``"wavelet"`` uses batched
            range-selection queries in a wavelet matrix. ``"auto"``
            (default) chooses based on the input size and :attr:`span`.

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
        ValueError: If ``algorithm`` is not a supported implementation.

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
        The ``"wavelet"`` algorithm avoids the original
        ``O(N * span)`` all-window materialization and takes
        ``O(N * log(N) + N * log(U))`` time including coordinate
        compression, with linear workspace per slice, where ``U`` is the
        number of distinct values.

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
    if algorithm not in _MQUANTILE_ALGORITHMS:
        raise ValueError(
            "algorithm must be one of "
            f"{sorted(_MQUANTILE_ALGORITHMS)}, but got {algorithm!r}."
        )
    return apply_for_axis(lambda x: _mquantile(x, span, q, algorithm), x, dim)
