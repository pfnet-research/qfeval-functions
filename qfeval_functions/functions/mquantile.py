import math
import typing

import torch
import torch.nn.functional as F

from ._moving_order import _apply_partition
from ._moving_order import _coordinate_ranks
from ._moving_order import _stable_partition_destination
from ._moving_order import _window_has_nan
from .apply_for_axis import apply_for_axis
from .mmax import mmax

MQuantileAlgorithm = typing.Literal["auto", "sort", "select", "wavelet"]
_MQUANTILE_ALGORITHMS = frozenset(("auto", "sort", "select", "wavelet"))


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
    completed = completed.masked_fill(_window_has_nan(x, span), math.nan)
    return _prepend_incomplete(x, completed, span)


def _mquantile_select(x: torch.Tensor, span: int, q: float) -> torch.Tensor:
    """Compute moving quantiles with one or two order-statistic selections."""
    if x.shape[1] < span:
        return torch.full_like(x, math.nan)
    if span == 1 or x.shape[0] == 0:
        return x.clone()

    windows = x.unfold(1, span, 1)
    lo, hi, frac = _quantile_position(span, q)
    lower = windows.kthvalue(lo + 1, dim=-1).values
    upper = None if lo == hi else windows.kthvalue(hi + 1, dim=-1).values
    completed = _interpolate_order_statistics(lower, upper, frac)
    completed = completed.masked_fill(_window_has_nan(x, span), math.nan)
    return _prepend_incomplete(x, completed, span)


def _mquantile_extremum(
    x: torch.Tensor, span: int, largest: bool
) -> torch.Tensor:
    """Compute the exact endpoint quantiles with a linear moving extremum."""
    if x.shape[1] < span:
        return torch.full_like(x, math.nan)
    if span == 1 or x.shape[0] == 0:
        return x.clone()

    moving = mmax(x, span, dim=1) if largest else -mmax(-x, span, dim=1)
    return _prepend_incomplete(x, moving[:, span - 1 :], span)


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
    sequence = _coordinate_ranks(ordering_values)

    batch_size, length = values.shape
    window_count = length - span + 1
    starts = torch.arange(
        window_count, dtype=torch.long, device=values.device
    ).view(1, 1, window_count)
    query_count = len(orders)
    left = starts.expand(batch_size, query_count, window_count).clone()
    right = left + span
    kth = (
        torch.tensor(orders, dtype=torch.long, device=values.device)
        .view(1, query_count, 1)
        .expand(batch_size, query_count, window_count)
        .clone()
    )

    permutation = (
        torch.arange(length, dtype=torch.long, device=values.device)
        .view(1, length)
        .expand(batch_size, length)
        .clone()
    )
    bit_count = max(1, (length - 1).bit_length())
    for shift in range(bit_count - 1, -1, -1):
        is_zero = ((sequence >> shift) & 1) == 0
        zero_prefix = F.pad(
            is_zero.to(torch.long).cumsum(dim=1),
            (1, 0),
        )
        zero_left = zero_prefix.gather(
            1, left.reshape(batch_size, -1)
        ).reshape_as(left)
        zero_right = zero_prefix.gather(
            1, right.reshape(batch_size, -1)
        ).reshape_as(right)
        zeros_in_range = zero_right - zero_left
        take_one = kth >= zeros_in_range
        zero_count = zero_prefix[:, -1:].view(batch_size, 1, 1)
        left = torch.where(take_one, zero_count + left - zero_left, zero_left)
        right = torch.where(
            take_one, zero_count + right - zero_right, zero_right
        )
        kth = torch.where(take_one, kth - zeros_in_range, kth)

        destination = _stable_partition_destination(is_zero)
        permutation = _apply_partition(permutation, destination)
        sequence = _apply_partition(sequence, destination)

    selected = permutation.gather(
        1, (left + kth).reshape(batch_size, -1)
    ).reshape(batch_size, query_count, window_count)
    nan_prefix = F.pad(
        is_nan.to(torch.long).cumsum(dim=1),
        (1, 0),
    )
    invalid = (nan_prefix[:, span:] - nan_prefix[:, :-span]) != 0
    return selected, invalid


def _mquantile_wavelet(x: torch.Tensor, span: int, q: float) -> torch.Tensor:
    """Compute moving quantiles with batched wavelet-matrix queries."""
    if x.shape[1] < span:
        return torch.full_like(x, math.nan)
    if span == 1 or x.shape[0] == 0:
        return x.clone()

    lo, hi, frac = _quantile_position(span, q)
    orders = (lo,) if lo == hi else (lo, hi)
    indices, invalid = _wavelet_order_indices(x, span, orders)
    selected = x.gather(1, indices.flatten(1)).reshape_as(indices)
    lower = selected[:, 0]
    upper = None if lo == hi else selected[:, 1]
    completed = _interpolate_order_statistics(lower, upper, frac)
    completed = completed.masked_fill(invalid, math.nan)
    return _prepend_incomplete(x, completed, span)


def _choose_mquantile_algorithm(
    x: torch.Tensor, span: int
) -> MQuantileAlgorithm:
    """Choose an implementation without changing numerical semantics."""
    window_count = max(0, x.shape[1] - span + 1)
    # Sorting is highly vectorized and wins for narrow windows.  It also
    # avoids wavelet-matrix setup when only a handful of windows exist.
    # Benchmarks in `benchmarks/mquantile_results.md` show the crossover
    # near span=128 on CPU after vectorizing across slices.
    if span < 128 or window_count <= 8:
        return "sort"
    return "wavelet"


def _mquantile(
    x: torch.Tensor,
    span: int,
    q: float,
    algorithm: MQuantileAlgorithm,
) -> torch.Tensor:
    if algorithm == "auto":
        if q == 0.0 or q == 1.0:
            return _mquantile_extremum(x, span, largest=q == 1.0)
        lo, hi, _ = _quantile_position(span, q)
        if span < 128 and lo == hi:
            return _mquantile_select(x, span, q)

    selected_algorithm = (
        _choose_mquantile_algorithm(x, span)
        if algorithm == "auto"
        else algorithm
    )
    if selected_algorithm == "sort":
        return _mquantile_sort(x, span, q)
    if selected_algorithm == "select":
        return _mquantile_select(x, span, q)
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
        algorithm ({"auto", "sort", "select", "wavelet"}, optional):
            The implementation to use. ``"sort"`` is the original
            all-window sort, ``"select"`` uses one or two ``kthvalue``
            operations per window, and ``"wavelet"`` uses batched range
            selection in a wavelet matrix. ``"auto"`` (default) also
            specializes endpoint quantiles as linear moving extrema.

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
        compression, with linear workspace, where ``U`` is the number of
        distinct values. Endpoint quantiles use an ``O(N)`` algorithm.

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
